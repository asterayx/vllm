#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Codex compatibility proxy in front of a Spark vLLM server.

Codex (``wire_api = "chat"`` or ``"responses"``) reads the user-visible
answer from ``message.content`` / Responses ``output_text``. DeepSeek-V4
often leaves ``content`` empty and puts the answer in ``response_content``
or ``reasoning_content``. This proxy forwards OpenAI-compatible traffic
to vLLM and copies those fields into the Codex-visible slots.

Run on the Spark head (next to Codex), not inside the vLLM container:

    python3 docker/gb10/codex_proxy.py \\
      --upstream http://127.0.0.1:30001 --bind 127.0.0.1 --port 30002

Point Codex ``base_url`` at ``http://127.0.0.1:30002/v1``.
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_REASONING_KEYS = ("reasoning_content", "reasoning")


def _first_text(obj: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = obj.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def apply_codex_message(message: Any) -> Any:
    """Fill empty ``content`` from ``response_content`` then reasoning."""
    if not isinstance(message, dict):
        return message
    if _first_text(message, ("content",)):
        return message
    answer = _first_text(message, ("response_content", "text"))
    if answer is None:
        answer = _first_text(message, _REASONING_KEYS)
    if answer is None:
        return message
    out = dict(message)
    out["content"] = answer
    return out


def transform_chat_completion(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return payload
    out = dict(payload)
    new_choices = []
    for choice in choices:
        if not isinstance(choice, dict):
            new_choices.append(choice)
            continue
        item = dict(choice)
        if "message" in item:
            item["message"] = apply_codex_message(item["message"])
        if "delta" in item:
            item["delta"] = apply_codex_message(item["delta"])
        new_choices.append(item)
    out["choices"] = new_choices
    return out


def _fill_output_text(part: Any) -> Any:
    if not isinstance(part, dict):
        return part
    if part.get("type") not in (None, "output_text", "text"):
        return part
    if _first_text(part, ("text",)):
        return part
    answer = _first_text(part, ("response_content", "content"))
    if answer is None:
        return part
    out = dict(part)
    out["text"] = answer
    return out


def transform_responses(payload: Any) -> Any:
    """Fill empty Responses ``output_text`` from ``response_content``."""
    if not isinstance(payload, dict):
        return payload
    output = payload.get("output")
    if not isinstance(output, list):
        return payload
    out = dict(payload)
    new_output = []
    for item in output:
        if not isinstance(item, dict):
            new_output.append(item)
            continue
        cloned = dict(item)
        content = cloned.get("content")
        if isinstance(content, list):
            cloned["content"] = [_fill_output_text(part) for part in content]
        elif isinstance(content, dict):
            cloned["content"] = apply_codex_message(content)
        if cloned.get("type") == "message" and not cloned.get("content"):
            answer = _first_text(cloned, ("response_content",) + _REASONING_KEYS)
            if answer:
                cloned["content"] = [
                    {"type": "output_text", "text": answer},
                ]
        new_output.append(cloned)
    out["output"] = new_output
    return out


def transform_openai_json(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    if isinstance(payload.get("choices"), list):
        return transform_chat_completion(payload)
    if isinstance(payload.get("output"), list):
        return transform_responses(payload)
    return payload


def transform_sse_block(block: str) -> str:
    """Rewrite one SSE event (possibly several ``data:`` lines)."""
    if not block.strip() or block.strip() == "data: [DONE]":
        return block
    lines = block.splitlines()
    data_lines = [
        line[5:].lstrip() for line in lines if line.startswith("data:")
    ]
    if not data_lines:
        return block
    raw = "\n".join(data_lines)
    if raw.strip() == "[DONE]":
        return block
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return block
    rewritten = json.dumps(
        transform_openai_json(payload), ensure_ascii=False, separators=(",", ":")
    )
    prefix = [line for line in lines if not line.startswith("data:")]
    return "\n".join([*prefix, f"data: {rewritten}"])


def transform_sse_body(body: str) -> str:
    # Keep the trailing delimiter so clients still see the last event.
    parts = body.split("\n\n")
    return "\n\n".join(transform_sse_block(part) for part in parts)


class _ProxyHandler(BaseHTTPRequestHandler):
    upstream: str = "http://127.0.0.1:30001"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write(f"{self.address_string()} - {fmt % args}\n")

    def _forward(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        url = self.upstream.rstrip("/") + self.path
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in {"host", "content-length"}
        }
        req = Request(url, data=body or None, method=self.command, headers=headers)
        try:
            with urlopen(req, timeout=600) as resp:
                raw = resp.read()
                content_type = resp.headers.get("Content-Type", "")
                out = _rewrite_body(raw, content_type)
                self.send_response(resp.status)
                skip = {"transfer-encoding", "content-length", "content-encoding"}
                for key, value in resp.headers.items():
                    if key.lower() not in skip:
                        self.send_header(key, value)
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
        except HTTPError as exc:
            raw = exc.read()
            content_type = exc.headers.get("Content-Type", "") if exc.headers else ""
            out = _rewrite_body(raw, content_type)
            self.send_response(exc.code)
            self.send_header("Content-Type", content_type or "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
        except URLError as exc:
            msg = json.dumps({"error": {"message": str(exc.reason)}}).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)

    def do_GET(self) -> None:  # noqa: N802
        self._forward()

    def do_POST(self) -> None:  # noqa: N802
        self._forward()

    def do_DELETE(self) -> None:  # noqa: N802
        self._forward()


def _rewrite_body(raw: bytes, content_type: str) -> bytes:
    if not raw:
        return raw
    lowered = content_type.lower()
    if "text/event-stream" in lowered:
        return transform_sse_body(raw.decode("utf-8", errors="replace")).encode("utf-8")
    if "json" not in lowered:
        return raw
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    return json.dumps(
        transform_openai_json(payload), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", default="http://127.0.0.1:30001")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30002)
    args = parser.parse_args(argv)
    _ProxyHandler.upstream = args.upstream
    server = ThreadingHTTPServer((args.bind, args.port), _ProxyHandler)
    print(
        f"codex_proxy: {args.bind}:{args.port} -> {args.upstream}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("codex_proxy: stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
