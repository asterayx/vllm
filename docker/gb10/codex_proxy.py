#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Codex proxy: vLLM ``reasoning`` → ``reasoning_content``.

v0.28 chat/Responses emit ``reasoning``. Codex still reads the older
``reasoning_content`` field. This host-side proxy (default ``:30000``)
forwards to vLLM (``:30001``) and copies ``reasoning`` onto
``reasoning_content`` when the latter is missing. SSE is rewritten
line-by-line.

    python3 docker/gb10/codex_proxy.py
    # 0.0.0.0:30000 -> http://127.0.0.1:30001
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def alias(obj: Any) -> None:
    """In-place: copy ``reasoning`` to ``reasoning_content`` if absent."""
    if isinstance(obj, dict):
        if "reasoning" in obj and "reasoning_content" not in obj:
            obj["reasoning_content"] = obj["reasoning"]
        for value in obj.values():
            alias(value)
    elif isinstance(obj, list):
        for item in obj:
            alias(item)


def rewrite_sse(line: str) -> str:
    if not line.startswith("data: "):
        return line
    payload = line[6:].strip()
    if payload in ("[DONE]", ""):
        return line
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        return line
    alias(obj)
    return "data: " + json.dumps(obj, ensure_ascii=False)


def _alias_json_bytes(data: bytes) -> bytes:
    try:
        obj = json.loads(data)
        alias(obj)
        return json.dumps(obj, ensure_ascii=False).encode()
    except Exception:
        return data


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    upstream: str = "http://127.0.0.1:30001"

    def do_GET(self) -> None:  # noqa: N802
        self.fwd()

    def do_POST(self) -> None:  # noqa: N802
        self.fwd()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.fwd()

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} {fmt % args}")

    def fwd(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower()
            not in (
                "host",
                "content-length",
                "connection",
                "transfer-encoding",
            )
        }
        req = Request(
            self.upstream + self.path,
            data=body,
            method=self.command,
            headers=headers,
        )
        try:
            resp = urlopen(req, timeout=600)
        except HTTPError as exc:
            data = _alias_json_bytes(exc.read())
            self.send_response(exc.code)
            self.send_header(
                "Content-Type",
                exc.headers.get("Content-Type", "application/json"),
            )
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)
            return

        content_type = resp.headers.get("Content-Type", "")
        hop = {
            "transfer-encoding",
            "connection",
            "content-length",
            "keep-alive",
        }
        self.send_response(resp.status)
        for key, value in resp.headers.items():
            if key.lower() not in hop:
                self.send_header(key, value)
        self.send_header("Connection", "close")
        if "text/event-stream" in content_type:
            self.end_headers()
            while True:
                raw = resp.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                self.wfile.write((rewrite_sse(line) + "\n").encode("utf-8"))
                self.wfile.flush()
            return
        data = _alias_json_bytes(resp.read())
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", default="http://127.0.0.1:30001")
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=30000)
    args = parser.parse_args(argv)
    _Handler.upstream = args.upstream
    print(
        f"{args.bind}:{args.port} -> {args.upstream}  reasoning => reasoning_content",
        flush=True,
    )
    ThreadingHTTPServer((args.bind, args.port), _Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
