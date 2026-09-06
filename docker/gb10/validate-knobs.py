#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Greedy-output equivalence check for SM12x serving knobs.

The Spark branch ships several kernel-path switches that could not be
validated without the GPU (VLLM_SM12X_BATCHED_DECODE_NEXT_N,
VLLM_SM12X_DECODE_Q_ALIGN_ALLOW_4, VLLM_SM12X_ATTN_AUX_STREAMS,
VLLM_SM12X_SHARED_EXPERTS_STREAM, VLLM_SM12X_SPLIT_IMAGE_PREFILL, ...).
Each must produce the same greedy tokens as the conservative default.

    # baseline server, then:
    ./docker/gb10/validate-knobs.py run --out base.json [--images dir]
    # restart with e.g. VLLM_SM12X_BATCHED_DECODE_NEXT_N=4, then:
    ./docker/gb10/validate-knobs.py run --out batched.json [--images dir]
    ./docker/gb10/validate-knobs.py compare base.json batched.json

`run` sends fixed prompts (short, long, multi-turn, and one per image file)
with temperature 0 and records the generated token ids plus latency.
`compare` reports the first diverging token per prompt and exits non-zero
on any mismatch. Timing columns show the speed effect of the knob.
"""

from __future__ import annotations

import argparse
import binascii
import json
import mimetypes
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

LONG_PARAGRAPH = (
    "The DGX Spark pairs a Grace CPU with a Blackwell GPU in one package. "
    "Its unified memory lets a single node hold a large MoE checkpoint while "
    "two nodes share the tensor-parallel work over a ConnectX link. "
)


def _text_prompts() -> list[list[dict]]:
    short = [{"role": "user", "content": "Reply with the word ready."}]
    medium = [
        {
            "role": "user",
            "content": "Summarize the following in two sentences:\n"
            + LONG_PARAGRAPH * 3,
        }
    ]
    long_ = [
        {
            "role": "user",
            "content": "List three facts from this text:\n" + LONG_PARAGRAPH * 60,
        }
    ]
    multi_turn = [
        {"role": "user", "content": "Name a prime number below ten."},
        {"role": "assistant", "content": "Seven."},
        {"role": "user", "content": "Add three to it and answer with digits."},
    ]
    return [short, medium, long_, multi_turn]


def _image_prompts(images: Path | None) -> list[list[dict]]:
    if images is None:
        return []
    prompts = []
    for path in sorted(images.iterdir()):
        mime = mimetypes.guess_type(path.name)[0]
        if not mime or not mime.startswith("image/"):
            continue
        data = binascii.b2a_base64(path.read_bytes(), newline=False).decode()
        url = f"data:{mime};base64,{data}"
        image = {"type": "image_url", "image_url": {"url": url}}
        prompts.append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image briefly."},
                        image,
                    ],
                }
            ]
        )
        prompts.append(
            [
                {
                    "role": "user",
                    "content": [
                        image,
                        {"type": "text", "text": "What is the dominant color?"},
                    ],
                }
            ]
        )
    return prompts


def _chat(url: str, model: str, messages: list[dict], max_tokens: int) -> dict:
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "seed": 0,
        "max_tokens": max_tokens,
        "logprobs": True,
        "top_logprobs": 1,
        "chat_template_kwargs": {"thinking": False},
    }
    request = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=600) as response:
        payload = json.loads(response.read())
    elapsed = time.perf_counter() - start
    choice = payload["choices"][0]
    tokens = [
        entry["token"] for entry in (choice.get("logprobs") or {}).get("content", [])
    ]
    usage = payload.get("usage", {})
    return {
        "text": choice["message"]["content"],
        "tokens": tokens,
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "seconds": elapsed,
    }


def _model_name(url: str) -> str:
    with urllib.request.urlopen(f"{url}/v1/models", timeout=60) as response:
        return json.loads(response.read())["data"][0]["id"]


def cmd_run(args: argparse.Namespace) -> int:
    try:
        model = args.model or _model_name(args.url)
    except urllib.error.URLError as exc:
        print(
            f"cannot reach {args.url}: {exc.reason}. Start the serve containers "
            "first (docker/gb10/smoke.sh) or pass --url http://<head-ip>:30001.",
            file=sys.stderr,
        )
        return 2
    prompts = _text_prompts() + _image_prompts(args.images)
    results = []
    for index, messages in enumerate(prompts):
        result = _chat(args.url, model, messages, args.max_tokens)
        result["prompt_index"] = index
        results.append(result)
        rate = (result["completion_tokens"] or 0) / max(result["seconds"], 1e-6)
        print(
            f"[{index}] prompt_tokens={result['prompt_tokens']} "
            f"completion_tokens={result['completion_tokens']} "
            f"{result['seconds']:.2f}s {rate:.1f} tok/s",
            flush=True,
        )
    Path(args.out).write_text(
        json.dumps({"model": model, "results": results}, indent=1)
    )
    print("wrote", args.out)
    return 0


def compare_results(base: list[dict], other: list[dict]) -> list[str]:
    """Return one line per prompt; lines starting with MISMATCH differ."""
    lines = []
    for a, b in zip(base, other, strict=True):
        ta, tb = a["tokens"], b["tokens"]
        speed = a["seconds"] / max(b["seconds"], 1e-6)
        if ta == tb:
            lines.append(
                f"OK       prompt {a['prompt_index']}: {len(ta)} tokens, "
                f"{speed:.2f}x faster"
            )
            continue
        first = next(
            (i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb))
        )
        lines.append(
            f"MISMATCH prompt {a['prompt_index']}: diverge at token {first} "
            f"({ta[first : first + 3]!r} vs {tb[first : first + 3]!r}), "
            f"{speed:.2f}x faster"
        )
    return lines


def cmd_compare(args: argparse.Namespace) -> int:
    base = json.loads(Path(args.base).read_text())["results"]
    other = json.loads(Path(args.other).read_text())["results"]
    lines = compare_results(base, other)
    print("\n".join(lines))
    mismatches = sum(line.startswith("MISMATCH") for line in lines)
    print(f"{len(lines) - mismatches}/{len(lines)} prompts identical")
    return 1 if mismatches > args.allow_mismatch else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--url", default="http://127.0.0.1:30001")
    run.add_argument("--model", default=None)
    run.add_argument("--out", required=True)
    run.add_argument("--images", type=Path, default=None)
    run.add_argument("--max-tokens", type=int, default=64)
    run.set_defaults(func=cmd_run)
    compare = sub.add_parser("compare")
    compare.add_argument("base")
    compare.add_argument("other")
    compare.add_argument("--allow-mismatch", type=int, default=0)
    compare.set_defaults(func=cmd_compare)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
