# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Record paired serving timings and bounded, decode-only profiler captures.

Run against an otherwise idle server with the CUDA profiler configured with
delay_iterations=4 and max_iterations=24, under Nsight Systems. Start profiling
only after the first output arrives so long prefill cannot enter the capture.
Raw requests, SSE, metrics and timings are retained for each case.
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18029")
    parser.add_argument("--model", default="qwen38-nvfp4")
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[8192, 65536, 262144, 523000]
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=["warmup", "baseline", "profile", "post"],
        default=["warmup", "baseline", "profile", "post"],
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    def control(path):
        req = urllib.request.Request(args.base_url + path, data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=120) as response:
            return response.read()

    def metrics(path):
        with urllib.request.urlopen(args.base_url + "/metrics", timeout=30) as r:
            path.write_bytes(r.read())

    results = []
    for size in args.sizes:
        unit = "The archive records river levels and weather observations. "
        ids = tokenizer.encode(unit * (size // 8 + 100), add_special_tokens=False)
        prompt = tokenizer.decode(ids[: size - 100])
        prompt += "\nWrite 100 numbered detailed weather station maintenance tips."
        body = dict(
            model=args.model,
            messages=[dict(role="user", content=prompt)],
            max_tokens=256,
            temperature=0,
            seed=42,
            stream=True,
            ignore_eos=True,
            stream_options={"include_usage": True},
            chat_template_kwargs={"enable_thinking": False},
        )
        encoded = json.dumps(body).encode()
        (args.output / f"{size}-request.json").write_bytes(encoded)
        for phase in args.phases:
            prefix = args.output / f"{size}-{phase}"
            metrics(prefix.with_suffix(".metrics-before.txt"))
            first = None
            usage = None
            content = ""
            started = time.monotonic()
            wall = time.time()
            req = urllib.request.Request(
                args.base_url + "/v1/chat/completions",
                encoded,
                {"Content-Type": "application/json"},
            )
            try:
                with (
                    urllib.request.urlopen(req, timeout=1800) as response,
                    prefix.with_suffix(".sse").open("wb") as saved,
                ):
                    for line in response:
                        saved.write(line)
                        if (
                            not line.startswith(b"data: ")
                            or line.strip() == b"data: [DONE]"
                        ):
                            continue
                        event = json.loads(line[6:])
                        if "error" in event:
                            raise RuntimeError(event)
                        usage = event.get("usage") or usage
                        for choice in event.get("choices", []):
                            text = choice.get("delta", {}).get("content") or ""
                            if text and first is None:
                                first = time.monotonic()
                                if phase == "profile":
                                    control("/start_profile")
                            content += text
                ended = time.monotonic()
            finally:
                if phase == "profile":
                    control("/stop_profile")
            if first is None or not usage or not content:
                raise RuntimeError("Missing output or usage")
            result = dict(
                size=size,
                phase=phase,
                started_unix=wall,
                total_s=ended - started,
                ttft_s=first - started,
                usage=usage,
                decode_tokens_s=(usage["completion_tokens"] - 1) / (ended - first),
                output_preview=content[:160],
            )
            results.append(result)
            (args.output / "results.json").write_text(json.dumps(results, indent=2))
            metrics(prefix.with_suffix(".metrics-after.txt"))
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
