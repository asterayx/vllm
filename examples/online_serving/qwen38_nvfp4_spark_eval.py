# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small GSM8K chat evaluation for the Spark TP2 launch example."""

import argparse
import asyncio
import importlib.util
import json
import tempfile
import time
from pathlib import Path

import aiohttp


async def main(base_url: str, output: Path):
    root = Path(__file__).resolve().parents[2]
    cache = root / ".run/gsm8k-data"
    cache.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(cache)
    spec = importlib.util.spec_from_file_location(
        "gsm8k_eval", root / "tests/evals/gsm8k/gsm8k_eval.py"
    )
    gsm8k = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gsm8k)
    prompts, labels = gsm8k._build_gsm8k_prompts(16, 5, "")
    semaphore = asyncio.Semaphore(2)
    responses = [None] * len(prompts)
    start = time.perf_counter()
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=900)
    ) as session:

        async def query(index):
            async with semaphore:
                payload = {
                    "model": "qwen38-nvfp4",
                    "messages": [{"role": "user", "content": prompts[index]}],
                    "temperature": 0,
                    "seed": 42,
                    "max_tokens": 1024,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "stop": ["Question", "Assistant:", "<|separator|>"],
                }
                async with session.post(
                    f"{base_url.rstrip('/')}/chat/completions", json=payload
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                responses[index] = result
                print(f"Completed {index + 1}/{len(prompts)}", flush=True)

        await asyncio.gather(*(query(i) for i in range(len(prompts))))
    elapsed = time.perf_counter() - start
    texts = [r["choices"][0]["message"]["content"] or "" for r in responses]
    counts = [r["usage"]["completion_tokens"] for r in responses]
    metrics = gsm8k._score_gsm8k(texts, counts, labels, 5, 1024, elapsed)
    report = {
        "protocol": "GSM8K first 16 test questions, 5-shot, chat, thinking disabled",
        "temperature": 0,
        "seed": 42,
        "concurrency": 2,
        "metrics": metrics,
        "samples": [
            {"index": i, "expected": labels[i], "response": responses[i]}
            for i in range(len(prompts))
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18029/v1")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2] / ".run/gsm8k-16.json",
    )
    args = parser.parse_args()
    asyncio.run(main(args.base_url, args.output))
