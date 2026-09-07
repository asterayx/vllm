#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize a torch profiler Chrome trace from a vLLM decode run.

    ./docker/gb10/summarize-profile.py <trace.pt.trace.json[.gz]> [--top 30]
        [--tokens N]

Prints GPU time per category (attention, MoE, mHC, dense GEMM, NCCL, glue),
the top kernels by self GPU time, and, with --tokens, GPU milliseconds per
generated token. Kernels inside CUDA-graph replays are included (CUPTI
records them individually).
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

GPU_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset", "gpu_user_annotation"}

# Order matters: the first group with a matching substring wins.
KERNEL_GROUPS: list[tuple[str, tuple[str, ...]]] = [
    ("nccl", ("nccl",)),
    (
        "attention",
        (
            "sparse_mla",
            "mla",
            "indexer",
            "flash",
            "attn",
            "swa",
            "topk_indices",
            "rope",
            "fp8_einsum",
            "kv_cache",
            "compress",
            "paged",
            "cache_utils",
            "qnorm",
        ),
    ),
    (
        "moe",
        (
            "b12x",
            "moe",
            "expert",
            "topk_softplus",
            "softplus",
            "silu",
            "swiglu",
            "router",
            "gate",
            "cute_dsl",
            "group_gemm",
            "grouped",
        ),
    ),
    ("mhc", ("mhc", "hc_pre", "hc_post", "hc_head", "prenorm", "tilelang")),
    (
        "dense_gemm",
        ("gemm", "cutlass", "humming", "matmul", "gemv", "cublas", "nvjet", "marlin"),
    ),
    ("memcpy", ("memcpy", "memset")),
    ("sampler", ("sampler", "sample", "argmax", "softmax", "logits", "top_p", "top_k")),
]


def categorize(name: str) -> str:
    lowered = name.lower()
    for label, needles in KERNEL_GROUPS:
        if any(needle in lowered for needle in needles):
            return label
    return "glue"


def load_events(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        doc = json.load(handle)
    return doc["traceEvents"] if isinstance(doc, dict) else doc


def summarize(events: list[dict], top: int, tokens: int | None) -> list[str]:
    gpu = [e for e in events if e.get("ph") == "X" and e.get("cat") in GPU_CATEGORIES]
    gpu = [e for e in gpu if e.get("cat") != "gpu_user_annotation"]
    if not gpu:
        return ["no GPU kernel events found (was the trace taken on a worker rank?)"]
    by_name: dict[str, list[float]] = defaultdict(list)
    for e in gpu:
        by_name[e["name"]].append(float(e.get("dur", 0.0)))
    total_us = sum(sum(v) for v in by_name.values())
    first = min(e["ts"] for e in gpu)
    last = max(e["ts"] + e.get("dur", 0.0) for e in gpu)
    wall_us = last - first

    per_group: dict[str, tuple[float, int]] = defaultdict(lambda: (0.0, 0))
    for name, durs in by_name.items():
        group = categorize(name)
        t, n = per_group[group]
        per_group[group] = (t + sum(durs), n + len(durs))

    lines = [
        f"GPU busy {total_us / 1e3:.1f} ms over {wall_us / 1e3:.1f} ms profiled "
        f"({100 * total_us / max(wall_us, 1):.0f}% utilization), "
        f"{len(gpu)} kernel launches",
    ]
    if tokens:
        lines.append(
            f"{total_us / 1e3 / tokens:.2f} GPU ms and {wall_us / 1e3 / tokens:.2f} "
            f"wall ms per generated token ({tokens} tokens)"
        )
    lines.append("")
    lines.append(f"{'category':<12}{'ms':>10}{'share':>8}{'launches':>10}")
    for group, (t, n) in sorted(per_group.items(), key=lambda kv: -kv[1][0]):
        lines.append(f"{group:<12}{t / 1e3:>10.1f}{100 * t / total_us:>7.1f}%{n:>10}")
    lines.append("")
    lines.append(f"top {top} kernels by self GPU time")
    lines.append(f"{'ms':>9}{'share':>8}{'count':>8}{'avg us':>9}  name")
    ranked = sorted(by_name.items(), key=lambda kv: -sum(kv[1]))[:top]
    for name, durs in ranked:
        t = sum(durs)
        lines.append(
            f"{t / 1e3:>9.1f}{100 * t / total_us:>7.1f}%{len(durs):>8}"
            f"{t / len(durs):>9.1f}  [{categorize(name)}] {name[:110]}"
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("trace", type=Path)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--tokens", type=int, default=None)
    args = parser.parse_args(argv)
    print("\n".join(summarize(load_events(args.trace), args.top, args.tokens)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
