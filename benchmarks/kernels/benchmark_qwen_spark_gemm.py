# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare Qwen GB10/TP2 BF16 projections with cold-L2 CUDA-graph CUPTI timing."""

import argparse
import dataclasses
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    SkinnyGemmConfig,
    shape_dynamic_skinny_gemm,
)

# Local TP2 shapes derived from Qwen4Exp config and Linear partition rules.
SHAPES = [
    ("hc_down_inject", 336, 10240),
    ("hc_down_final", 320, 10240),
    ("hc_up", 10240, 320),
    ("gdn_qkvz", 8192, 2560),
    ("attn_out", 2560, 3072),
    ("gdn_ba", 48, 2560),
    ("qsa_qkv_gate", 6656, 2560),
    ("indexer_qk_shared_gate_up", 640, 2560),
    ("shared_down", 2560, 320),
    ("moe_router", 512, 2560),
    ("lm_head", 124160, 2560),
    ("mtp_fc", 1280, 2560),
    ("ple_key", 10240, 2560),
    ("ple_value", 2560, 2560),
]


def main():
    # Fail closed: the event fallback cannot preserve this benchmark's contract.
    from cupti import cupti  # noqa: F401

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 2, 3, 4, 8, 12])
    parser.add_argument("--shapes", nargs="+")
    parser.add_argument("--plans", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    torch.set_num_threads(2)
    records = []
    plans = json.loads(args.plans.read_text()) if args.plans else None
    for name, n, k in SHAPES:
        if args.shapes and name not in args.shapes:
            continue
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        for m in args.rows:
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            reference = F.linear(x.float(), w.float())
            configs = [
                SkinnyGemmConfig(m, b, o, k_unroll=u, vector_width=v)
                for b, o, u, v in [
                    (32, 1, 2, 2),
                    (32, 2, 2, 2),
                    (32, 2, 2, 8),
                    (64, 2, 2, 8),
                    (64, 4, 2, 4),
                    (128, 1, 2, 4),
                    (128, 2, 2, 4),
                    (128, 4, 4, 4),
                ]
                if k % (b * v) == 0 and n % o == 0
            ]
            if plans is not None:
                configs = [
                    SkinnyGemmConfig(**plan["config"])
                    for plan in plans
                    if (plan["m"], plan["n"], plan["k"]) == (m, n, k)
                ]
                if not configs:
                    continue
            candidates = [("torch", None)] + [("cute", c) for c in configs]
            if m % 2 == 0:
                candidates.reverse()
            for backend, config in candidates:

                def run(x=x, w=w, config=config):
                    if config is None:
                        return F.linear(x, w)
                    return shape_dynamic_skinny_gemm(x, w, config)

                record = dict(
                    name=name,
                    m=m,
                    n=n,
                    k=k,
                    backend=backend,
                    config=dataclasses.asdict(config) if config else None,
                )
                try:
                    actual = run().float()
                    # BF16 rounding error against an independent FP32 accumulation.
                    normalized_error = (
                        (actual - reference).norm() / reference.norm()
                    ).item()
                    torch.testing.assert_close(actual, reference, rtol=0.02, atol=0.5)
                    assert normalized_error < 0.004
                    timings = []
                    for _ in range(2):
                        timings.append(
                            statistics.median(
                                bench_gpu_time_with_cupti(
                                    run,
                                    dry_run_iters=3,
                                    repeat_iters=12,
                                    use_cuda_graph=True,
                                    cold_l2_cache=True,
                                )
                            )
                            * 1000
                        )
                    us = statistics.median(timings)
                    record.update(
                        us=us,
                        rounds_us=timings,
                        normalized_error=normalized_error,
                        gbps=2 * (m * k + n * k + m * n) / (us * 1000),
                    )
                except Exception as exc:
                    record["error"] = str(exc)
                records.append(record)
                (args.output / "results.json").write_text(json.dumps(records, indent=2))
                print(json.dumps(record), flush=True)
            del reference
        del w
    (args.output / "metadata.json").write_text(
        json.dumps(
            dict(
                device=torch.cuda.get_device_name(),
                capability=torch.cuda.get_device_capability(),
                torch=torch.__version__,
                cuda=torch.version.cuda,
                timing="CUPTI CUDA graph, cold L2, 2 rounds x 12 repeats",
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
