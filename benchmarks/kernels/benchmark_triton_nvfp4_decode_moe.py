# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A/B helper for DSv4.1 NVFP4 decode MoE on RTX PRO 5000 (SM120, 110 SMs).

Default serve (Marlin / auto)::

    unset DSV41_EXP_TRITON_MOE
    ./examples/deployment/deepseek_v4_1_flash_rtx_pro_5000.sh

Triton small-M decode (target NVFP4 MoE only; draft DSpark stays MXFP4)::

    DSV41_EXP_TRITON_MOE=1 DSV41_EXP_TRITON_MOE_MAX_M=96 \\
      ./examples/deployment/deepseek_v4_1_flash_rtx_pro_5000.sh

This script always prints the SM120 tile pick and estimated CTA/waves for
M=4/8/16/24 (DSpark k=3 is about 4 rows per request). It does **not** time
Marlin: that needs the RTX box. Optional CUDA path only JIT-warms the Triton
runner on a tiny expert count.

    .venv/bin/python benchmarks/kernels/benchmark_triton_nvfp4_decode_moe.py
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import torch

from vllm.model_executor.layers.fused_moe.triton_nvfp4_decode import (
    DSV41_DECODE_MS,
    DSV41_FLASH_EXPERTS,
    DSV41_FLASH_HIDDEN,
    DSV41_FLASH_INTERMEDIATE_TP4,
    DSV41_FLASH_TOPK,
    SM120_NUM_SMS,
    TritonNvfp4DecodeMoE,
    choose_triton_nvfp4_decode_cfg,
    decode_occupancy_rows,
)


def _print_occupancy() -> dict[str, int]:
    cfg = choose_triton_nvfp4_decode_cfg(
        DSV41_FLASH_HIDDEN,
        DSV41_FLASH_INTERMEDIATE_TP4,
        profile="dsv41_sm120",
        num_sms=SM120_NUM_SMS,
        topk=DSV41_FLASH_TOPK,
        num_experts=DSV41_FLASH_EXPERTS,
    )
    print("DSv4.1-Flash TP4 Triton NVFP4 decode tiles")
    print(
        f"  H={DSV41_FLASH_HIDDEN} I={DSV41_FLASH_INTERMEDIATE_TP4} "
        f"E={DSV41_FLASH_EXPERTS} topk={DSV41_FLASH_TOPK} SMs={SM120_NUM_SMS}"
    )
    print(f"  cfg={cfg}")
    print(
        f"  {'M':>4} {'FC1 CTAs':>10} {'FC1 waves':>10} "
        f"{'FC2 CTAs':>10} {'FC2 waves':>10}"
    )
    for row in decode_occupancy_rows(
        cfg,
        DSV41_FLASH_HIDDEN,
        DSV41_FLASH_INTERMEDIATE_TP4,
        topk=DSV41_FLASH_TOPK,
        num_sms=SM120_NUM_SMS,
        num_experts=DSV41_FLASH_EXPERTS,
        token_counts=DSV41_DECODE_MS,
    ):
        print(
            f"  {int(row['M']):4d} {int(row['fc1_ctas']):10d} "
            f"{row['fc1_waves']:10.2f} {int(row['fc2_ctas']):10d} "
            f"{row['fc2_waves']:10.2f}"
        )
    print(
        "A/B: unset DSV41_EXP_TRITON_MOE for Marlin; set it to 1 for Triton "
        "at M<=DSV41_EXP_TRITON_MOE_MAX_M (b12x above that, not Marlin)."
    )
    print(
        "Limits: target NVFP4 MoE only; DSpark draft experts stay MXFP4. "
        "Do not use --moe-backend b12x as the default serve path."
    )
    return cfg


def _tiny_prepared(e: int, h: int, i: int, device: torch.device):
    c41 = (h // 16 + 3) // 4
    c42 = (i // 16 + 3) // 4
    s1_cols = ((2 * i + 127) // 128 * 128) * c41 * 4
    s2_cols = ((h + 127) // 128 * 128) * c42 * 4
    return SimpleNamespace(
        num_experts=e,
        hidden_size=h,
        intermediate_size=i,
        w13_layout="w13",
        w1_fp4=torch.zeros(e, 2 * i, h // 2, dtype=torch.uint8, device=device),
        w2_fp4=torch.zeros(e, h, i // 2, dtype=torch.uint8, device=device),
        w1_blockscale=torch.zeros(e, s1_cols, dtype=torch.float8_e4m3fn, device=device),
        w2_blockscale=torch.zeros(e, s2_cols, dtype=torch.float8_e4m3fn, device=device),
        w1_alphas=torch.ones(e, dtype=torch.float32, device=device),
        w2_alphas=torch.ones(e, dtype=torch.float32, device=device),
        a1_gscale=torch.ones(e, dtype=torch.float32, device=device),
        a2_gscale=torch.ones(e, dtype=torch.float32, device=device),
    )


def _try_cuda_warmup(cfg: dict[str, int]) -> None:
    if not torch.cuda.is_available():
        print("CUDA: not available; skipped Triton warmup and Marlin timing.")
        return
    device = torch.device("cuda")
    # Tiny E so this is a compile/launch smoke test, not a Flash-sized bench.
    prepared = _tiny_prepared(
        2, DSV41_FLASH_HIDDEN, DSV41_FLASH_INTERMEDIATE_TP4, device
    )
    runner = TritonNvfp4DecodeMoE(
        prepared, topk=DSV41_FLASH_TOPK, max_m=24, cfg=cfg, profile="dsv41_sm120"
    )
    warmed = runner.warmup(DSV41_DECODE_MS, torch.bfloat16)
    print(f"CUDA: warmed Triton NVFP4 decode for M={warmed} (E=2 smoke test).")
    print(
        "CUDA: Marlin comparison is not run here; A/B on the serve box with "
        "the env flag on vs off."
    )


def main() -> None:
    os.environ.setdefault("DSV41_EXP_TRITON_MOE", "0")
    cfg = _print_occupancy()
    _try_cuda_warmup(cfg)


if __name__ == "__main__":
    main()
