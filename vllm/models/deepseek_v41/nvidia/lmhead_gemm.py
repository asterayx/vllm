# SPDX-License-Identifier: Apache-2.0
"""DPSK_LMHEAD_SKINNY: route the DPSK V4.1 bf16 LM head (TP4 shard 32320x5120)
off cuBLAS' cutlass_80 wmma fallback onto the cuteDSL skinny GEMM on sm120.

Measured (RTX PRO 5000 Blackwell, CUDA-graph replay, 3 L2-rotated weights):
M=3..8 cuBLAS ~282us -> skinny ~256us (1.10x); M=12 1.08x; M>=16 no win (kept
on cuBLAS)."""
import logging

import torch
from torch import nn

from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    SkinnyGemmConfig,
    shape_dynamic_skinny_gemm,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)
from vllm.utils.torch_utils import direct_register_custom_op

logger = logging.getLogger(__name__)

_K = 5120
DPSK_LMHEAD_PLANS: dict[tuple[int, int], dict[int, SkinnyGemmConfig]] = {
    (32320, _K): {
        3: SkinnyGemmConfig(3, 32, 1, k_unroll=1, vector_width=4, static_k=_K),
        4: SkinnyGemmConfig(4, 32, 1, k_unroll=1, vector_width=4, static_k=_K),
        6: SkinnyGemmConfig(6, 32, 1, k_unroll=1, vector_width=4, static_k=_K),
        8: SkinnyGemmConfig(8, 32, 2, k_unroll=1, vector_width=4, static_k=_K),
        12: SkinnyGemmConfig(12, 32, 2, k_unroll=1, vector_width=4, static_k=_K),
    },
}


def _ok(x: torch.Tensor, w: torch.Tensor) -> bool:
    return (
        x.dim() == 2
        and x.stride() == (x.shape[1], 1)
        and w.stride() == (w.shape[1], 1)
        and x.dtype == torch.bfloat16
        and w.dtype == torch.bfloat16
        and x.shape[1] == w.shape[1]
    )


def _dpsk_lmhead_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    plan = DPSK_LMHEAD_PLANS.get((weight.shape[0], weight.shape[1]))
    cfg = None if plan is None else plan.get(x.shape[0])
    if cfg is not None and _ok(x, weight):
        return shape_dynamic_skinny_gemm(x, weight, cfg)
    return torch.nn.functional.linear(x, weight)


def _dpsk_lmhead_gemm_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


direct_register_custom_op(
    op_name="dpsk_lmhead_gemm",
    op_func=_dpsk_lmhead_gemm,
    fake_impl=_dpsk_lmhead_gemm_fake,
)


class DpskLmHeadSkinnyMethod(UnquantizedEmbeddingMethod):
    def apply(self, layer, x, bias=None):
        if bias is None:
            return torch.ops.vllm.dpsk_lmhead_gemm(x, layer.weight)
        return super().apply(layer, x, bias)


def enable_dpsk_lmhead_skinny(*roots: nn.Module | None) -> int:
    if not shape_dynamic_skinny_gemm.is_available():
        logger.warning("DPSK_LMHEAD_SKINNY: cuteDSL unavailable, disabled")
        return 0
    n = 0
    seen = set()
    for root in roots:
        if root is None:
            continue
        for m in root.modules():
            if id(m) in seen or not isinstance(m, ParallelLMHead):
                continue
            seen.add(id(m))
            if type(m.quant_method) is not UnquantizedEmbeddingMethod:
                continue
            w = getattr(m, "weight", None)
            if w is None or w.dim() != 2:
                continue
            plan = DPSK_LMHEAD_PLANS.get((w.shape[0], w.shape[1]))
            if plan is None:
                continue
            # Eager JIT compile + correctness check before CUDA graph capture.
            for M, cfg in plan.items():
                x = torch.randn(M, w.shape[1], device=w.device, dtype=w.dtype)
                y = shape_dynamic_skinny_gemm(x, w, cfg)
                ref = torch.nn.functional.linear(x, w)
                err = (y.float() - ref.float()).abs().max().item()
                scale = ref.float().abs().max().item() + 1e-6
                if not err <= 1e-2 * scale:
                    logger.warning(
                        "DPSK_LMHEAD_SKINNY: M=%d mismatch err=%g scale=%g; disabled",
                        M, err, scale)
                    return n
            m.quant_method = DpskLmHeadSkinnyMethod()
            n += 1
    logger.warning("DPSK_LMHEAD_SKINNY: enabled on %d lm_head module(s)", n)
    return n
