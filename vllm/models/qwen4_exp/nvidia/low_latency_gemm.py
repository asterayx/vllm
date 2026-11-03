# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp decode GEMM selection on Hopper and Blackwell.

Dispatch follows Kimi-K3 and uses the local ``(N, K)`` shape and token count.
Plans contain measured CUDA graph capture sizes; other token counts use the
standard linear implementation.
"""

import torch
from torch import nn

import vllm.envs as envs
from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    SkinnyGemmConfig,
    shape_dynamic_skinny_gemm,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

QWEN4_EXP_GEMM_PLANS: dict[tuple[int, int], dict[int, SkinnyGemmConfig]] = {
    # GDN fused QKVZ projection, TP=4.
    (4096, 2560): {
        1: SkinnyGemmConfig(1, 64, 4, k_unroll=4),
        2: SkinnyGemmConfig(2, 64, 4, k_unroll=4),
    },
    # GDN and QSA output projections, TP=4.
    (2560, 1536): {
        1: SkinnyGemmConfig(1, 128, 2, k_unroll=2, vector_width=4),
        2: SkinnyGemmConfig(2, 128, 2, k_unroll=2, vector_width=4),
        4: SkinnyGemmConfig(4, 64, 2, k_unroll=2),
    },
    # GDN fused B/A projection, TP=4.
    (24, 2560): {
        1: SkinnyGemmConfig(1, 128, 2, k_unroll=4, vector_width=4),
        2: SkinnyGemmConfig(2, 128, 2, k_unroll=4, vector_width=4),
        4: SkinnyGemmConfig(4, 128, 2, k_unroll=4, vector_width=4),
        8: SkinnyGemmConfig(8, 128, 1, k_unroll=4, vector_width=4),
        16: SkinnyGemmConfig(16, 128, 1, k_unroll=4, vector_width=4),
    },
    # QSA fused QKV/gate projection, TP=4.
    (3584, 2560): {
        1: SkinnyGemmConfig(1, 128, 4, k_unroll=4, vector_width=4),
        2: SkinnyGemmConfig(2, 64, 2, k_unroll=2),
        4: SkinnyGemmConfig(4, 64, 2, k_unroll=2),
    },
    # QSA indexer Q/K projection, replicated in a TP=4 deployment.
    (640, 2560): {
        1: SkinnyGemmConfig(1, 128, 1, k_unroll=4, vector_width=4),
        2: SkinnyGemmConfig(2, 128, 1, k_unroll=4, vector_width=4),
        4: SkinnyGemmConfig(4, 128, 1, k_unroll=4, vector_width=4),
        8: SkinnyGemmConfig(8, 128, 1, k_unroll=4, vector_width=4),
    },
    # Shared-expert fused gate/up projection, TP=4.
    (320, 2560): {
        1: SkinnyGemmConfig(1, 128, 2, k_unroll=4, vector_width=4),
        2: SkinnyGemmConfig(2, 128, 2, k_unroll=4, vector_width=4),
        4: SkinnyGemmConfig(4, 128, 2, k_unroll=4, vector_width=4),
        8: SkinnyGemmConfig(8, 64, 1, k_unroll=4),
        16: SkinnyGemmConfig(16, 128, 2, k_unroll=4, vector_width=4),
    },
    # LM head, TP=4.
    (62080, 2560): {
        1: SkinnyGemmConfig(1, 64, 4, k_unroll=2),
        2: SkinnyGemmConfig(2, 32, 4, k_unroll=2),
    },
    # HC merged down/injection projection, replicated in a TP=4 deployment.
    (336, 10240): {
        1: SkinnyGemmConfig(1, 128, 1, static_k=10240),
        2: SkinnyGemmConfig(2, 128, 1, static_k=10240),
        4: SkinnyGemmConfig(4, 128, 2, static_k=10240),
        8: SkinnyGemmConfig(8, 128, 1, k_unroll=4),
    },
}

# H200 plans selected by exhaustive CUDA graph replay measurements over
# M={1, 2, 4, 8, 16}. Only points that beat the standard linear implementation
# in both hot-cache and L2-flush measurements are retained; other token counts
# keep the standard implementation and its GEMM heuristics.
QWEN4_EXP_SM90_GEMM_PLANS: dict[tuple[int, int], dict[int, SkinnyGemmConfig]] = {
    # GDN fused QKVZ projection, TP=4.
    (4096, 2560): {
        1: SkinnyGemmConfig(1, 128, 2, vector_width=4, static_k=2560),
        2: SkinnyGemmConfig(2, 64, 4, vector_width=4, static_k=2560),
    },
    # GDN and QSA output projections, TP=4.
    (2560, 1536): {
        1: SkinnyGemmConfig(1, 128, 4, vector_width=2, static_k=1536),
        2: SkinnyGemmConfig(2, 128, 4, vector_width=4, static_k=1536),
        4: SkinnyGemmConfig(4, 64, 4, k_unroll=6, vector_width=4),
    },
    # GDN fused B/A projection, TP=4.
    (24, 2560): {
        1: SkinnyGemmConfig(1, 128, 3, vector_width=4, static_k=2560),
        2: SkinnyGemmConfig(2, 64, 2, vector_width=4, static_k=2560),
        4: SkinnyGemmConfig(4, 64, 1, static_k=2560),
        8: SkinnyGemmConfig(8, 128, 1, vector_width=4, static_k=2560),
        16: SkinnyGemmConfig(16, 128, 1, vector_width=4, static_k=2560),
    },
    # QSA fused QKV/gate projection, TP=4.
    (3584, 2560): {
        1: SkinnyGemmConfig(1, 128, 4, vector_width=2, static_k=2560),
        2: SkinnyGemmConfig(2, 64, 4, k_unroll=5),
    },
    # QSA indexer Q/K projection, replicated in a TP=4 deployment.
    (640, 2560): {
        1: SkinnyGemmConfig(1, 256, 2, vector_width=2, static_k=2560),
        2: SkinnyGemmConfig(2, 128, 2, vector_width=4, static_k=2560),
        4: SkinnyGemmConfig(4, 128, 1, vector_width=2, static_k=2560),
        8: SkinnyGemmConfig(8, 128, 2, vector_width=4, static_k=2560),
    },
    # Shared-expert fused gate/up projection, TP=4.
    (320, 2560): {
        1: SkinnyGemmConfig(1, 64, 2, vector_width=4, static_k=2560),
        2: SkinnyGemmConfig(2, 128, 4, k_unroll=5, vector_width=4),
        4: SkinnyGemmConfig(4, 160, 1, k_unroll=2),
        8: SkinnyGemmConfig(8, 128, 1, vector_width=4, static_k=2560),
        16: SkinnyGemmConfig(16, 128, 1, vector_width=4, static_k=2560),
    },
    # LM head, TP=4.
    (62080, 2560): {
        1: SkinnyGemmConfig(1, 64, 2, vector_width=2, static_k=2560),
        2: SkinnyGemmConfig(2, 64, 2, vector_width=2, static_k=2560),
    },
    # HC merged down/injection projection, replicated in a TP=4 deployment.
    (336, 10240): {
        1: SkinnyGemmConfig(1, 256, 1, k_unroll=5),
        2: SkinnyGemmConfig(2, 256, 3, static_k=10240),
        4: SkinnyGemmConfig(4, 256, 3, static_k=10240),
        8: SkinnyGemmConfig(8, 256, 3, static_k=10240),
    },
    # Final HC down projection, replicated in a TP=4 deployment.
    (320, 10240): {
        1: SkinnyGemmConfig(1, 256, 1, static_k=10240),
        2: SkinnyGemmConfig(2, 128, 1, k_unroll=10),
        4: SkinnyGemmConfig(4, 128, 1, k_unroll=10),
        8: SkinnyGemmConfig(8, 128, 1, k_unroll=10),
    },
}


# RTX PRO 5000 Blackwell (sm120) plans selected by CUDA graph replay sweeps
# over M={1, 2, 3, 4, 6, 8, 12, 16} (the CUDA graph capture sizes <= 16 used
# with MTP=2) with weights rotated to defeat L2. Only points >= 5% faster
# than torch.nn.functional.linear (cuBLAS) are retained.
QWEN4_EXP_SM120_GEMM_PLANS: dict[tuple[int, int], dict[int, SkinnyGemmConfig]] = {
    # GDN fused QKVZ projection, TP=4.
    (4096, 2560): {
        1: SkinnyGemmConfig(1, 32, 1, vector_width=4, static_k=2560),
        2: SkinnyGemmConfig(2, 32, 1, vector_width=4, static_k=2560),
        3: SkinnyGemmConfig(3, 32, 1, vector_width=4, static_k=2560),
        4: SkinnyGemmConfig(4, 32, 2, vector_width=4, static_k=2560),
        6: SkinnyGemmConfig(6, 32, 1, vector_width=4, static_k=2560),
        8: SkinnyGemmConfig(8, 32, 2, vector_width=4, static_k=2560),
        12: SkinnyGemmConfig(12, 32, 2, vector_width=4, static_k=2560),
    },
    # GDN and QSA output projections, TP=4.
    (2560, 1536): {
        1: SkinnyGemmConfig(1, 32, 1, vector_width=4, static_k=1536),
        2: SkinnyGemmConfig(2, 32, 1, vector_width=4, static_k=1536),
        3: SkinnyGemmConfig(3, 32, 1, vector_width=4, static_k=1536),
        4: SkinnyGemmConfig(4, 32, 1, vector_width=4, static_k=1536),
        6: SkinnyGemmConfig(6, 32, 2, vector_width=4, static_k=1536),
        8: SkinnyGemmConfig(8, 32, 2, vector_width=4, static_k=1536),
        12: SkinnyGemmConfig(12, 32, 2, vector_width=4, static_k=1536),
        16: SkinnyGemmConfig(16, 32, 2, vector_width=4, static_k=1536),
    },
    # GDN fused B/A projection, TP=4.
    (24, 2560): {
        1: SkinnyGemmConfig(1, 160, 1, vector_width=4, static_k=2560),
        2: SkinnyGemmConfig(2, 160, 1, static_k=2560),
        3: SkinnyGemmConfig(3, 160, 1, static_k=2560),
        4: SkinnyGemmConfig(4, 128, 1, vector_width=4, static_k=2560),
        6: SkinnyGemmConfig(6, 128, 1, vector_width=4, static_k=2560),
        8: SkinnyGemmConfig(8, 128, 1, vector_width=4, static_k=2560),
        12: SkinnyGemmConfig(12, 128, 1, vector_width=4, static_k=2560),
        16: SkinnyGemmConfig(16, 160, 1, static_k=2560),
    },
    # QSA fused QKV/gate projection, TP=4.
    (3584, 2560): {
        1: SkinnyGemmConfig(1, 32, 1, vector_width=4, static_k=2560),
        2: SkinnyGemmConfig(2, 32, 1, vector_width=4, static_k=2560),
        3: SkinnyGemmConfig(3, 32, 1, vector_width=4, static_k=2560),
        4: SkinnyGemmConfig(4, 32, 2, vector_width=4, static_k=2560),
        6: SkinnyGemmConfig(6, 32, 1, vector_width=4, static_k=2560),
        8: SkinnyGemmConfig(8, 32, 1, vector_width=4, static_k=2560),
        12: SkinnyGemmConfig(12, 32, 2, vector_width=4, static_k=2560),
    },
    # QSA indexer Q/K projection, replicated in a TP=4 deployment.
    (640, 2560): {
        1: SkinnyGemmConfig(1, 128, 1, vector_width=4, static_k=2560),
        2: SkinnyGemmConfig(2, 128, 1, vector_width=4, static_k=2560),
        3: SkinnyGemmConfig(3, 128, 1, vector_width=4, static_k=2560),
        4: SkinnyGemmConfig(4, 128, 1, vector_width=4, static_k=2560),
        6: SkinnyGemmConfig(6, 128, 1, vector_width=4, static_k=2560),
        8: SkinnyGemmConfig(8, 64, 2, static_k=2560),
        12: SkinnyGemmConfig(12, 64, 2, static_k=2560),
        16: SkinnyGemmConfig(16, 64, 2, static_k=2560),
    },
    # Shared-expert fused gate/up projection, TP=4.
    (320, 2560): {
        1: SkinnyGemmConfig(1, 160, 2, vector_width=4, static_k=2560),
        2: SkinnyGemmConfig(2, 160, 2, vector_width=4, static_k=2560),
        3: SkinnyGemmConfig(3, 160, 2, vector_width=4, static_k=2560),
        4: SkinnyGemmConfig(4, 160, 1, vector_width=4, static_k=2560),
        6: SkinnyGemmConfig(6, 160, 1, vector_width=4, static_k=2560),
        8: SkinnyGemmConfig(8, 64, 1, static_k=2560),
        12: SkinnyGemmConfig(12, 64, 1, static_k=2560),
        16: SkinnyGemmConfig(16, 64, 1, static_k=2560),
    },
    # LM head, TP=4.
    (62080, 2560): {
        1: SkinnyGemmConfig(1, 32, 1, vector_width=4, static_k=2560),
        2: SkinnyGemmConfig(2, 32, 1, vector_width=4, static_k=2560),
        3: SkinnyGemmConfig(3, 32, 1, vector_width=4, static_k=2560),
        4: SkinnyGemmConfig(4, 32, 2, vector_width=4, static_k=2560),
        6: SkinnyGemmConfig(6, 32, 1, vector_width=4, static_k=2560),
        8: SkinnyGemmConfig(8, 32, 2, vector_width=4, static_k=2560),
        12: SkinnyGemmConfig(12, 32, 2, vector_width=4, static_k=2560),
    },
    # HC merged down/injection projection, replicated in a TP=4 deployment.
    (336, 10240): {
        1: SkinnyGemmConfig(1, 160, 1, vector_width=4, static_k=10240),
        2: SkinnyGemmConfig(2, 160, 1, vector_width=4, static_k=10240),
        3: SkinnyGemmConfig(3, 160, 2, vector_width=4, static_k=10240),
        4: SkinnyGemmConfig(4, 160, 2, vector_width=4, static_k=10240),
        6: SkinnyGemmConfig(6, 128, 2, static_k=10240),
        8: SkinnyGemmConfig(8, 160, 1, k_unroll=2),
        12: SkinnyGemmConfig(12, 128, 1, k_unroll=2),
    },
    # Final HC down projection, replicated in a TP=4 deployment.
    (320, 10240): {
        1: SkinnyGemmConfig(1, 256, 1, vector_width=4, static_k=10240),
        2: SkinnyGemmConfig(2, 256, 1, vector_width=4, static_k=10240),
        3: SkinnyGemmConfig(3, 256, 1, vector_width=4, static_k=10240),
        4: SkinnyGemmConfig(4, 256, 1, vector_width=4, static_k=10240),
        6: SkinnyGemmConfig(6, 160, 1, k_unroll=2),
        8: SkinnyGemmConfig(8, 128, 2, static_k=10240),
        12: SkinnyGemmConfig(12, 160, 1),
    },
}


def _is_sm103() -> bool:
    return current_platform.is_device_capability((10, 3))


def _is_sm90() -> bool:
    return current_platform.is_device_capability((9, 0))


def _is_sm120() -> bool:
    return current_platform.is_device_capability((12, 0))


def _gemm_plans() -> dict[tuple[int, int], dict[int, SkinnyGemmConfig]]:
    if _is_sm103():
        return QWEN4_EXP_GEMM_PLANS
    if _is_sm90():
        return QWEN4_EXP_SM90_GEMM_PLANS
    if _is_sm120():
        return QWEN4_EXP_SM120_GEMM_PLANS
    return {}


# Step 4 (Ardelva opt): Triton BF16 GEMM for the M=17..96 CUDA graph sizes on
# RTX PRO 5000 Blackwell (sm120), where cuBLAS picks cutlass_80 wmma/s16816
# kernels running ~2x off the HBM roofline. Plans: (BM, BN, BK, warps, stages),
# kept only where the CUDA-graph replay sweep (L2-rotated weights) showed
# >=5% over torch.nn.functional.linear.
_T_A = (16, 32, 64, 2, 6)
_T_B = (32, 64, 64, 2, 4)
_T_O1 = (16, 32, 256, 2, 3)
_T_O2 = (16, 32, 64, 2, 6)
_T_O3 = (32, 32, 128, 4, 4)
_T_S1 = (16, 16, 64, 4, 6)
_T_S2 = (32, 16, 64, 4, 6)
QWEN4_EXP_SM120_TRITON_PLANS: dict[tuple[int, int], dict[int, tuple]] = {
    (4096, 2560): {
        24: (16, 32, 64, 4, 6), 32: _T_A, 40: (16, 64, 128, 4, 3),
        48: (16, 64, 128, 2, 3), 72: _T_B, 80: _T_B, 88: _T_B, 96: _T_B,
    },
    (2560, 1536): {
        24: _T_O1, 32: _T_O1, 40: _T_O2, 48: _T_O2, 56: _T_O3, 64: _T_O3,
        72: _T_B, 80: (64, 64, 64, 4, 6), 88: _T_B, 96: (64, 64, 64, 4, 6),
    },
    (3584, 2560): {
        24: _T_A, 32: _T_A, 40: (16, 64, 128, 2, 3), 48: (16, 64, 128, 2, 3),
        72: _T_B, 80: _T_B, 88: _T_B, 96: _T_B,
    },
    (640, 2560): {
        24: (16, 16, 256, 4, 4), 32: (16, 16, 256, 2, 4), 64: (16, 32, 256, 2, 4),
    },
    (320, 2560): {
        24: _T_S1, 32: _T_S1, 40: _T_S1, 48: _T_S1, 56: _T_S1, 64: _T_S1,
        72: _T_S2, 80: _T_S2, 88: _T_S2, 96: _T_S2,
    },
}


@triton.jit
def _qwen4_exp_tl_gemm_kernel(
    x_ptr, w_ptr, y_ptr, M, N,
    K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    xp = x_ptr + rm[:, None] * K + rk[None, :]
    wp = w_ptr + rn[None, :] * K + rk[:, None]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    mm = rm[:, None] < M
    nm = rn[None, :] < N
    for _ in range(0, K, BK):
        a = tl.load(xp, mask=mm, other=0.0)
        b = tl.load(wp, mask=nm, other=0.0, eviction_policy="evict_first")
        acc = tl.dot(a, b, acc)
        xp += BK
        wp += BK
    tl.store(y_ptr + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16), mask=mm & nm)


def _qwen4_exp_triton_gemm(x: torch.Tensor, weight: torch.Tensor, cfg: tuple):
    bm, bn, bk, nw, ns = cfg
    m, k = x.shape
    n = weight.shape[0]
    y = torch.empty((m, n), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(n, bn), triton.cdiv(m, bm))
    _qwen4_exp_tl_gemm_kernel[grid](
        x, weight, y, m, n, K=k, BM=bm, BN=bn, BK=bk, num_warps=nw, num_stages=ns
    )
    return y


def _triton_plans() -> dict[tuple[int, int], dict[int, tuple]]:
    if _is_sm120() and not envs.VLLM_BATCH_INVARIANT:
        import os

        if os.environ.get("QWEN4_EXP_TRITON_GEMM", "1") == "1":
            return QWEN4_EXP_SM120_TRITON_PLANS
    return {}


def _is_packed_row_major(tensor: torch.Tensor) -> bool:
    return tensor.dim() == 2 and tensor.stride() == (tensor.shape[1], 1)


def _runtime_ok(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        not envs.VLLM_BATCH_INVARIANT
        and _is_packed_row_major(x)
        and _is_packed_row_major(weight)
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.is_cuda
        and weight.is_cuda
        and x.device == weight.device
        and x.shape[1] == weight.shape[1]
    )


class _Qwen4ExpLowLatencyApply:
    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bias is None and not envs.VLLM_BATCH_INVARIANT:
            return torch.ops.vllm.qwen4_exp_low_latency_gemm(x, layer.weight)
        return super().apply(layer, x, bias)  # type: ignore[misc]


class Qwen4ExpLowLatencyLinearMethod(_Qwen4ExpLowLatencyApply, UnquantizedLinearMethod):
    pass


class Qwen4ExpLowLatencyEmbeddingMethod(
    _Qwen4ExpLowLatencyApply, UnquantizedEmbeddingMethod
):
    pass


def _qwen4_exp_low_latency_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    plan = _gemm_plans().get((weight.shape[0], weight.shape[1]))
    config = None if plan is None else plan.get(x.shape[0])
    if (
        config is not None
        and _runtime_ok(x, weight)
        and shape_dynamic_skinny_gemm.is_available()
    ):
        return shape_dynamic_skinny_gemm(x, weight, config)
    tplan = _triton_plans().get((weight.shape[0], weight.shape[1]))
    tcfg = None if tplan is None else tplan.get(x.shape[0])
    if tcfg is not None and _runtime_ok(x, weight):
        return _qwen4_exp_triton_gemm(x, weight, tcfg)
    return torch.nn.functional.linear(x, weight)


def _qwen4_exp_low_latency_gemm_fake(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


direct_register_custom_op(
    op_name="qwen4_exp_low_latency_gemm",
    op_func=_qwen4_exp_low_latency_gemm,
    fake_impl=_qwen4_exp_low_latency_gemm_fake,
)


def enable_qwen4_exp_low_latency_gemm(
    module: nn.Module,
    dtype: torch.dtype,
) -> None:
    plans = _gemm_plans()
    if dtype != torch.bfloat16 or not plans:
        return
    if not shape_dynamic_skinny_gemm.is_available():
        return

    warmup_configs: set[SkinnyGemmConfig] = set()
    for child in module.modules():
        is_linear = (
            isinstance(child, LinearBase)
            and type(child.quant_method) is UnquantizedLinearMethod
        )
        is_head = (
            isinstance(child, ParallelLMHead)
            and type(child.quant_method) is UnquantizedEmbeddingMethod
        )
        if not (is_linear or is_head):
            continue
        weight = getattr(child, "weight", None)
        if weight is None or weight.dim() != 2:
            continue
        plan = plans.get((weight.shape[0], weight.shape[1]))
        if plan is None and (weight.shape[0], weight.shape[1]) in _triton_plans():
            plan = {}
        if plan is None:
            continue
        if is_linear:
            child.quant_method = Qwen4ExpLowLatencyLinearMethod()
        else:
            child.quant_method = Qwen4ExpLowLatencyEmbeddingMethod()
        warmup_configs.update(plan.values())

    if warmup_configs:
        shape_dynamic_skinny_gemm.request_warmup_configs(dtype, warmup_configs)
