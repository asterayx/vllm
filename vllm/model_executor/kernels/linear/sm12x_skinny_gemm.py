# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split-K Triton GEMM for skinny bf16 decode projections on SM12x.

For ``M <= 32`` rows and a ``[N, K]`` bf16 weight, cuBLAS on GB10 picks a
``cutlass_80_wmma ... 16x16_128x2`` kernel whose grid is ``N / 16`` CTAs, so
a ``[16, 4096] x [4096, 256]`` gate GEMM runs on 16 of 48 SMs and takes
50-114 us for 2-8 MB of weight (5-6x the HBM floor). Streaming the weight
across ``K`` splits and reducing the fp32 partials brings the launch to the
bandwidth floor. The reduction order is fixed (no atomics), so outputs are
deterministic run to run.
"""

from __future__ import annotations

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

SKINNY_GEMM_MAX_ROWS = 32
_BLOCK_N = 64
_BLOCK_K = 128
_TARGET_CTAS = 128


@triton.jit(do_not_specialize=["num_rows"])
def _skinny_gemm_split_k_kernel(
    x_ptr,
    w_ptr,
    part_ptr,
    num_rows,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_ps,
    stride_pm,
    stride_pn,
    K_PER_SPLIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_base = pid_k * K_PER_SPLIT
    # tl.full is a builtin; tl.zeros is a jitted helper the CPU interpreter
    # cannot call when TRITON_INTERPRET is set after triton was imported.
    acc = tl.full((BLOCK_M, BLOCK_N), 0.0, tl.float32)
    for k0 in range(0, K_PER_SPLIT, BLOCK_K):
        offs_k = k_base + k0 + tl.arange(0, BLOCK_K)
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=(offs_m[:, None] < num_rows) & (offs_k[None, :] < K),
            other=0.0,
        )
        w = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
            mask=(offs_n[:, None] < N) & (offs_k[None, :] < K),
            other=0.0,
        )
        acc += tl.dot(x, tl.trans(w), out_dtype=tl.float32)
    tl.store(
        part_ptr
        + pid_k * stride_ps
        + offs_m[:, None] * stride_pm
        + offs_n[None, :] * stride_pn,
        acc,
        mask=(offs_m[:, None] < num_rows) & (offs_n[None, :] < N),
    )


@triton.jit
def _skinny_gemm_reduce_kernel(
    part_ptr,
    out_ptr,
    N,
    stride_ps,
    stride_pm,
    stride_pn,
    stride_om,
    stride_on,
    SPLIT_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs_n < N
    acc = tl.full((BLOCK_N,), 0.0, tl.float32)
    for s in range(SPLIT_K):
        acc += tl.load(
            part_ptr + s * stride_ps + row * stride_pm + offs_n * stride_pn,
            mask=mask,
            other=0.0,
        )
    tl.store(
        out_ptr + row * stride_om + offs_n * stride_on,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def skinny_gemm_split_plan(N: int, K: int) -> tuple[int, int]:
    """Return ``(split_k, k_per_split)`` so the grid has ~_TARGET_CTAS CTAs."""
    n_blocks = triton.cdiv(N, _BLOCK_N)
    want = max(1, -(-_TARGET_CTAS // n_blocks))
    k_blocks = triton.cdiv(K, _BLOCK_K)
    split_k = max(1, min(want, k_blocks))
    k_per_split = triton.cdiv(k_blocks, split_k) * _BLOCK_K
    split_k = triton.cdiv(K, k_per_split)
    return split_k, k_per_split


def skinny_gemm_applicable(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """``x @ weight.T`` with a bf16 ``[N, K]`` weight and at most 32 rows."""
    return (
        x.dim() == 2
        and weight.dim() == 2
        and 0 < x.shape[0] <= SKINNY_GEMM_MAX_ROWS
        and x.shape[1] == weight.shape[1]
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.is_cuda
        and weight.is_cuda
        and x.stride(1) == 1
        and weight.stride(1) == 1
        and weight.shape[1] % 8 == 0
    )


def skinny_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    out_dtype: torch.dtype,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``x[M, K] @ weight[N, K].T`` in fp32 accumulate, cast to ``out_dtype``."""
    M, K = x.shape
    N = weight.shape[0]
    split_k, k_per_split = skinny_gemm_split_plan(N, K)
    block_m = 16 if M <= 16 else 32
    partials = torch.empty((split_k, M, N), dtype=torch.float32, device=x.device)
    _skinny_gemm_split_k_kernel[(triton.cdiv(N, _BLOCK_N), split_k)](
        x,
        weight,
        partials,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        partials.stride(0),
        partials.stride(1),
        partials.stride(2),
        K_PER_SPLIT=k_per_split,
        BLOCK_M=block_m,
        BLOCK_N=_BLOCK_N,
        BLOCK_K=_BLOCK_K,
        num_warps=4,
        num_stages=2,
    )
    if out is None:
        out = torch.empty((M, N), dtype=out_dtype, device=x.device)
    reduce_block = 1024
    _skinny_gemm_reduce_kernel[(M, triton.cdiv(N, reduce_block))](
        partials,
        out,
        N,
        partials.stride(0),
        partials.stride(1),
        partials.stride(2),
        out.stride(0),
        out.stride(1),
        SPLIT_K=split_k,
        BLOCK_N=reduce_block,
        num_warps=4,
    )
    return out


_verified_shapes: set[tuple[int, int]] = set()
_disabled = False


def _is_capturing() -> bool:
    is_capturing = getattr(torch.cuda, "is_current_stream_capturing", None)
    return bool(is_capturing is not None and is_capturing())


def skinny_linear(
    x: torch.Tensor, weight: torch.Tensor, out_dtype: torch.dtype
) -> torch.Tensor:
    """``F.linear(x, weight)`` for skinny bf16 inputs via the split-K kernel.

    The first eager call per ``(N, K)`` is checked against ``torch.mm`` and
    the kernel is disabled for the process if it disagrees, so a Triton
    regression degrades to the cuBLAS path instead of corrupting outputs.
    Callers must check :func:`skinny_gemm_applicable` first.
    """
    global _disabled
    if _disabled:
        return torch.mm(x, weight.T, out_dtype=out_dtype)
    shape = (weight.shape[0], weight.shape[1])
    if shape in _verified_shapes or _is_capturing():
        return skinny_gemm(x, weight, out_dtype)
    out = skinny_gemm(x, weight, torch.float32)
    ref = torch.mm(x, weight.T, out_dtype=torch.float32)
    if not torch.allclose(out, ref, rtol=1e-2, atol=1e-2 * ref.abs().max().item()):
        _disabled = True
        logger.warning(
            "SM12x skinny GEMM mismatch for N=%d K=%d (max abs err %.3e); "
            "falling back to cuBLAS for the rest of the process.",
            shape[0],
            shape[1],
            (out - ref).abs().max().item(),
        )
        return ref.to(out_dtype)
    _verified_shapes.add(shape)
    logger.info_once("SM12x skinny GEMM enabled for N=%d K=%d", shape[0], shape[1])
    return out.to(out_dtype)
