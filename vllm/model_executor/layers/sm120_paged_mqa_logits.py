# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120 Triton paged MQA logits for the fp8 indexer cache with block sizes
DeepGEMM's sm120 kernel does not accept (DeepSeek-V4.1 ratio-1 layers: 128-row
pages). Replaces the fp32 torch reference (dequant + einsum simt sgemm).

Page layout (segregated, same as `_paged_mqa_logits_ref_fp8`):
    page p at byte offset p * page_stride:
        [bs * D fp8 e4m3 values | bs fp32 scales]

logits[b*N + n, t] = scale[t] * sum_h w[b,n,h] * relu(q[b,n,h,:] . k[t,:])
for t < context_lens[b, n], else -inf.  Output: [B*N, max_model_len] fp32.
Gated by DPSK_INDEXER_SM120_FIX=1. No host syncs (graph/capture friendly).

DSV41_EXP_INDEXER_LOGITS_SHRINK=1 allocates and stores only the used context
(host upper bound, rounded up to 128) instead of max_model_len. Column t is
still position t. Unset keeps the full-width allocation and launch.
"""

import math
import numbers
import os
from typing import NamedTuple

import torch
import triton
import triton.language as tl

from vllm.logger import init_logger

logger = init_logger(__name__)

INDEXER_LOGITS_SHRINK_ENV = "DSV41_EXP_INDEXER_LOGITS_SHRINK"
# Top-k row stride must stay a multiple of 4. The kernel tile is at most 128.
_LOGITS_WIDTH_ALIGN = 128
_SM120_BACKEND = "sm120_paged_mqa_logits"
_DEEPGEMM_BACKEND = "DeepGEMM fp8_fp4_paged_mqa_logits"
_QUIET_WIDTH_REASONS = frozenset({"disabled", "full_bound"})


class Sm120LogitsWidth(NamedTuple):
    """Host decision for the fp32 logits row width.

    ``active`` means the allocation and the kernel launch both use ``width``
    and ``width < max_model_len``. ``log_line`` is None when the flag is off
    or when the host bound already covers ``max_model_len``.
    """

    width: int
    active: bool
    reason: str
    log_line: str | None


def indexer_logits_shrink_enabled() -> bool:
    """Return whether the opt-in used-length logits shrink is requested."""
    return os.environ.get(INDEXER_LOGITS_SHRINK_ENV, "0") == "1"


def cuda_stream_capturing() -> bool:
    """True when the current stream is recording a CUDA graph."""
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except RuntimeError:
        return False


def _align_up(n: int, align: int) -> int:
    return (n + align - 1) // align * align


def _fallback(
    max_model_len: int,
    reason: str,
    detail: str,
    backend: str,
) -> Sm120LogitsWidth:
    return Sm120LogitsWidth(
        width=max_model_len,
        active=False,
        reason=reason,
        log_line=(
            "DSV41_EXP_INDEXER_LOGITS_SHRINK fallback: "
            f"{backend} keeps full width {max_model_len} ({detail})"
        ),
    )


def resolve_sm120_paged_logits_width(
    max_model_len: int,
    used_len: int | None = None,
    *,
    enabled: bool,
    capturing: bool = False,
    out_width: int | None = None,
    align: int = _LOGITS_WIDTH_ALIGN,
    backend: str = _SM120_BACKEND,
) -> Sm120LogitsWidth:
    """Choose the fp32 logits width for ``sm120_paged_mqa_logits_fp8``.

    Args:
        max_model_len: Legacy full row width. Returned whenever shrink is off
            or cannot be applied.
        used_len: Host upper bound on every context length this launch will
            read. Positions ``t < used_len`` keep their column index.
        enabled: ``DSV41_EXP_INDEXER_LOGITS_SHRINK=1``.
        capturing: This launch would be recorded into a CUDA graph whose
            replay can outgrow ``used_len``.
        out_width: Width of a caller-provided output, if any.
        align: Row alignment. DeepGEMM writes whole KV pages, so its align
            must be a multiple of the page size.
        backend: Name used in the active and fallback log lines.

    Returns:
        The width to allocate and to pass as the kernel's output length.

    """
    if not enabled:
        return Sm120LogitsWidth(max_model_len, False, "disabled", None)
    if capturing:
        return _fallback(
            max_model_len,
            "cuda_graph_capture",
            "cuda graph capture; replay can outgrow the capture-time context",
            backend,
        )
    if isinstance(used_len, bool) or not isinstance(used_len, numbers.Integral):
        return _fallback(
            max_model_len,
            "missing_used_len",
            "used length was not provided",
            backend,
        )
    used = int(used_len)
    if used <= 0 or max_model_len <= 0:
        return _fallback(
            max_model_len,
            "invalid_used_len",
            f"used_len={used_len} is not a positive length",
            backend,
        )
    if used > max_model_len:
        return _fallback(
            max_model_len,
            "used_len_exceeds_max_model_len",
            f"used_len={used} exceeds max_model_len",
            backend,
        )
    align_to = align if align > 0 else _LOGITS_WIDTH_ALIGN
    width = min(max_model_len, _align_up(used, align_to))
    if width >= max_model_len:
        return Sm120LogitsWidth(max_model_len, False, "full_bound", None)
    if out_width is not None and out_width != width:
        return _fallback(
            max_model_len,
            "preallocated_out",
            f"preallocated logits width {out_width} is not the shrunk width {width}",
            backend,
        )
    return Sm120LogitsWidth(
        width=width,
        active=True,
        reason="shrunk",
        log_line=(
            "DSV41_EXP_INDEXER_LOGITS_SHRINK active: "
            f"{backend} fp32 width={width} used_len={used} "
            f"max_model_len={max_model_len}"
        ),
    )


def deepgemm_paged_logits_width(
    max_model_len: int,
    used_len: int | None,
    page_size: int,
    *,
    enabled: bool,
    capturing: bool,
) -> Sm120LogitsWidth:
    """Width for ``fp8_fp4_paged_mqa_logits``.

    SM120 DeepSeek-V4.1 indexer pages are 32 or 64, which DeepGEMM accepts,
    so ``DPSK_INDEXER_SM120_FIX`` does not select the Triton kernel. The
    DeepGEMM ``max_model_len`` argument is the output row width. Rounding up
    to a multiple of both the page size and 128 keeps every page the kernel
    writes for ``used_len`` inside the row. Column ``t`` stays position ``t``.
    """
    page = page_size if page_size > 0 else 1
    return resolve_sm120_paged_logits_width(
        max_model_len,
        used_len,
        enabled=enabled,
        capturing=capturing,
        align=math.lcm(page, _LOGITS_WIDTH_ALIGN),
        backend=_DEEPGEMM_BACKEND,
    )


def commit_indexer_logits_shrink(decision: Sm120LogitsWidth) -> int:
    """Return the logits width, or raise if shrink was requested and refused.

    ``disabled`` and ``full_bound`` keep today's width with no log. An active
    shrink logs the attach line. Any other reason refuses the launch instead
    of continuing at full width.

    Raises:
        RuntimeError: Shrink was requested and this launch would stay full
            width.

    """
    if decision.reason in _QUIET_WIDTH_REASONS:
        return decision.width
    if decision.active:
        if decision.log_line is not None:
            logger.info_once(decision.log_line)
        return decision.width
    message = decision.log_line or (
        "DSV41_EXP_INDEXER_LOGITS_SHRINK fallback: refusing full-width "
        f"logits ({decision.reason})"
    )
    logger.error(message)
    raise RuntimeError(message)


@triton.jit
def _sm120_paged_mqa_logits_kernel(
    q_ptr,  # fp8 [B, N, H, D] contiguous
    kv_ptr,  # uint8 base of kv cache
    w_ptr,  # [B*N, H] (any float dtype)
    cl_ptr,  # int32 [B, CLN] context lens
    bt_ptr,  # int32 [B, BTW] block table
    out_ptr,  # fp32 [B*N, L]
    page_stride,
    nb_total,
    bt_stride,
    bt_width,
    cl_stride,
    w_stride,
    out_stride,
    L,
    N: tl.constexpr,
    CLN: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BS: tl.constexpr,
    BT: tl.constexpr,
    FP8_DOT: tl.constexpr,
):
    b = tl.program_id(0)
    tile = tl.program_id(1)
    t0 = tile * BT
    offs_t = t0 + tl.arange(0, BT)
    tmask = offs_t < L
    neg_inf = tl.full([BT], float("-inf"), tl.float32)

    # max context over the N rows of this request
    maxcl = 0
    for n in tl.static_range(N):
        if CLN == 1:
            c = tl.load(cl_ptr + b * cl_stride)
        else:
            c = tl.load(cl_ptr + b * cl_stride + n)
        maxcl = tl.maximum(maxcl, c)

    blk = t0 // BS
    if (t0 >= maxcl) or (blk >= bt_width):
        for n in tl.static_range(N):
            tl.store(
                out_ptr + (b * N + n).to(tl.int64) * out_stride + offs_t,
                neg_inf,
                mask=tmask,
            )
        return

    page = tl.load(bt_ptr + b * bt_stride + blk)
    page = tl.minimum(tl.maximum(page, 0), nb_total - 1).to(tl.int64)
    base = kv_ptr + page * page_stride
    row0 = t0 % BS
    offs_r = row0 + tl.arange(0, BT)
    offs_d = tl.arange(0, D)
    k = tl.load(base + offs_r[:, None] * D + offs_d[None, :])  # uint8 [BT, D]
    k = k.to(tl.float8e4nv, bitcast=True)
    sc = tl.load((base + BS * D).to(tl.pointer_type(tl.float32)) + offs_r)  # [BT]
    if not FP8_DOT:
        k = k.to(tl.bfloat16)
    offs_h = tl.arange(0, H)
    for n in tl.static_range(N):
        row = b * N + n
        q = tl.load(
            q_ptr + (row * H + offs_h[:, None]).to(tl.int64) * D + offs_d[None, :]
        )  # fp8 [H, D]
        if not FP8_DOT:
            q = q.to(tl.bfloat16)
        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32)  # [H, BT]
        s = tl.maximum(s, 0.0)
        w = tl.load(w_ptr + row * w_stride + offs_h).to(tl.float32)  # [H]
        lg = tl.sum(s * w[:, None], axis=0) * sc  # [BT]
        if CLN == 1:
            c = tl.load(cl_ptr + b * cl_stride)
        else:
            c = tl.load(cl_ptr + b * cl_stride + n)
        lg = tl.where(offs_t < c, lg, neg_inf)
        tl.store(out_ptr + row.to(tl.int64) * out_stride + offs_t, lg, mask=tmask)


_FP8_DOT = True


def _launch_sm120_paged_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    w: torch.Tensor,
    cl: torch.Tensor,
    bt: torch.Tensor,
    out: torch.Tensor,
    width: int,
    *,
    n: int,
    cln: int,
    h: int,
    d: int,
    bs: int,
    bt_tile: int,
) -> None:
    grid = (q.shape[0], triton.cdiv(width, bt_tile))
    _sm120_paged_mqa_logits_kernel[grid](
        q,
        kv_cache,
        w,
        cl,
        bt,
        out,
        int(kv_cache.stride(0)),
        kv_cache.shape[0],
        bt.stride(0),
        bt.shape[1],
        cl.stride(0),
        w.stride(0),
        out.stride(0),
        width,
        N=n,
        CLN=cln,
        H=h,
        D=d,
        BS=bs,
        BT=bt_tile,
        FP8_DOT=_FP8_DOT,
        num_warps=4,
        num_stages=1,
    )


def sm120_paged_mqa_logits_fp8(
    q_fp8: torch.Tensor,  # [B, N, H, D] fp8
    kv_cache: torch.Tensor,  # [nb, bs, ...] uint8 (segregated page layout)
    weights: torch.Tensor,  # [B*N, H]
    context_lens: torch.Tensor,  # [B, N] or [B, 1] / [B]
    block_table: torch.Tensor,  # [>=B, W] int32
    max_model_len: int,
    out: torch.Tensor | None = None,
    used_len: int | None = None,
    *,
    capturing: bool | None = None,
) -> torch.Tensor:
    """Score paged fp8 keys into fp32 logits of shape ``[B*N, width]``.

    ``width`` is ``max_model_len`` unless ``DSV41_EXP_INDEXER_LOGITS_SHRINK=1``.
    ``used_len`` is the host upper bound on context; the shrunk row is that
    bound aligned up to 128, and column ``t`` stays position ``t``.
    ``capturing=None`` reads the current stream. A captured launch raises:
    replay cannot grow a recorded width.

    Raises:
        RuntimeError: Shrink was requested and this launch is being captured
            or otherwise cannot leave the full width.

    """
    b, n, h, d = q_fp8.shape
    bs = kv_cache.shape[1]
    enabled = indexer_logits_shrink_enabled()
    if capturing is None:
        capturing = cuda_stream_capturing() if enabled else False
    decision = resolve_sm120_paged_logits_width(
        max_model_len,
        used_len,
        enabled=enabled,
        capturing=capturing,
        out_width=None if out is None else int(out.shape[1]),
    )
    width = commit_indexer_logits_shrink(decision)
    if out is None:
        out = torch.empty((b * n, width), dtype=torch.float32, device=q_fp8.device)
    if b == 0:
        return out
    q = q_fp8.contiguous()
    w = weights.reshape(b * n, h)
    if w.stride(1) != 1:
        w = w.contiguous()
    cl = context_lens.reshape(b, -1)
    if cl.dtype != torch.int32:
        cl = cl.to(torch.int32)
    if cl.stride(1) != 1 and cl.shape[1] > 1:
        cl = cl.contiguous()
    cln = cl.shape[1]
    assert cln in (1, n), (cl.shape, n)
    bt = block_table
    if bt.dtype != torch.int32:
        bt = bt.to(torch.int32)
    bt_tile = min(bs, 128)
    assert bs % bt_tile == 0
    # Flag off: width is max_model_len, same launch as before.
    _launch_sm120_paged_mqa_logits(
        q,
        kv_cache,
        w,
        cl,
        bt,
        out,
        width,
        n=n,
        cln=cln,
        h=h,
        d=d,
        bs=bs,
        bt_tile=bt_tile,
    )
    return out
