# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM12x (GB10 / SM120/SM121) small-batch alignment helpers."""

from collections.abc import Iterator
from contextlib import contextmanager

import torch

from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

# Size-16 PIECEWISE dummies succeed on GB10; size 8 then IMA'd in MHC / b12x
# MoE (same compiled kernels, dynamic token count). Pad small batches up.
SM12X_SAFE_MIN_TOKENS = 16

# Per-request FlashInfer decode-form q_len. FlashInfer SM120 keeps
# num_tokens <= 64 on decode kernels.
# Decode dummies: 4 real tokens as [1, 4] survived capture (size 16 = 4×4).
# Do not pad those to [1, 6] by repeating the last row — that IMA'd
# (async, reported at DSpark capture start).
# Prefill: 2 and 4 real first-prefills IMA'd at [1, 4]; pad to 6.
# 2/3 must not snap to 4 (mixed-warmup 2→4 IMA'd).
SM12X_SAFE_DECODE_Q_LENS = (1, 4, 6, 8, 16, 24, 32, 36)
SM12X_SAFE_PREFILL_DECODE_Q_LENS = (1, 6, 8, 16, 24, 32, 36)
SM12X_UNSAFE_PER_REQUEST_Q_LENS = (2, 3)
# DSpark FULL dummies after aligning q_len 5→6. 36=6×6 and 24=4×6
# already went green in main capture. 6=1×6 MHC-pads to 16. Drop
# 18=3×6 and 12=2×6 (main capture never ran those token counts).
SM12X_DSPARK_SAFE_CAPTURE_TOKENS = (6, 24, 36)
# SM120 sparse *prefill* kernel asserts num_tokens > 64. Decode-form
# [1, 6] launched on the mixed-warmup seed (Spark 17:03) then IMA'd;
# DSpark [1, 6] *decode* dummies were fine. First prefills use this.
SM12X_PREFILL_KERNEL_MIN_TOKENS = 65


def sm12x_align_tokens(num_tokens: int, min_tokens: int = SM12X_SAFE_MIN_TOKENS) -> int:
    """Return a safe token count for SM12x kernels. Unchanged off SM12x."""
    if (
        num_tokens > 0
        and num_tokens < min_tokens
        and current_platform.is_device_capability_family(120)
    ):
        return min_tokens
    return num_tokens


def sm12x_mixed_warmup_decode_prompt_len() -> int:
    """Prefill length used to seed mixed-warmup's decode request.

    Keep the historical 2-token seed. SM12x pads that ``q_len=2`` prefill
    to one decode-form launch instead of splitting it or growing the seed
    into the untested ``num_tokens > 64`` prefill orchestrator.
    """
    return 2


def sm12x_mixed_warmup_prefill_len(requested_prefill: int) -> int:
    """Scheduled mixed-step prefill tokens.

    On SM12x keep ``q_len=2`` so FlashInfer pads one decode launch.
    Growing the scheduled prefill to 16 makes a 16-token *real* prefill,
    which is not the pad path and already IMA'd at 4 real tokens.
    """
    if requested_prefill <= 0:
        return requested_prefill
    if current_platform.is_device_capability_family(120):
        return sm12x_mixed_warmup_decode_prompt_len()
    return requested_prefill


def sm12x_dspark_capture_sizes(
    capture_sizes: list[int] | None,
    decode_query_len: int,
) -> list[int]:
    """Restrict SM12x DSpark FULL capture to proven token counts.

    Off SM12x, or when no safe size is divisible by ``decode_query_len``,
    return the original list.
    """
    sizes = list(capture_sizes or [])
    if not sizes or not current_platform.is_device_capability_family(120):
        return sizes
    max_cg = max(sizes)
    allow = [
        n
        for n in SM12X_DSPARK_SAFE_CAPTURE_TOKENS
        if n <= max_cg and n % decode_query_len == 0
    ]
    return allow or sizes


def sm12x_align_decode_q_len(q_len: int) -> int:
    """Snap a FlashInfer *decode dummy* q_len to a GB10-safe width.

    Keep 4 (CUDA-graph dummy). Snap 2/3 to 6, not 4. Unchanged off SM12x.
    """
    if q_len <= 0 or not current_platform.is_device_capability_family(120):
        return q_len
    for safe in SM12X_SAFE_DECODE_Q_LENS:
        if safe == 4 and q_len < 4:
            continue
        if q_len <= safe:
            return safe
    return q_len


def sm12x_treat_short_extends_as_decodes() -> bool:
    """Whether short first prefills may ride the decode split.

    DSpark sets ``decode_threshold = k+1 = 6``. A 2-token mixed-warmup
    seed then becomes an all-decode batch and FlashInfer launches
    ``[1, 4]`` (decode align), the width that IMA'd. On SM12x keep those
    requests on the prefill path so ``q_len=2`` pads to one ``[1, 6]``.
    """
    return not current_platform.is_device_capability_family(120)


def sm12x_disable_attn_aux_streams() -> bool:
    """Do not overlap indexer/compressor GEMMs on SM12x aux streams.

    ``maybe_execute_in_parallel`` already drops aux during breakable
    CUDA-graph capture. Mixed warmup is eager after capture and is the
    first time those aux streams run; Marlin already IMA'd on aux.
    Shared experts disable their aux stream on SM12x for the same reason.
    """
    return current_platform.is_device_capability_family(120)


def reject_sm12x_unsafe_decode_query(query: torch.Tensor) -> None:
    """Refuse SM12x per-request FlashInfer shapes that IMA'd on GB10.

    ``[1, 2]`` / ``[1, 3]`` IMA'd. ``[1, 4]`` is a valid decode dummy
    (size-16 capture); do not reject it.
    """
    if (
        query.ndim == 4
        and query.shape[0] == 1
        and int(query.shape[1]) in SM12X_UNSAFE_PER_REQUEST_Q_LENS
        and current_platform.is_device_capability_family(120)
    ):
        raise RuntimeError(
            "SM12x FlashInfer refused per-request query shape "
            f"{tuple(query.shape)}; pad q_len to 6"
        )


def sm12x_align_prefill_kernel_tokens(num_tokens: int) -> int:
    """Grow a short SM12x prefill so FlashInfer uses the >64 kernel."""
    if (
        0 < num_tokens <= 64
        and current_platform.is_device_capability_family(120)
    ):
        return SM12X_PREFILL_KERNEL_MIN_TOKENS
    return num_tokens


def sm12x_align_prefill_q_len(q_len: int) -> int:
    """Pad a short SM12x prefill to one decode-form launch.

    FlashInfer SM120 routes ``num_tokens <= 64`` to decode kernels. Do not
    split ``q_len=2`` into two ``q_len=1`` launches. Skip width 4: a
    4-token real seed IMA'd after capture.
    """
    if q_len <= 0 or not current_platform.is_device_capability_family(120):
        return q_len
    for safe in SM12X_SAFE_PREFILL_DECODE_Q_LENS:
        if q_len <= safe:
            return safe
    return q_len


def pad_token_rows(t: torch.Tensor, target: int) -> torch.Tensor:
    """Pad ``t`` along dim 0 with zeros up to ``target`` rows."""
    n = t.shape[0]
    if n >= target:
        return t
    return torch.cat((t, t.new_zeros((target - n, *t.shape[1:]))), dim=0)


def sm12x_pad_token_rows(
    t: torch.Tensor,
    min_tokens: int = SM12X_SAFE_MIN_TOKENS,
    *,
    what: str | None = None,
) -> tuple[torch.Tensor, int]:
    """Pad ``t`` to ``sm12x_align_tokens`` and return ``(padded, orig_rows)``."""
    orig = t.shape[0]
    target = sm12x_align_tokens(orig, min_tokens)
    if target == orig:
        return t, orig
    if what is not None:
        logger.info_once(
            "SM12x %s: padding token dim %d -> %d to avoid small-batch IMA",
            what,
            orig,
            target,
        )
    return pad_token_rows(t, target), orig


def extend_padding_mask(
    is_padding: torch.Tensor | None, num_tokens: int
) -> torch.Tensor | None:
    """Slice or grow ``is_padding`` so it has exactly ``num_tokens`` rows.

    Extra rows from SM12x kernel alignment are marked True (not real tokens).
    """
    if is_padding is None:
        return None
    n = is_padding.shape[0]
    if n >= num_tokens:
        return is_padding[:num_tokens]
    extra = num_tokens - n
    return torch.cat((is_padding, is_padding.new_ones((extra,), dtype=torch.bool)))


@contextmanager
def sm12x_align_is_padding(num_tokens: int) -> Iterator[None]:
    """Temporarily grow ForwardContext.is_padding to ``num_tokens``."""
    if not is_forward_context_available():
        yield
        return
    ctx = get_forward_context()
    orig = ctx.is_padding
    if orig is not None and orig.shape[0] != num_tokens:
        ctx.is_padding = extend_padding_mask(orig, num_tokens)
    try:
        yield
    finally:
        ctx.is_padding = orig
