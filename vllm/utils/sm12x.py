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

# Per-request FlashInfer decode-form q_len that survived GB10 CUDA-graph
# capture. 2 and 3 IMA; splitting a 2-token *prefill* into two q_len=1
# launches also IMA'd (token 1's KV is not in cache yet).
SM12X_SAFE_DECODE_Q_LENS = (1, 4, 6, 8, 16, 24, 32, 36)


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

    SM12x FlashInfer IMA'd on the historical 2-token seed. q_len=4 survived
    CUDA-graph capture on GB10.
    """
    if current_platform.is_device_capability_family(120):
        return 4
    return 2


def sm12x_align_decode_q_len(q_len: int) -> int:
    """Snap a FlashInfer decode-form q_len to a GB10-safe width.

    Unchanged off SM12x. Values already in ``SM12X_SAFE_DECODE_Q_LENS`` or
    larger than the last entry are left as-is.
    """
    if q_len <= 0 or not current_platform.is_device_capability_family(120):
        return q_len
    for safe in SM12X_SAFE_DECODE_Q_LENS:
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
