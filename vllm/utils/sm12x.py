# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM12x (GB10 / SM120/SM121) small-batch alignment helpers."""

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

# Size-16 PIECEWISE dummies succeed on GB10; size 8 then IMA'd in MHC / b12x
# MoE (same compiled kernels, dynamic token count). Pad small batches up.
SM12X_SAFE_MIN_TOKENS = 16


def sm12x_align_tokens(num_tokens: int, min_tokens: int = SM12X_SAFE_MIN_TOKENS) -> int:
    """Return a safe token count for SM12x kernels. Unchanged off SM12x."""
    if (
        num_tokens > 0
        and num_tokens < min_tokens
        and current_platform.is_device_capability_family(120)
    ):
        return min_tokens
    return num_tokens


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
