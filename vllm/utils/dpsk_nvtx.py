# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in NVTX ranges for the DeepSeek-V4.1 RTX PRO 5000 serve path.

DPSK_NVTX=1 emits host-side NVTX ranges around the suspected non-GEMM costs
(input prep, PP send/recv, eager CUDA-graph breaks, Engram lookups, sampling,
drafting) so an nsys trace can attribute step time. Unset, every call returns
a shared nullcontext and costs one flag check.
"""

import functools
import os
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from typing import Any, TypeVar

import torch

DPSK_NVTX = os.environ.get("DPSK_NVTX", "0") == "1"
_NULL = nullcontext()

F = TypeVar("F", bound=Callable[..., Any])


def dpsk_nvtx(name: str) -> AbstractContextManager:
    if not DPSK_NVTX:
        return _NULL
    return torch.cuda.nvtx.range(name)


def dpsk_nvtx_fn(name: str) -> Callable[[F], F]:
    """Decorator form of dpsk_nvtx; returns the function unchanged when off.

    Under ``@eager_break_during_capture`` apply it as the inner decorator so
    the range is pushed at every replay of the eager segment.
    """

    def decorate(fn: F) -> F:
        if not DPSK_NVTX:
            return fn

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with torch.cuda.nvtx.range(name):
                return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorate
