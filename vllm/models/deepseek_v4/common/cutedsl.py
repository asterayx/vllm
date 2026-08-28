# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dispatch helper for in-tree DeepSeek V4 CuteDSL kernels."""

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_cutedsl

logger = init_logger(__name__)


def use_dsv4_cutedsl() -> bool:
    """Whether DSv4 should compile/run in-tree CuteDSL kernels.

    SM12x (GB10 / sm_121a, capability family 120) Python nvidia-cutlass-dsl
    emits ``enable-pyir=false`` that the installed cute-to-nvvm pass does
    not know. That is a compiler/package mismatch ICE, not a kernel bug.
    Official extra stays ``b12x==1.2.6``; we do not bump the global
    cutlass-dsl pin. Callers fall back to Triton.
    """
    if not has_cutedsl():
        return False
    cap = current_platform.get_device_capability()
    if cap is not None and cap.major == 12:
        logger.info_once(
            "Skipping DSv4 CuteDSL on SM12x (cute-to-nvvm enable-pyir "
            "mismatch); using Triton"
        )
        return False
    return True
