# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Whether in-tree CuteDSL JIT can compile on this GPU."""

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_cutedsl

logger = init_logger(__name__)


def cutedsl_jit_supported() -> bool:
    """Return whether ``cute.compile`` is usable on this device.

    SM12x (GB10 / sm_121a, capability family 120) Python nvidia-cutlass-dsl
    emits ``enable-pyir=false`` that the installed cute-to-nvvm pass does
    not know. That is a compiler/package mismatch ICE, not a kernel bug.
    Callers keep their existing non-CuteDSL fallback (Triton, cuBLAS,
    ``F.linear``). Official extra stays ``b12x==1.2.6``.
    """
    if not has_cutedsl():
        return False
    cap = current_platform.get_device_capability()
    if cap is not None and cap.major == 12:
        logger.info_once(
            "Skipping in-tree CuteDSL JIT on SM12x "
            "(cute-to-nvvm enable-pyir mismatch)"
        )
        return False
    return True
