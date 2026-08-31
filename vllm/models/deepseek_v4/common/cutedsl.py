# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dispatch helper for in-tree DeepSeek V4 CuteDSL kernels."""

from vllm.utils.cutedsl import cutedsl_jit_supported


def use_dsv4_cutedsl() -> bool:
    """Whether DSv4 should compile/run in-tree CuteDSL kernels.

    SM12x is rejected by ``cutedsl_jit_supported`` (cute-to-nvvm
    ``enable-pyir`` ICE). Callers fall back to Triton.
    """
    return cutedsl_jit_supported()
