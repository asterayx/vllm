# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the #52018 b12x MXFP4 MoE oracle policy."""

from unittest.mock import patch

import pytest

from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    B12X_BACKENDS,
    Mxfp4MoeBackend,
    _get_requested_backends,
    map_mxfp4_backend,
    mxfp4_round_up_hidden_size_and_intermediate_size,
)
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    NvFp4MoeBackend,
    _use_a16,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic

pytestmark = pytest.mark.cpu_test


def test_map_mxfp4_backend_b12x_lists_w4a8_then_w4a16():
    assert map_mxfp4_backend("b12x") == [
        Mxfp4MoeBackend.B12X_MXFP4_MXFP8,
        Mxfp4MoeBackend.B12X_MXFP4_BF16,
    ]
    assert not hasattr(Mxfp4MoeBackend, "B12X")
    assert not hasattr(Mxfp4MoeBackend, "B12X_MXFP8")


def test_b12x_prefers_w4a8_when_activation_unset():
    backends = _get_requested_backends("b12x", None)
    assert backends == list(B12X_BACKENDS)
    assert backends[0] == Mxfp4MoeBackend.B12X_MXFP4_MXFP8


def test_b12x_force_a16_selects_bf16_only():
    with patch(
        "vllm.model_executor.layers.fused_moe.oracle.mxfp4.envs."
        "VLLM_B12X_MOE_FP4_FORCE_A16",
        True,
    ):
        backends = _get_requested_backends("b12x", None)
    assert backends == [Mxfp4MoeBackend.B12X_MXFP4_BF16]


def test_b12x_explicit_mxfp8_activation_filters_to_w4a8():
    backends = _get_requested_backends("b12x", kMxfp8Dynamic)
    assert backends == [Mxfp4MoeBackend.B12X_MXFP4_MXFP8]


def test_b12x_does_not_round_up_hidden_or_intermediate():
    for backend in B12X_BACKENDS:
        hidden, inter = mxfp4_round_up_hidden_size_and_intermediate_size(
            backend, 2880, 1536
        )
        assert (hidden, inter) == (2880, 1536)


def test_nvfp4_force_a16_only_for_b12x():
    assert _use_a16(NvFp4MoeBackend.B12X, False) is False
    assert _use_a16(NvFp4MoeBackend.B12X, True) is True
    assert _use_a16(NvFp4MoeBackend.MARLIN, True) is True
    with patch(
        "vllm.model_executor.layers.fused_moe.oracle.nvfp4.envs."
        "VLLM_B12X_MOE_FP4_FORCE_A16",
        True,
    ):
        assert _use_a16(NvFp4MoeBackend.B12X, False) is True
        assert _use_a16(NvFp4MoeBackend.MARLIN, False) is False
