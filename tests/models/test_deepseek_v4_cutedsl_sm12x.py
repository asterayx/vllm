# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM12x must not JIT in-tree DSv4 CuteDSL (cute-to-nvvm enable-pyir ICE)."""

import inspect
from unittest.mock import patch

from vllm.models.deepseek_v4.common.cutedsl import use_dsv4_cutedsl
from vllm.models.deepseek_v4.common.ops import cache_utils, fused_indexer_q
from vllm.models.deepseek_v4.compressor import DeepseekCompressor
from vllm.platforms.interface import DeviceCapability

pytestmark = __import__("pytest").mark.cpu_test


def _patch_cutedsl_platform(major: int, minor: int = 0):
    return patch(
        "vllm.models.deepseek_v4.common.cutedsl.current_platform"
    ), patch(
        "vllm.models.deepseek_v4.common.cutedsl.has_cutedsl",
        return_value=True,
    )


def test_sm12x_does_not_select_dsv4_cutedsl():
    plat_patch, has_patch = _patch_cutedsl_platform(12, 1)
    with plat_patch as mock_platform, has_patch:
        mock_platform.get_device_capability.return_value = DeviceCapability(
            major=12, minor=1
        )
        assert use_dsv4_cutedsl() is False


def test_sm90_keeps_dsv4_cutedsl_when_installed():
    plat_patch, has_patch = _patch_cutedsl_platform(9, 0)
    with plat_patch as mock_platform, has_patch:
        mock_platform.get_device_capability.return_value = DeviceCapability(
            major=9, minor=0
        )
        assert use_dsv4_cutedsl() is True


def test_sm100_keeps_dsv4_cutedsl_when_installed():
    plat_patch, has_patch = _patch_cutedsl_platform(10, 0)
    with plat_patch as mock_platform, has_patch:
        mock_platform.get_device_capability.return_value = DeviceCapability(
            major=10, minor=0
        )
        assert use_dsv4_cutedsl() is True


def test_missing_cutedsl_package_disables_path():
    with patch(
        "vllm.models.deepseek_v4.common.cutedsl.has_cutedsl",
        return_value=False,
    ):
        assert use_dsv4_cutedsl() is False


def test_dsv4_forward_sites_gate_cutedsl_on_sm12x():
    """Indexer Q, compressor, and K-gather must share the SM12x CuteDSL skip."""
    assert "use_dsv4_cutedsl()" in inspect.getsource(
        fused_indexer_q.fused_indexer_q_rope_quant
    )
    assert "use_dsv4_cutedsl()" in inspect.getsource(
        cache_utils.dequantize_and_gather_k_cache
    )
    assert "use_dsv4_cutedsl()" in inspect.getsource(DeepseekCompressor.forward)
