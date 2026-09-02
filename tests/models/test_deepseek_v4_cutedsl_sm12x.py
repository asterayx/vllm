# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM12x must not JIT in-tree CuteDSL (cute-to-nvvm enable-pyir ICE)."""

import inspect
from pathlib import Path
from unittest.mock import patch

from vllm.model_executor.kernels.linear.cute_dsl import ll_bf16
from vllm.models.deepseek_v4.common.cutedsl import use_dsv4_cutedsl
from vllm.models.deepseek_v4.common.ops import cache_utils, fused_indexer_q
from vllm.models.deepseek_v4.compressor import DeepseekCompressor
from vllm.platforms.interface import DeviceCapability
from vllm.utils.cutedsl import cutedsl_jit_supported

pytestmark = __import__("pytest").mark.cpu_test

_WARMUP = (
    Path(__file__).resolve().parents[2] / "vllm" / "model_executor" / "warmup"
)


def _patch_cutedsl_platform(major: int, minor: int = 0):
    return patch("vllm.utils.cutedsl.current_platform"), patch(
        "vllm.utils.cutedsl.has_cutedsl",
        return_value=True,
    )


def test_sm12x_does_not_select_dsv4_cutedsl():
    plat_patch, has_patch = _patch_cutedsl_platform(12, 1)
    with plat_patch as mock_platform, has_patch:
        mock_platform.get_device_capability.return_value = DeviceCapability(
            major=12, minor=1
        )
        assert cutedsl_jit_supported() is False
        assert use_dsv4_cutedsl() is False


def test_sm90_keeps_dsv4_cutedsl_when_installed():
    plat_patch, has_patch = _patch_cutedsl_platform(9, 0)
    with plat_patch as mock_platform, has_patch:
        mock_platform.get_device_capability.return_value = DeviceCapability(
            major=9, minor=0
        )
        assert cutedsl_jit_supported() is True
        assert use_dsv4_cutedsl() is True


def test_sm100_keeps_dsv4_cutedsl_when_installed():
    plat_patch, has_patch = _patch_cutedsl_platform(10, 0)
    with plat_patch as mock_platform, has_patch:
        mock_platform.get_device_capability.return_value = DeviceCapability(
            major=10, minor=0
        )
        assert cutedsl_jit_supported() is True
        assert use_dsv4_cutedsl() is True


def test_missing_cutedsl_package_disables_path():
    with patch("vllm.utils.cutedsl.has_cutedsl", return_value=False):
        assert cutedsl_jit_supported() is False
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


def test_ll_bf16_unavailable_on_sm12x_even_if_cutlass_imports():
    plat_patch, has_patch = _patch_cutedsl_platform(12, 1)
    with (
        plat_patch as mock_platform,
        has_patch,
        patch.object(ll_bf16, "_cutedsl_available", True),
    ):
        mock_platform.get_device_capability.return_value = DeviceCapability(
            major=12, minor=1
        )
        assert ll_bf16.is_available() is False


def test_ll_bf16_available_on_sm90_when_cutlass_imports():
    plat_patch, has_patch = _patch_cutedsl_platform(9, 0)
    with (
        plat_patch as mock_platform,
        has_patch,
        patch.object(ll_bf16, "_cutedsl_available", True),
    ):
        mock_platform.get_device_capability.return_value = DeviceCapability(
            major=9, minor=0
        )
        assert ll_bf16.is_available() is True


def test_warmup_sites_gate_cutedsl_on_sm12x():
    """Router GEMM / FA4 / generic CuteDSL warmup must not cute.compile on SM12x."""
    ll_src = (_WARMUP / "kernel_warmup.py").read_text()
    assert "is_ll_bf16_gemm_available()" in ll_src
    assert "ll_bf16_gemm_kernel.warmup" in ll_src
    assert "cutedsl_jit_supported()" in (_WARMUP / "cutedsl_warmup.py").read_text()
    assert "cutedsl_jit_supported()" in (
        _WARMUP / "fa4_cutedsl_warmup.py"
    ).read_text()
