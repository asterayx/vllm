# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from vllm.models.deepseek_v4.nvidia.ops.o_proj import compute_fp8_einsum_recipe
from vllm.platforms.interface import DeviceCapability

pytestmark = __import__("pytest").mark.cpu_test


@patch("vllm.models.deepseek_v4.nvidia.ops.o_proj.current_platform")
def test_sm12x_uses_hopper_fp32_recipe(mock_platform):
    mock_platform.get_device_capability.return_value = DeviceCapability(
        major=12, minor=1
    )
    recipe, tma_aligned = compute_fp8_einsum_recipe()
    # Hopper K-granularity (1, 128, 128) with FP32 activation scales: the
    # INT32-packed UE8M0 variant is numerically wrong on SM12x (measured
    # ~2^32 error on GB10 vs an fp32 reference).
    assert recipe == (1, 128, 128)
    assert tma_aligned is False


@patch("vllm.models.deepseek_v4.nvidia.ops.o_proj.current_platform")
def test_sm100_uses_packed_int32_recipe(mock_platform):
    mock_platform.get_device_capability.return_value = DeviceCapability(
        major=10, minor=0
    )
    recipe, tma_aligned = compute_fp8_einsum_recipe()
    assert recipe == (1, 1, 128)
    assert tma_aligned is True


@patch("vllm.models.deepseek_v4.nvidia.ops.o_proj.current_platform")
def test_sm90_uses_hopper_fp32_recipe(mock_platform):
    mock_platform.get_device_capability.return_value = DeviceCapability(
        major=9, minor=0
    )
    recipe, tma_aligned = compute_fp8_einsum_recipe()
    assert recipe == (1, 128, 128)
    assert tma_aligned is False


@patch("vllm.models.deepseek_v4.nvidia.ops.fp8_einsum.current_platform")
def test_sm12x_triton_predicate_accepts_hopper_recipe(mock_platform):
    from vllm.models.deepseek_v4.nvidia.ops.fp8_einsum import (
        _use_deepseek_v4_sm12x_triton_fp8_einsum,
    )

    mock_platform.get_device_capability.return_value = DeviceCapability(
        major=12, minor=1
    )
    scale = torch.zeros(1, dtype=torch.float32)
    assert _use_deepseek_v4_sm12x_triton_fp8_einsum(
        "bhr,hdr->bhd", (1, 128, 128), scale
    )
    assert not _use_deepseek_v4_sm12x_triton_fp8_einsum(
        "bhr,hdr->bhd", (1, 1, 128), scale
    )
    uint8_scale = torch.zeros(1, dtype=torch.uint8)
    assert _use_deepseek_v4_sm12x_triton_fp8_einsum(
        "bhr,hdr->bhd", (1, 128, 128), uint8_scale
    )


@patch("vllm.models.deepseek_v4.nvidia.ops.fp8_einsum.current_platform")
def test_sm100_does_not_take_triton_fallback(mock_platform):
    from vllm.models.deepseek_v4.nvidia.ops.fp8_einsum import (
        _use_deepseek_v4_sm12x_triton_fp8_einsum,
    )

    mock_platform.get_device_capability.return_value = DeviceCapability(
        major=10, minor=0
    )
    scale = torch.zeros(1, dtype=torch.float32)
    assert not _use_deepseek_v4_sm12x_triton_fp8_einsum(
        "bhr,hdr->bhd", (1, 128, 128), scale
    )


@patch("vllm.models.deepseek_v4.nvidia.ops.o_proj.current_platform")
def test_o_proj_calls_triton_not_deepgemm_on_sm12x(mock_platform):
    from vllm.models.deepseek_v4.nvidia.ops import o_proj

    mock_platform.get_device_capability.return_value = DeviceCapability(
        major=12, minor=1
    )
    triton_calls: list[int] = []
    deepgemm_calls: list[int] = []
    with (
        patch.object(
            o_proj,
            "fused_inv_rope_fp8_quant",
            lambda *a, **k: (MagicMock(), MagicMock()),
        ),
        patch.object(
            o_proj,
            "_use_deepseek_v4_sm12x_triton_fp8_einsum",
            lambda *a, **k: True,
        ),
        patch.object(
            o_proj,
            "deepseek_v4_fp8_einsum",
            lambda *a, **k: triton_calls.append(1),
        ),
        patch.object(o_proj, "fp8_einsum", lambda *a, **k: deepgemm_calls.append(1)),
    ):
        o = torch.empty(2, 8, 192)
        wo_a = SimpleNamespace(
            weight=torch.empty(4, 8),
            weight_scale=torch.empty(1, dtype=torch.float32),
        )
        o_proj.deep_gemm_fp8_o_proj(
            o,
            torch.zeros(2, dtype=torch.long),
            torch.zeros(1),
            wo_a,
            lambda x: x,
            n_groups=1,
            heads_per_group=8,
            nope_dim=128,
            rope_dim=64,
            o_lora_rank=4,
            einsum_recipe=(1, 128, 128),
            tma_aligned_scales=False,
        )
    assert triton_calls == [1]
    assert deepgemm_calls == []


def test_marlin_skips_packing_is_bmm_wo_a():
    from vllm.model_executor.kernels.linear.scaled_mm.marlin import (
        MarlinFP8ScaledMMLinearKernel,
    )

    layer = torch.nn.Module()
    layer.is_bmm = True
    layer.weight = torch.nn.Parameter(torch.empty(256, 512), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(torch.empty(2, 4), requires_grad=False)
    kernel = MarlinFP8ScaledMMLinearKernel.__new__(MarlinFP8ScaledMMLinearKernel)
    kernel.block_quant = True
    packed = {"called": False}

    def _fail_pack(*args, **kwargs):
        packed["called"] = True

    with (
        patch(
            "vllm.model_executor.kernels.linear.scaled_mm.marlin."
            "process_fp8_weight_block_strategy",
            lambda w, s: (w, s),
        ),
        patch(
            "vllm.model_executor.kernels.linear.scaled_mm.marlin."
            "prepare_fp8_layer_for_marlin",
            _fail_pack,
        ),
    ):
        kernel.process_weights_after_loading(layer)
    assert packed["called"] is False
