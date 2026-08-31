# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM12x FP8 linear selection: skip Cutlass UE8M0 and Marlin weight-only."""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm.model_executor.kernels.linear.scaled_mm.BlockScaledMMLinearKernel import (
    upcast_ue8m0_weight_scale_if_needed,
)
from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
    CutlassFp8BlockScaledMMKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.marlin import (
    MarlinFP8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
    SharedExpertsOrder,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _upcast_e8m0_to_fp32,
)

pytestmark = __import__("pytest").mark.cpu_test


def test_cutlass_block_fp8_unsupported_on_sm12x():
    with patch(
        "vllm.model_executor.kernels.linear.scaled_mm.cutlass.current_platform"
    ) as mock_platform:
        mock_platform.is_device_capability_family.side_effect = lambda fam: fam == 120
        ok, reason = CutlassFp8BlockScaledMMKernel.is_supported(121)
    assert ok is False
    assert reason is not None
    assert "SM12x" in reason


def test_marlin_fp8_unsupported_on_sm12x():
    with patch(
        "vllm.model_executor.kernels.linear.scaled_mm.marlin.current_platform"
    ) as mock_platform:
        mock_platform.is_cuda.return_value = True
        mock_platform.is_device_capability_family.side_effect = lambda fam: fam == 120
        ok, reason = MarlinFP8ScaledMMLinearKernel.is_supported(121)
    assert ok is False
    assert reason is not None
    assert "SM12x" in reason


def test_shared_experts_disables_aux_stream_on_sm12x():
    platform = SimpleNamespace(
        is_cuda=lambda: True,
        is_device_capability_family=lambda fam: fam == 120,
    )
    moe_config = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(
            enable_eplb=False,
            all2all_backend="",
            use_fi_nvl_two_sided_kernels=False,
        )
    )
    with (
        patch(
            "vllm.model_executor.layers.fused_moe.runner.shared_experts."
            "current_platform",
            platform,
        ),
        patch(
            "vllm.model_executor.layers.fused_moe.runner.shared_experts.aux_stream",
            lambda: object(),
        ),
    ):
        experts = SharedExperts(
            layer=torch.nn.Identity(),
            moe_config=moe_config,
            enable_dbo=False,
            mk_can_overlap_shared_experts=lambda: False,
        )
    assert experts._stream is None
    hidden = torch.zeros(8, 16)
    assert experts._determine_shared_experts_order(hidden) is (
        SharedExpertsOrder.NO_OVERLAP
    )


def test_ue8m0_weight_scale_upcasts_to_float32():
    # 2^(0) = 1.0 stored as biased exponent 127.
    raw = torch.tensor([127], dtype=torch.uint8)
    layer = torch.nn.Module()
    layer.weight_scale_inv = torch.nn.Parameter(
        raw.view(torch.float8_e8m0fnu), requires_grad=False
    )
    upcast_ue8m0_weight_scale_if_needed(layer, "weight_scale_inv")
    assert layer.weight_scale_inv.dtype == torch.float32
    assert torch.allclose(layer.weight_scale_inv, torch.tensor([1.0]))
    assert torch.allclose(_upcast_e8m0_to_fp32(raw), torch.tensor([1.0]))
