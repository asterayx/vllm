# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM12x must not pick Cutlass block FP8 for DSv4 UE8M0 scales."""

from unittest.mock import patch

import torch

from vllm.model_executor.kernels.linear.scaled_mm.BlockScaledMMLinearKernel import (
    upcast_ue8m0_weight_scale_if_needed,
)
from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
    CutlassFp8BlockScaledMMKernel,
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
