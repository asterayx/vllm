# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from torch import nn

from vllm.model_executor.layers.fused_moe.utils import (
    is_model_fused_shared_expert_compatible,
)


class _Moe(nn.Module):
    def __init__(self, enabled: bool | None) -> None:
        super().__init__()
        self.is_fused_shared_expert_enabled = enabled


class _Layer(nn.Module):
    def __init__(self, enabled: bool | None) -> None:
        super().__init__()
        self.mlp = _Moe(enabled)


def test_fused_shared_expert_compat_treats_none_as_disabled() -> None:
    layers = nn.ModuleList([_Layer(None), _Layer(None)])
    assert is_model_fused_shared_expert_compatible(layers, _Moe, "mlp") is False


def test_fused_shared_expert_compat_requires_all_layers_enabled() -> None:
    layers = nn.ModuleList([_Layer(True), _Layer(True)])
    assert is_model_fused_shared_expert_compatible(layers, _Moe, "mlp") is True
