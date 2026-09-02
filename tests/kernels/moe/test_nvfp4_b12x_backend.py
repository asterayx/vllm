# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-safe NVFP4 oracle wiring for official --moe-backend b12x."""

import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.b12x import B12xExperts
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    NvFp4MoeBackend,
    backend_to_kernel_cls,
    map_nvfp4_backend,
    select_nvfp4_moe_backend,
)


def test_map_nvfp4_backend_accepts_official_b12x():
    assert map_nvfp4_backend("b12x") is NvFp4MoeBackend.B12X


def test_nvfp4_b12x_kernel_cls():
    assert backend_to_kernel_cls(NvFp4MoeBackend.B12X) == [B12xExperts]


def test_select_nvfp4_allows_b12x_with_swiglu_limit(monkeypatch):
    class SupportedExperts:
        @staticmethod
        def is_supported_config(*args, **kwargs):
            return True, None

    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.oracle.nvfp4.backend_to_kernel_cls",
        lambda backend: [SupportedExperts],
    )
    cfg = FusedMoEConfig(
        num_experts=1,
        experts_per_token=1,
        hidden_dim=256,
        intermediate_size=256,
        num_local_experts=1,
        num_logical_experts=1,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device="cpu",
        routing_method=RoutingMethodType.TopK,
        moe_backend="b12x",
        swiglu_limit=7.0,
    )
    selected, _ = select_nvfp4_moe_backend(cfg, None, None)
    assert selected is NvFp4MoeBackend.B12X
