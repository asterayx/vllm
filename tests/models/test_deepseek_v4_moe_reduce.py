# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM12x: the fused MoE all-reduce covers only the real token rows."""

import pytest
import torch

from vllm.models.deepseek_v4.nvidia import model as dsv4
from vllm.utils import sm12x as sm12x_utils


class _Experts:
    def __init__(self):
        self.calls: list[tuple[int, ...]] = []

    def __call__(self, hidden_states, router_logits, input_ids):
        self.calls.append(tuple(hidden_states.shape))
        return hidden_states * 2


@pytest.fixture
def moe(monkeypatch):
    monkeypatch.setattr(
        sm12x_utils.current_platform,
        "is_device_capability_family",
        lambda fam: fam == 120,
    )
    layer = dsv4.DeepseekV4MoE.__new__(dsv4.DeepseekV4MoE)
    layer.experts = _Experts()
    layer._reduce_real_rows = True
    return layer


def test_fused_moe_reduces_only_real_rows(monkeypatch, moe):
    reduced: list[tuple[int, ...]] = []

    def fake_all_reduce(t):
        reduced.append(tuple(t.shape))
        return t + 1

    monkeypatch.setattr(dsv4, "tensor_model_parallel_all_reduce", fake_all_reduce)
    hidden = torch.ones(4, 8, dtype=torch.bfloat16)
    input_ids = torch.arange(4)

    out = moe._forward_fused_moe(hidden, input_ids)

    assert moe.experts.calls == [(16, 8)], "experts still run on the padded block"
    assert reduced == [(4, 8)], "only the real rows are all-reduced"
    assert out.shape == (4, 8)
    torch.testing.assert_close(out, torch.full((4, 8), 3.0, dtype=torch.bfloat16))


def test_fused_moe_leaves_reduce_to_fused_moe_when_disabled(monkeypatch, moe):
    moe._reduce_real_rows = False
    monkeypatch.setattr(
        dsv4,
        "tensor_model_parallel_all_reduce",
        lambda t: pytest.fail("must not reduce here"),
    )
    out = moe._forward_fused_moe(torch.ones(4, 8, dtype=torch.bfloat16), None)
    assert out.shape == (4, 8)
    assert moe.experts.calls == [(16, 8)]
