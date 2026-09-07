# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM12x split-K skinny GEMM: plan, dispatch gating and numerics.

The CPU cases run the Triton interpreter in fp16/fp32 (the interpreter has
no bf16 arithmetic); the bf16 case needs a GPU.
"""

import importlib

import pytest
import torch

from vllm.model_executor.kernels.linear import sm12x_skinny_gemm as sg


@pytest.fixture
def interpreted(monkeypatch):
    triton = pytest.importorskip("triton")
    import triton.language as tl

    import vllm.triton_utils as tu

    monkeypatch.setenv("TRITON_INTERPRET", "1")
    monkeypatch.setattr(tu, "triton", triton)
    monkeypatch.setattr(tu, "tl", tl)
    mod = importlib.reload(sg)
    yield mod
    monkeypatch.undo()
    importlib.reload(sg)


@pytest.mark.parametrize(
    ("n", "k", "expected"),
    [
        # gate: 4 N-blocks -> 32 splits of one 128-wide K block
        (256, 4096, (32, 128)),
        # compressor: 16 N-blocks -> 8 splits of 512
        (1024, 4096, (8, 512)),
        # tiny K never splits below one block; big N never splits
        (64, 128, (1, 128)),
        (65536, 4096, (1, 4096)),
        # non-multiple K still covers every column
        (200, 1000, (8, 128)),
    ],
)
def test_split_plan_covers_k(n, k, expected):
    split_k, k_per_split = sg.skinny_gemm_split_plan(n, k)
    assert (split_k, k_per_split) == expected
    assert split_k * k_per_split >= k
    assert (split_k - 1) * k_per_split < k


def test_applicable_requires_skinny_contiguous_bf16():
    x = torch.zeros(16, 4096, dtype=torch.bfloat16)
    w = torch.zeros(256, 4096, dtype=torch.bfloat16)
    if not torch.cuda.is_available():
        assert not sg.skinny_gemm_applicable(x, w)
        return
    x, w = x.cuda(), w.cuda()
    assert sg.skinny_gemm_applicable(x, w)
    assert not sg.skinny_gemm_applicable(x.repeat(3, 1), w)
    assert not sg.skinny_gemm_applicable(x.float(), w)
    assert not sg.skinny_gemm_applicable(x, w.T)
    assert not sg.skinny_gemm_applicable(x, w[:, :4095])


@pytest.mark.parametrize(
    ("m", "n", "k"), [(16, 256, 4096), (4, 1024, 4096), (32, 64, 512), (7, 200, 1000)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_interpreted_matches_reference(interpreted, m, n, k, dtype):
    torch.manual_seed(0)
    x = torch.randn(m, k, dtype=dtype)
    w = torch.randn(n, k, dtype=dtype)
    ref = x.float() @ w.float().T
    out = interpreted.skinny_gemm(x, w, torch.float32)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-2)
    out16 = interpreted.skinny_gemm(x, w, torch.float16)
    assert out16.dtype == torch.float16
    torch.testing.assert_close(out16.float(), ref, rtol=2e-2, atol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize(
    ("m", "n", "k"), [(1, 256, 4096), (16, 256, 4096), (16, 1024, 4096), (32, 64, 4096)]
)
def test_cuda_bf16_matches_cublas(m, n, k):
    torch.manual_seed(0)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    ref = torch.mm(x, w.T, out_dtype=torch.float32)
    out = sg.skinny_gemm(x, w, torch.float32)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-2)
    again = sg.skinny_gemm(x, w, torch.float32)
    assert torch.equal(out, again), "split-K reduction must be deterministic"


def test_gate_linear_tier_gated_by_platform(monkeypatch):
    from vllm.model_executor.layers.fused_moe.router import gate_linear
    from vllm.utils import sm12x

    monkeypatch.setattr(sm12x.envs, "VLLM_SM12X_SKINNY_GEMM", True)
    monkeypatch.setattr(sm12x.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        sm12x.current_platform, "is_device_capability_family", lambda f: f == 120
    )
    assert gate_linear.sm12x_use_skinny_gemm()
    monkeypatch.setattr(sm12x.envs, "VLLM_SM12X_SKINNY_GEMM", False)
    assert not gate_linear.sm12x_use_skinny_gemm()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_attention_mm_helper_matches_reference():
    from vllm.models.deepseek_v4 import attention

    x = torch.randn(4, 64, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(8, 64, dtype=torch.bfloat16, device="cuda")
    out = attention._bf16_mm_fp32(x, w, skinny=True)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, x.float() @ w.float().T, rtol=1e-2, atol=1e-2)
