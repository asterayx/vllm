# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types

import torch

from vllm.model_executor.kernels.mhc.tilelang import mhc_pre_broadcast_tilelang


def test_mhc_pre_broadcast_falls_back_without_deep_gemm(monkeypatch):
    """GB10 / stock SM12x must not call DeepGEMM prenorm when it is unsupported."""
    monkeypatch.setattr("vllm.utils.deep_gemm.is_deep_gemm_supported", lambda: False)

    def _boom(*_args, **_kwargs):
        raise AssertionError("tf32_hc_prenorm_gemm should not run")

    monkeypatch.setattr("vllm.utils.deep_gemm.tf32_hc_prenorm_gemm", _boom)

    fuse_called = {"value": False}

    def _fake_fuse(*_args, **_kwargs):
        fuse_called["value"] = True

    fake_kernels = types.ModuleType("vllm.model_executor.kernels.mhc.tilelang_kernels")
    fake_kernels.compute_num_split = lambda *_args, **_kwargs: 1
    fake_kernels.mhc_pre_big_fuse_broadcast_with_norm_tilelang = _fake_fuse
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.kernels.mhc.tilelang_kernels",
        fake_kernels,
    )

    num_tokens = 2
    hidden_size = 512
    hc_mult = 4
    hc_mult3 = 2 * hc_mult + hc_mult**2

    residual = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16)
    fn = torch.randn(hc_mult3, hc_mult * hidden_size, dtype=torch.float32)
    fn_broadcast = fn.view(hc_mult3, hc_mult, hidden_size).sum(dim=1)
    hc_scale = torch.ones(3, dtype=torch.float32)
    hc_base = torch.zeros(hc_mult3, dtype=torch.float32)
    norm_weight = torch.ones(hidden_size, dtype=torch.bfloat16)

    mhc_pre_broadcast_tilelang(
        residual,
        fn,
        hc_scale,
        hc_base,
        1e-5,
        1e-5,
        1e-5,
        1.0,
        1,
        norm_weight=norm_weight,
        fn_broadcast=fn_broadcast,
    )

    assert fuse_called["value"]
