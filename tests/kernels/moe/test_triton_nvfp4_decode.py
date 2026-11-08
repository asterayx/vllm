# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the DSv4.1 Triton NVFP4 decode MoE opt-in and tile picks."""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.b12x import B12xExperts
from vllm.model_executor.layers.fused_moe.config import nvfp4_moe_quant_config
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import MarlinExperts
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    NvFp4MoeBackend,
    select_nvfp4_moe_backend,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.triton_nvfp4_decode import (
    DSV41_DECODE_MS,
    DSV41_FLASH_EXPERTS,
    DSV41_FLASH_HIDDEN,
    DSV41_FLASH_INTERMEDIATE_TP4,
    DSV41_FLASH_TOPK,
    SM120_NUM_SMS,
    TritonNvfp4DecodeMoE,
    cfg_divides_shapes,
    choose_triton_nvfp4_decode_cfg,
    decode_occupancy_rows,
    dsv41_triton_nvfp4_decode_requested,
    normalize_swiglu_clamp_limit,
    parse_tile_override,
    silu_mul_with_optional_clamp,
    triton_nvfp4_decode_overlay_opt_in,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.models.deepseek_v41.quant_config import DeepseekV4FP8Config


def test_dsv41_flag_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DSV41_EXP_TRITON_MOE", raising=False)
    monkeypatch.delenv("QWEN4_EXP_TRITON_MOE", raising=False)
    assert dsv41_triton_nvfp4_decode_requested() is False
    enabled, max_m, flag, profile = triton_nvfp4_decode_overlay_opt_in()
    assert (enabled, max_m, flag, profile) == (False, 0, "", "default")


def test_qwen_overlay_opt_in_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DSV41_EXP_TRITON_MOE", raising=False)
    monkeypatch.setenv("QWEN4_EXP_TRITON_MOE", "1")
    enabled, max_m, flag, profile = triton_nvfp4_decode_overlay_opt_in()
    assert enabled is True
    assert max_m == 96
    assert flag == "QWEN4_EXP_TRITON_MOE"
    assert profile == "default"


def test_dsv41_overlay_opt_in_parses_max_m(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DSV41_EXP_TRITON_MOE", "1")
    monkeypatch.setenv("DSV41_EXP_TRITON_MOE_MAX_M", "24")
    enabled, max_m, flag, profile = triton_nvfp4_decode_overlay_opt_in()
    assert enabled is True
    assert max_m == 24
    assert flag == "DSV41_EXP_TRITON_MOE"
    assert profile == "dsv41_sm120"


def test_dsv41_flag_ignored_for_other_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DSV41_EXP_TRITON_MOE", "1")
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.triton_nvfp4_decode."
        "current_hf_model_type",
        lambda: "qwen4_exp",
    )
    assert dsv41_triton_nvfp4_decode_requested() is False


def test_dsv41_flag_honored_for_deepseek_v41(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DSV41_EXP_TRITON_MOE", "1")
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.triton_nvfp4_decode."
        "current_hf_model_type",
        lambda: "deepseek_v41",
    )
    assert dsv41_triton_nvfp4_decode_requested() is True


def test_sm120_dsv41_tiles_divide_flash_tp4() -> None:
    cfg = choose_triton_nvfp4_decode_cfg(
        DSV41_FLASH_HIDDEN,
        DSV41_FLASH_INTERMEDIATE_TP4,
        profile="dsv41_sm120",
        num_sms=SM120_NUM_SMS,
        topk=DSV41_FLASH_TOPK,
        num_experts=DSV41_FLASH_EXPERTS,
    )
    assert cfg["BN1"] == 32
    assert cfg["BN2"] == 128
    assert cfg["BK1"] == 512
    assert cfg["BK2"] == 64
    assert cfg_divides_shapes(cfg, DSV41_FLASH_HIDDEN, DSV41_FLASH_INTERMEDIATE_TP4)
    rows = decode_occupancy_rows(
        cfg,
        DSV41_FLASH_HIDDEN,
        DSV41_FLASH_INTERMEDIATE_TP4,
        topk=DSV41_FLASH_TOPK,
        num_sms=SM120_NUM_SMS,
        num_experts=DSV41_FLASH_EXPERTS,
        token_counts=DSV41_DECODE_MS,
    )
    by_m = {int(row["M"]): row for row in rows}
    # One request at DSpark k=3 is about M=4; FC1 should cover 110 SMs.
    assert by_m[4]["fc1_ctas"] >= SM120_NUM_SMS
    for m in DSV41_DECODE_MS:
        assert by_m[m]["fc1_ctas"] >= SM120_NUM_SMS
        assert by_m[m]["fc2_ctas"] >= SM120_NUM_SMS


def test_sm120_bn1_shrinks_when_m4_underfills_sms() -> None:
    # I=32, BN1=32 -> 24 CTAs at M=4/topk=6, below 110 SMs, so BN1 becomes 16.
    cfg = choose_triton_nvfp4_decode_cfg(
        512,
        32,
        profile="dsv41_sm120",
        num_sms=SM120_NUM_SMS,
        topk=6,
        num_experts=384,
    )
    assert cfg["BN1"] == 16


def test_tile_override_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DSV41_EXP_TRITON_MOE_TILE", "BN1=16,S1=3")
    cfg = choose_triton_nvfp4_decode_cfg(
        DSV41_FLASH_HIDDEN,
        DSV41_FLASH_INTERMEDIATE_TP4,
        profile="dsv41_sm120",
        num_sms=SM120_NUM_SMS,
    )
    assert cfg["BN1"] == 16
    assert cfg["S1"] == 3
    assert parse_tile_override("BM=8", {"BM": 16, "BN1": 32})["BM"] == 8
    with pytest.raises(ValueError, match="unknown"):
        parse_tile_override("NOPE=1", {"BM": 16})


def test_auto_without_flag_does_not_select_b12x(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DSV41_EXP_TRITON_MOE", raising=False)
    monkeypatch.setattr(
        B12xExperts, "is_supported_config", lambda *args, **kwargs: (True, None)
    )
    config = make_dummy_moe_config(hidden_dim=5120, intermediate_size=576)
    config.moe_backend = "auto"
    try:
        backend, _ = select_nvfp4_moe_backend(
            config, weight_key=kNvfp4Static, activation_key=kNvfp4Dynamic
        )
    except (ValueError, NotImplementedError):
        return
    assert backend != NvFp4MoeBackend.B12X


def test_dsv41_flag_selects_b12x_on_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DSV41_EXP_TRITON_MOE", "1")
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.triton_nvfp4_decode."
        "current_hf_model_type",
        lambda: "deepseek_v41",
    )
    monkeypatch.setattr(
        B12xExperts, "is_supported_config", lambda *args, **kwargs: (True, None)
    )
    config = make_dummy_moe_config(hidden_dim=5120, intermediate_size=576)
    config.moe_backend = "auto"
    backend, experts_cls = select_nvfp4_moe_backend(
        config, weight_key=kNvfp4Static, activation_key=kNvfp4Dynamic
    )
    assert backend == NvFp4MoeBackend.B12X
    assert experts_cls is B12xExperts


def test_explicit_marlin_wins_over_dsv41_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DSV41_EXP_TRITON_MOE", "1")
    monkeypatch.setattr(
        B12xExperts, "is_supported_config", lambda *args, **kwargs: (True, None)
    )
    monkeypatch.setattr(
        MarlinExperts, "is_supported_config", lambda *args, **kwargs: (True, None)
    )
    config = make_dummy_moe_config(hidden_dim=5120, intermediate_size=576)
    config.moe_backend = "marlin"
    backend, experts_cls = select_nvfp4_moe_backend(
        config, weight_key=kNvfp4Static, activation_key=kNvfp4Dynamic
    )
    assert backend == NvFp4MoeBackend.MARLIN
    assert experts_cls is MarlinExperts


def test_dspark_draft_stays_mxfp4_when_triton_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DSV41_EXP_TRITON_MOE", "1")

    class FakeMxfp4:
        def __init__(self, moe_config):
            self.moe_config = moe_config

    monkeypatch.setattr(
        "vllm.models.deepseek_v41.quant_config.Mxfp4MoEMethod",
        FakeMxfp4,
    )
    quant_config = DeepseekV4FP8Config(
        is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128]
    )
    quant_config._resolved_expert_dtype = "fp4"
    quant_config._resolved_moe_quant_algo = "NVFP4"
    quant_config.dspark_draft_moe_mxfp4 = True
    layer = MagicMock(spec=RoutedExperts)
    layer.moe_config = make_dummy_moe_config()
    method = quant_config.get_quant_method(layer, "mtp.layers.0.experts")
    assert isinstance(method, FakeMxfp4)


def _fake_prepared(num_experts: int, hidden: int, intermediate: int):
    e, h, i = num_experts, hidden, intermediate
    c41 = (h // 16 + 3) // 4
    c42 = (i // 16 + 3) // 4
    s1_cols = ((2 * i + 127) // 128 * 128) * c41 * 4
    s2_cols = ((h + 127) // 128 * 128) * c42 * 4
    return SimpleNamespace(
        num_experts=e,
        hidden_size=h,
        intermediate_size=i,
        w13_layout="w13",
        w1_fp4=torch.zeros(e, 2 * i, h // 2, dtype=torch.uint8),
        w2_fp4=torch.zeros(e, h, i // 2, dtype=torch.uint8),
        w1_blockscale=torch.zeros(e, s1_cols, dtype=torch.float8_e4m3fn),
        w2_blockscale=torch.zeros(e, s2_cols, dtype=torch.float8_e4m3fn),
        w1_alphas=torch.ones(e, dtype=torch.float32),
        w2_alphas=torch.ones(e, dtype=torch.float32),
        a1_gscale=torch.ones(e, dtype=torch.float32),
        a2_gscale=torch.ones(e, dtype=torch.float32),
    )


def test_runner_accepts_dsv41_tp4_tiles() -> None:
    # E=8 keeps M=4 occupancy above 110 SMs so BN1 stays on the 32-wide tile.
    prepared = _fake_prepared(8, DSV41_FLASH_HIDDEN, DSV41_FLASH_INTERMEDIATE_TP4)
    runner = TritonNvfp4DecodeMoE(
        prepared, topk=DSV41_FLASH_TOPK, max_m=24, profile="dsv41_sm120"
    )
    assert runner.H == DSV41_FLASH_HIDDEN
    assert runner.I == DSV41_FLASH_INTERMEDIATE_TP4
    assert runner.c["BN1"] == 32
    assert runner.swiglu_limit is None
    assert cfg_divides_shapes(runner.c, runner.H, runner.I)


def test_normalize_swiglu_clamp_limit() -> None:
    assert normalize_swiglu_clamp_limit(None) is None
    assert normalize_swiglu_clamp_limit(0.0) is None
    assert normalize_swiglu_clamp_limit(-1.0) is None
    assert normalize_swiglu_clamp_limit(10.0) == 10.0


def test_silu_mul_clamp_matches_marlin_semantics() -> None:
    torch.manual_seed(0)
    n, d = 64, 32
    gate = torch.randn(n, d) * 20.0
    up = torch.randn(n, d) * 20.0
    # Marlin/b12x SILU: gate.clamp(max=L), up.clamp(-L, L), silu(gate)*up.
    g = torch.clamp(gate.float(), max=10.0)
    u = torch.clamp(up.float(), min=-10.0, max=10.0)
    ref = g * torch.sigmoid(g) * u
    got = silu_mul_with_optional_clamp(gate, up, limit=10.0)
    torch.testing.assert_close(got, ref, atol=0.0, rtol=0.0)

    unclamped = silu_mul_with_optional_clamp(gate, up, limit=None)
    assert not torch.allclose(got, unclamped)
    # Gate is max-clamped only; up is clamped to [-L, L].
    inner = (gate <= 10.0) & (up.abs() <= 10.0)
    if inner.any():
        torch.testing.assert_close(got[inner], unclamped[inner], atol=0.0, rtol=0.0)


def test_runner_stores_dsv41_swiglu_limit() -> None:
    prepared = _fake_prepared(8, DSV41_FLASH_HIDDEN, DSV41_FLASH_INTERMEDIATE_TP4)
    runner = TritonNvfp4DecodeMoE(
        prepared,
        topk=DSV41_FLASH_TOPK,
        max_m=24,
        profile="dsv41_sm120",
        swiglu_limit=10.0,
    )
    assert runner.swiglu_limit == 10.0


def _nvfp4_quant_config(clamp_limit: float | None = None):
    scale = torch.ones(1)
    return nvfp4_moe_quant_config(
        g1_alphas=scale,
        g2_alphas=scale,
        a1_gscale=scale,
        a2_gscale=scale,
        w1_scale=scale,
        w2_scale=scale,
        gemm1_clamp_limit=clamp_limit,
    )


def test_tri_setup_attaches_with_dsv41_swiglu_limit(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("DSV41_EXP_TRITON_MOE", "1")
    monkeypatch.delenv("QWEN4_EXP_TRITON_MOE", raising=False)
    monkeypatch.setattr(B12xExperts, "_tri_logged", False)
    moe = make_dummy_moe_config(
        num_experts=8,
        experts_per_token=DSV41_FLASH_TOPK,
        hidden_dim=DSV41_FLASH_HIDDEN,
        intermediate_size=DSV41_FLASH_INTERMEDIATE_TP4,
        in_dtype=torch.bfloat16,
    )
    moe.swiglu_limit = 10.0
    experts = B12xExperts(moe, _nvfp4_quant_config())
    layer = SimpleNamespace(
        activation=MoEActivation.SILU,
        apply_router_weight_on_input=False,
    )
    prepared = _fake_prepared(8, DSV41_FLASH_HIDDEN, DSV41_FLASH_INTERMEDIATE_TP4)
    with caplog.at_level(
        logging.INFO, logger="vllm.model_executor.layers.fused_moe.b12x"
    ):
        experts._tri_setup(layer, prepared)
    assert experts._tri is not None
    assert experts._tri.swiglu_limit == 10.0
    assert "Triton NVFP4 decode MoE" in caplog.text
    assert "swiglu clamp limit set" not in caplog.text


def test_tri_setup_attaches_when_clamp_comes_from_quant_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DSV41_EXP_TRITON_MOE", "1")
    monkeypatch.setattr(B12xExperts, "_tri_logged", True)
    moe = make_dummy_moe_config(
        num_experts=8,
        experts_per_token=DSV41_FLASH_TOPK,
        hidden_dim=DSV41_FLASH_HIDDEN,
        intermediate_size=DSV41_FLASH_INTERMEDIATE_TP4,
        in_dtype=torch.bfloat16,
    )
    experts = B12xExperts(moe, _nvfp4_quant_config(clamp_limit=10.0))
    layer = SimpleNamespace(activation=MoEActivation.SILU)
    prepared = _fake_prepared(8, DSV41_FLASH_HIDDEN, DSV41_FLASH_INTERMEDIATE_TP4)
    experts._tri_setup(layer, prepared)
    assert experts._tri is not None
    assert experts._tri.swiglu_limit == 10.0


def test_tri_setup_still_refuses_swigluoai(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("DSV41_EXP_TRITON_MOE", "1")
    monkeypatch.setattr(B12xExperts, "_tri_logged", False)
    moe = make_dummy_moe_config(
        num_experts=8,
        experts_per_token=DSV41_FLASH_TOPK,
        hidden_dim=DSV41_FLASH_HIDDEN,
        intermediate_size=DSV41_FLASH_INTERMEDIATE_TP4,
        in_dtype=torch.bfloat16,
        activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE,
    )
    experts = B12xExperts(moe, _nvfp4_quant_config(clamp_limit=10.0))
    layer = SimpleNamespace(activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE)
    prepared = _fake_prepared(8, DSV41_FLASH_HIDDEN, DSV41_FLASH_INTERMEDIATE_TP4)
    with caplog.at_level(
        logging.WARNING, logger="vllm.model_executor.layers.fused_moe.b12x"
    ):
        experts._tri_setup(layer, prepared)
    assert experts._tri is None
    assert "activation=" in caplog.text
    assert "swiglu clamp limit set" not in caplog.text


def test_tri_setup_still_refuses_nonsilu(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("DSV41_EXP_TRITON_MOE", "1")
    monkeypatch.setattr(B12xExperts, "_tri_logged", False)
    moe = make_dummy_moe_config(
        num_experts=8,
        experts_per_token=DSV41_FLASH_TOPK,
        hidden_dim=DSV41_FLASH_HIDDEN,
        intermediate_size=DSV41_FLASH_INTERMEDIATE_TP4,
        in_dtype=torch.bfloat16,
    )
    experts = B12xExperts(moe, _nvfp4_quant_config())
    layer = SimpleNamespace(activation=MoEActivation.GELU)
    prepared = _fake_prepared(8, DSV41_FLASH_HIDDEN, DSV41_FLASH_INTERMEDIATE_TP4)
    with caplog.at_level(
        logging.WARNING, logger="vllm.model_executor.layers.fused_moe.b12x"
    ):
        experts._tri_setup(layer, prepared)
    assert experts._tri is None
    assert "activation=" in caplog.text


def test_tri_setup_skipped_when_flag_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DSV41_EXP_TRITON_MOE", raising=False)
    monkeypatch.delenv("QWEN4_EXP_TRITON_MOE", raising=False)
    moe = make_dummy_moe_config(
        num_experts=8,
        experts_per_token=DSV41_FLASH_TOPK,
        hidden_dim=DSV41_FLASH_HIDDEN,
        intermediate_size=DSV41_FLASH_INTERMEDIATE_TP4,
        in_dtype=torch.bfloat16,
    )
    moe.swiglu_limit = 10.0
    experts = B12xExperts(moe, _nvfp4_quant_config())
    layer = SimpleNamespace(activation=MoEActivation.SILU)
    prepared = _fake_prepared(8, DSV41_FLASH_HIDDEN, DSV41_FLASH_INTERMEDIATE_TP4)
    experts._tri_setup(layer, prepared)
    assert experts._tri_enabled is False
    assert experts._tri is None
