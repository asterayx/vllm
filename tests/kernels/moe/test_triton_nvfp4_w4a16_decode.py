# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the opt-in DSv4.1 NVFP4 W4A16 small-M overlay.

The Triton GEMM is GPU-only; those cases skip without CUDA. Dispatch,
snapshot, clamp-gating, and occupancy are CPU.
"""

from types import SimpleNamespace

import pytest
import torch

from tests.kernels.moe.utils import make_dummy_moe_config
from tests.kernels.quantization.nvfp4_utils import break_fp4_bytes
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    NvFp4MoeBackend,
    convert_to_nvfp4_moe_kernel_format,
    select_nvfp4_moe_backend,
)
from vllm.model_executor.layers.fused_moe.triton_nvfp4_w4a16_decode import (
    DSV41_FLASH_TP4_H,
    DSV41_FLASH_TP4_I,
    ENV_FLAG,
    ENV_MAX_M,
    LAYER_PACK_ATTR,
    Nvfp4W4A16DecodeMoE,
    Nvfp4W4A16DecodePack,
    fused_w4a16_decode_enabled,
    fused_w4a16_decode_max_m,
    maybe_snapshot_nvfp4_w4a16_decode_weights,
    occupancy_rows,
    overlay_disable_reason,
    try_attach_w4a16_decode,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kNvfp4Static,
)


def _make_pack(
    num_experts: int = 4,
    intermediate_size: int = 64,
    hidden_size: int = 128,
) -> Nvfp4W4A16DecodePack:
    e, i, h = num_experts, intermediate_size, hidden_size
    return Nvfp4W4A16DecodePack(
        w13=torch.zeros(e, 2 * i, h // 2, dtype=torch.uint8),
        w13_scale=torch.zeros(e, 2 * i, h // 16, dtype=torch.float8_e4m3fn),
        w13_scale_2=torch.ones(e, dtype=torch.float32),
        w2=torch.zeros(e, h, i // 2, dtype=torch.uint8),
        w2_scale=torch.zeros(e, h, i // 16, dtype=torch.float8_e4m3fn),
        w2_scale_2=torch.ones(e, dtype=torch.float32),
    )


def _ok_kwargs(pack: Nvfp4W4A16DecodePack | None = None, **overrides):
    kwargs = dict(
        pack=pack if pack is not None else _make_pack(),
        max_m=64,
        in_dtype=torch.bfloat16,
        activation=MoEActivation.SILU,
        clamp_limit=10.0,
        apply_router_weight_on_input=False,
        has_bias=False,
        is_lora=False,
    )
    kwargs.update(overrides)
    return kwargs


def test_flag_default_off(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(ENV_FLAG, raising=False)
    monkeypatch.delenv(ENV_MAX_M, raising=False)
    assert fused_w4a16_decode_enabled() is False
    assert fused_w4a16_decode_max_m() == 64
    assert overlay_disable_reason(**_ok_kwargs()) == "flag unset"


def test_flag_on_and_max_m(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(ENV_FLAG, "1")
    monkeypatch.setenv(ENV_MAX_M, "24")
    assert fused_w4a16_decode_enabled() is True
    assert fused_w4a16_decode_max_m() == 24
    assert overlay_disable_reason(**_ok_kwargs()) is None


def test_refuses_without_swiglu_limit(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(ENV_FLAG, "1")
    reason = overlay_disable_reason(**_ok_kwargs(clamp_limit=None))
    assert reason == "swiglu_limit is unset"


def test_refuses_non_silu_and_router_weight(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(ENV_FLAG, "1")
    assert "activation=" in overlay_disable_reason(
        **_ok_kwargs(activation=MoEActivation.GELU)
    )
    assert (
        overlay_disable_reason(**_ok_kwargs(apply_router_weight_on_input=True))
        == "apply_router_weight_on_input"
    )


def _snapshot_kwargs(pack: Nvfp4W4A16DecodePack):
    return dict(
        w13=pack.w13,
        w13_scale=pack.w13_scale,
        w13_scale_2=pack.w13_scale_2,
        w2=pack.w2,
        w2_scale=pack.w2_scale,
        w2_scale_2=pack.w2_scale_2,
    )


def _track_full_pack_clones(
    monkeypatch: pytest.MonkeyPatch, pack: Nvfp4W4A16DecodePack
) -> list[tuple[int, ...]]:
    """Record `.clone()` of the full w13/w2/scale tensors (the OOM peak)."""
    watched = {
        pack.w13.data_ptr(),
        pack.w13_scale.data_ptr(),
        pack.w2.data_ptr(),
        pack.w2_scale.data_ptr(),
    }
    cloned_shapes: list[tuple[int, ...]] = []
    orig_clone = torch.Tensor.clone

    def tracking_clone(self, *args, **kwargs):
        if self.data_ptr() in watched:
            cloned_shapes.append(tuple(self.shape))
        return orig_clone(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", tracking_clone)
    return cloned_shapes


def test_snapshot_only_when_flag_on(monkeypatch: pytest.MonkeyPatch):
    pack = _make_pack()
    layer = SimpleNamespace()
    monkeypatch.delenv(ENV_FLAG, raising=False)
    cloned = _track_full_pack_clones(monkeypatch, pack)
    assert (
        maybe_snapshot_nvfp4_w4a16_decode_weights(layer, **_snapshot_kwargs(pack))
        is None
    )
    assert not hasattr(layer, LAYER_PACK_ATTR)
    assert cloned == []

    monkeypatch.setenv(ENV_FLAG, "1")
    snapped = maybe_snapshot_nvfp4_w4a16_decode_weights(
        layer,
        **_snapshot_kwargs(pack),
    )
    assert snapped is not None
    assert getattr(layer, LAYER_PACK_ATTR) is snapped
    assert snapped.shares_weight_storage(
        pack.w13, pack.w13_scale, pack.w2, pack.w2_scale
    )
    assert cloned == []


def test_marlin_convert_snapshots_after_shuffle_without_clone(
    monkeypatch: pytest.MonkeyPatch,
):
    pack = _make_pack()
    layer = SimpleNamespace()
    pack_during_prepare: object | None = "unset"
    cloned = _track_full_pack_clones(monkeypatch, pack)

    def fake_prepare(**kwargs):
        nonlocal pack_during_prepare
        pack_during_prepare = getattr(layer, LAYER_PACK_ATTR, None)
        shuffled = kwargs["w13"] + 1
        return (
            shuffled,
            kwargs["w13_scale"],
            kwargs["w13_scale_2"],
            kwargs["w2"],
            kwargs["w2_scale"],
            kwargs["w2_scale_2"],
        )

    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.oracle.nvfp4."
        "prepare_nvfp4_moe_layer_for_marlin",
        fake_prepare,
    )
    monkeypatch.setenv(ENV_FLAG, "1")
    w13, *_rest = convert_to_nvfp4_moe_kernel_format(
        nvfp4_backend=NvFp4MoeBackend.MARLIN,
        layer=layer,
        w13=pack.w13,
        w13_scale=pack.w13_scale,
        w13_scale_2=pack.w13_scale_2,
        a13_scale=None,
        w2=pack.w2,
        w2_scale=pack.w2_scale,
        w2_scale_2=pack.w2_scale_2,
        a2_scale=None,
        is_act_and_mul=True,
    )
    assert pack_during_prepare is None
    snapped = getattr(layer, LAYER_PACK_ATTR)
    assert torch.equal(snapped.w13, pack.w13)
    assert not torch.equal(w13, pack.w13)
    assert snapped.shares_weight_storage(
        pack.w13, pack.w13_scale, pack.w2, pack.w2_scale
    )
    assert cloned == []

    monkeypatch.delenv(ENV_FLAG, raising=False)
    layer2 = SimpleNamespace()
    convert_to_nvfp4_moe_kernel_format(
        nvfp4_backend=NvFp4MoeBackend.MARLIN,
        layer=layer2,
        w13=pack.w13,
        w13_scale=pack.w13_scale,
        w13_scale_2=pack.w13_scale_2,
        a13_scale=None,
        w2=pack.w2,
        w2_scale=pack.w2_scale,
        w2_scale_2=pack.w2_scale_2,
        a2_scale=None,
        is_act_and_mul=True,
    )
    assert not hasattr(layer2, LAYER_PACK_ATTR)
    assert cloned == []


def test_w4a16_auto_stays_marlin_with_flag(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
        MarlinExperts,
    )

    monkeypatch.setenv(ENV_FLAG, "1")
    monkeypatch.setattr(MarlinExperts, "_supports_current_device", lambda: True)
    config = make_dummy_moe_config(
        num_experts=4,
        experts_per_token=2,
        hidden_dim=128,
        intermediate_size=64,
    )
    config.swiglu_limit = 10.0
    backend, experts_cls = select_nvfp4_moe_backend(
        config,
        weight_key=kNvfp4Static,
        activation_key=None,
    )
    assert backend == NvFp4MoeBackend.MARLIN
    assert experts_cls is MarlinExperts


def test_try_attach_logs_and_requires_clamp(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    import vllm.model_executor.layers.fused_moe.triton_nvfp4_w4a16_decode as mod

    monkeypatch.setenv(ENV_FLAG, "1")
    monkeypatch.setattr(mod, "_ATTACH_LOGGED", False)
    pack = _make_pack()
    layer = SimpleNamespace(apply_router_weight_on_input=False)
    setattr(layer, LAYER_PACK_ATTR, pack)
    experts = SimpleNamespace(
        activation_config=SimpleNamespace(clamp_limit=10.0),
        moe_config=SimpleNamespace(
            in_dtype=torch.bfloat16,
            activation=MoEActivation.SILU,
            experts_per_token=8,
            has_bias=False,
            is_lora_enabled=False,
        ),
    )
    with caplog.at_level("INFO", logger=mod.logger.name):
        runner = try_attach_w4a16_decode(experts, layer)
    assert isinstance(runner, Nvfp4W4A16DecodeMoE)
    assert runner.max_m == 64
    assert runner.clamp_limit == 10.0
    assert any(
        "DSV41_EXP_FUSED_W4A16_MOE=1: fused NVFP4 W4A16 decode MoE for M<=64"
        in rec.getMessage()
        and "swiglu_limit=10.0" in rec.getMessage()
        and "Marlin above" in rec.getMessage()
        for rec in caplog.records
    )

    monkeypatch.setattr(mod, "_ATTACH_LOGGED", False)
    experts.activation_config.clamp_limit = None
    caplog.clear()
    with caplog.at_level("WARNING", logger=mod.logger.name):
        assert try_attach_w4a16_decode(experts, layer) is None
    assert any(
        rec.getMessage()
        == "DSV41_EXP_FUSED_W4A16_MOE: disabled for this layer (swiglu_limit is unset)"
        for rec in caplog.records
    )


def test_can_run_small_m_only(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(ENV_FLAG, "1")
    runner = Nvfp4W4A16DecodeMoE(_make_pack(), topk=2, max_m=16, clamp_limit=10.0)
    x = torch.zeros(8, 128, dtype=torch.bfloat16)
    ids = torch.zeros(8, 2, dtype=torch.int32)
    assert runner.can_run(x, ids, MoEActivation.SILU, False, None)
    x_big = torch.zeros(32, 128, dtype=torch.bfloat16)
    ids_big = torch.zeros(32, 2, dtype=torch.int32)
    assert not runner.can_run(x_big, ids_big, MoEActivation.SILU, False, None)
    expert_map = torch.zeros(4, dtype=torch.int32)
    assert not runner.can_run(x, ids, MoEActivation.SILU, False, expert_map)


def test_occupancy_dsv41_flash_tp4():
    rows = occupancy_rows(
        token_counts=(4, 8, 16, 24, 64),
        hidden_size=DSV41_FLASH_TP4_H,
        intermediate_size=DSV41_FLASH_TP4_I,
        num_experts=256,
        topk=8,
        sm_count=110,
    )
    assert [r["M"] for r in rows] == [4, 8, 16, 24, 64]
    for row in rows:
        assert row["fc1_ctas"] > 0
        assert row["fc2_ctas"] > row["fc1_ctas"]
        assert row["fc1_waves"] > 0


def test_w4a16_dequant_multiplies_block_and_global_scale():
    # e2m1 1.0 (code 2) in the low nibble; high nibble unused/zero.
    packed = torch.tensor([[0x02] + [0] * 7], dtype=torch.uint8)  # K=16
    scale = torch.tensor([[2.0]], dtype=torch.float32).to(torch.float8_e4m3fn)
    gscale = 3.0
    vals = break_fp4_bytes(packed, torch.float32)
    sf = scale.view(torch.float8_e4m3fn).float()
    out = (vals.reshape(1, 1, 16) * (sf * gscale).unsqueeze(-1)).reshape(1, 16)
    assert out[0, 0].item() == pytest.approx(1.0 * 2.0 * 3.0, rel=1e-3)
    assert out[0, 1].item() == pytest.approx(0.0)


@pytest.mark.skipif(not torch.accelerator.is_available(), reason="CUDA required")
def test_w4a16_decode_kernel_gpu_smoke():
    from vllm.model_executor.layers.fused_moe.experts.nvfp4_emulation_moe import (
        invoke_fused_moe_nvfp4_emulation_kernel,
    )
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )
    from vllm.triton_utils import tl

    pack = _make_pack(num_experts=2, intermediate_size=64, hidden_size=128)
    device = torch.device("cuda")
    w13 = pack.w13.to(device)
    w13_scale = pack.w13_scale.to(device).view(torch.uint8)
    g1 = pack.w13_scale_2.to(device)
    x = torch.zeros(2, 128, dtype=torch.bfloat16, device=device)
    ids = torch.zeros(2, 2, dtype=torch.int32, device=device)
    out = torch.zeros(2, 2, 128, dtype=torch.bfloat16, device=device)
    cfg = dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64, GROUP_SIZE_M=1)
    sorted_ids, eids, npost = moe_align_block_size(ids, 16, 2, None)
    invoke_fused_moe_nvfp4_emulation_kernel(
        x,
        w13,
        out,
        w13_scale,
        g1,
        g1,
        None,
        sorted_ids,
        eids,
        npost,
        False,
        2,
        cfg,
        compute_type=tl.bfloat16,
    )
    torch.accelerator.synchronize()
    assert torch.isfinite(out.float()).all()
