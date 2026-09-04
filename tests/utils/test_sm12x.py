# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.utils.sm12x import (
    SM12X_SAFE_MIN_TOKENS,
    extend_padding_mask,
    pad_token_rows,
    reject_sm12x_unsafe_decode_query,
    sm12x_align_decode_q_len,
    sm12x_align_flashinfer_dual_prefill,
    sm12x_align_prefill_q_len,
    sm12x_align_tokens,
    sm12x_allow_full_decode_capture,
    sm12x_disable_attn_aux_streams,
    sm12x_disable_eager_scratch_pool,
    sm12x_dspark_capture_sizes,
    sm12x_extend_prefill_slots,
    sm12x_flashinfer_autotune_query_lens,
    sm12x_flashinfer_decode_tune_sizes,
    sm12x_kernel_warmup_prefill_len,
    sm12x_mixed_warmup_decode_prompt_len,
    sm12x_mixed_warmup_prefill_len,
    sm12x_pad_prefill_token_rows,
    sm12x_pad_token_rows,
    sm12x_replace_moe_topk_sentinels,
    sm12x_replace_negative_indices,
    sm12x_replace_swa_index_sentinels,
    sm12x_should_fill_compressed_prefill_slots,
    sm12x_should_fill_prefill_slots,
    sm12x_skip_padded_prefill_c4a,
    sm12x_treat_short_extends_as_decodes,
)


def test_sm12x_align_tokens_pads_small_batches(monkeypatch):
    from vllm.utils import sm12x as sm12x_utils

    monkeypatch.setattr(
        sm12x_utils.current_platform,
        "is_device_capability_family",
        lambda fam: fam == 120,
    )
    assert sm12x_align_tokens(8) == SM12X_SAFE_MIN_TOKENS
    assert sm12x_align_tokens(1) == SM12X_SAFE_MIN_TOKENS
    assert sm12x_align_tokens(16) == 16
    assert sm12x_align_tokens(36) == 36
    assert sm12x_align_tokens(0) == 0


def test_sm12x_align_decode_q_len_snaps_to_safe_widths(monkeypatch):
    from vllm.utils import sm12x as sm12x_utils

    monkeypatch.setattr(
        sm12x_utils.current_platform,
        "is_device_capability_family",
        lambda fam: fam == 120,
    )
    assert sm12x_align_decode_q_len(2) == 6
    assert sm12x_align_decode_q_len(3) == 6
    assert sm12x_align_decode_q_len(4) == 4
    assert sm12x_align_decode_q_len(5) == 6
    assert sm12x_align_decode_q_len(15) == 16
    assert sm12x_align_decode_q_len(1) == 1
    assert sm12x_align_prefill_q_len(2) == 6
    assert sm12x_align_prefill_q_len(4) == 6
    assert sm12x_align_prefill_q_len(5) == 6
    assert sm12x_align_prefill_q_len(16) == 16
    reject_sm12x_unsafe_decode_query(torch.zeros(1, 6, 8, 512))
    reject_sm12x_unsafe_decode_query(torch.zeros(1, 4, 8, 512))
    reject_sm12x_unsafe_decode_query(torch.zeros(2, 4, 8, 512))
    with pytest.raises(RuntimeError, match="refused per-request query shape"):
        reject_sm12x_unsafe_decode_query(torch.zeros(1, 2, 8, 512))
    with pytest.raises(RuntimeError, match="refused per-request query shape"):
        reject_sm12x_unsafe_decode_query(torch.zeros(1, 3, 8, 512))
    from vllm.utils import flashinfer as fi_utils

    with pytest.raises(RuntimeError, match="refused per-request query shape"):
        fi_utils.flashinfer_trtllm_batch_decode_sparse_mla_dsv4(
            query=torch.zeros(1, 2, 8, 512)
        )
    assert sm12x_mixed_warmup_decode_prompt_len() == 2
    assert sm12x_mixed_warmup_prefill_len(15) == 2
    assert sm12x_mixed_warmup_prefill_len(2) == 2
    assert sm12x_treat_short_extends_as_decodes() is False
    assert sm12x_disable_attn_aux_streams() is True
    assert sm12x_disable_eager_scratch_pool() is True


def test_sm12x_extend_prefill_slots_reuses_last_slot():
    """Extra KV slots reuse the last real slot, matching repeat-last query."""
    slots = torch.tensor([10, 11], dtype=torch.int64)
    ext = sm12x_extend_prefill_slots(slots, 6, block_size=256)
    assert torch.equal(ext, torch.tensor([10, 11, 11, 11, 11, 11], dtype=torch.int64))
    edge = torch.tensor([254, 255], dtype=torch.int64)
    reused = sm12x_extend_prefill_slots(edge, 6, block_size=256)
    assert torch.equal(
        reused, torch.tensor([254, 255, 255, 255, 255, 255], dtype=torch.int64)
    )
    rows = torch.arange(2 * 3, dtype=torch.int32).view(2, 3)
    padded = sm12x_pad_prefill_token_rows(rows, 6)
    assert padded.shape == (6, 3)
    assert torch.equal(padded[:2], rows)
    assert torch.equal(padded[2:], rows[-1:].expand(4, -1))


def test_sm12x_fill_slots_skips_capture_and_decode_dummies(monkeypatch):
    """tokens=4 decode dummy must not fill SWA slots during capture."""
    from vllm.utils import sm12x as sm12x_utils

    monkeypatch.setattr(
        sm12x_utils.current_platform,
        "is_device_capability_family",
        lambda fam: fam == 120,
    )
    monkeypatch.setattr(sm12x_utils, "sm12x_is_capturing", lambda: False)
    assert sm12x_should_fill_prefill_slots(2, 0) is True
    assert sm12x_should_fill_prefill_slots(4, 0) is True
    assert sm12x_should_fill_prefill_slots(4, 4) is False
    assert sm12x_should_fill_prefill_slots(16, 0) is False
    monkeypatch.setattr(sm12x_utils, "sm12x_is_capturing", lambda: True)
    assert sm12x_should_fill_prefill_slots(2, 0) is False
    assert sm12x_should_fill_prefill_slots(4, 0) is False


def test_sm12x_does_not_fill_compressed_prefill_slots(monkeypatch):
    """6-token C4A write IMA'd after SWA-only [1, 6]."""
    from vllm.utils import sm12x as sm12x_utils

    monkeypatch.setattr(
        sm12x_utils.current_platform,
        "is_device_capability_family",
        lambda fam: fam == 120,
    )
    monkeypatch.setattr(sm12x_utils, "sm12x_is_capturing", lambda: False)
    assert sm12x_should_fill_compressed_prefill_slots(2, 0) is False
    assert sm12x_should_fill_compressed_prefill_slots(4, 0) is False
    assert sm12x_skip_padded_prefill_c4a([2]) is True
    assert sm12x_skip_padded_prefill_c4a([2, 4]) is True
    assert sm12x_skip_padded_prefill_c4a([16]) is False
    assert sm12x_skip_padded_prefill_c4a([2, 16]) is False
    monkeypatch.setattr(
        sm12x_utils.current_platform,
        "is_device_capability_family",
        lambda fam: False,
    )
    assert sm12x_skip_padded_prefill_c4a([2]) is False


def test_sm12x_align_flashinfer_dual_prefill_vision_width(monkeypatch):
    """Vision SWA is window+384 wide; dual-cache prefill cubin is topk=128."""
    from vllm.utils import sm12x as sm12x_utils

    monkeypatch.setattr(
        sm12x_utils.current_platform,
        "is_device_capability_family",
        lambda fam: fam == 120,
    )
    swa = torch.arange(2 * 512, dtype=torch.int32).view(2, 512)
    lens = torch.tensor([128, 80], dtype=torch.int32)
    extra = torch.zeros(1, 64, 1, 8)
    extra_i = torch.zeros(2, 512, dtype=torch.int32)
    extra_l = torch.tensor([512, 512], dtype=torch.int32)

    sliced, slens, e_kv, _, _ = sm12x_align_flashinfer_dual_prefill(
        swa, lens, extra, extra_i, extra_l, has_image=False
    )
    assert sliced.shape[-1] == 128
    assert torch.equal(sliced, swa[..., :128])
    assert int(slens.max()) <= 128
    assert e_kv is extra

    wide, wlens, no_kv, no_i, no_l = sm12x_align_flashinfer_dual_prefill(
        swa, lens, extra, extra_i, extra_l, has_image=True
    )
    assert wide.shape[-1] == 512
    assert no_kv is None
    assert no_i is None
    assert no_l is None
    assert wlens is lens

    already = torch.zeros(2, 128, dtype=torch.int32)
    out, _, keep, _, _ = sm12x_align_flashinfer_dual_prefill(
        already, lens, extra, extra_i, extra_l, has_image=False
    )
    assert out is already
    assert keep is extra

    monkeypatch.setattr(
        sm12x_utils.current_platform,
        "is_device_capability_family",
        lambda fam: False,
    )
    raw, _, keep_extra, _, _ = sm12x_align_flashinfer_dual_prefill(
        swa, lens, extra, extra_i, extra_l, has_image=False
    )
    assert raw is swa
    assert keep_extra is extra


def test_sm12x_replace_moe_topk_sentinels_zeros_pad_experts(monkeypatch):
    """VLLM_MOE_SKIP_PADDING writes -1; b12x SiLU gathers that id."""
    from vllm.utils import sm12x as sm12x_utils

    monkeypatch.setattr(
        sm12x_utils.current_platform,
        "is_device_capability_family",
        lambda fam: fam == 120,
    )
    ids = torch.tensor([[3, 5, -1], [-1, -1, -1]], dtype=torch.int32)
    weights = torch.tensor([[0.5, 0.5, 0.2], [0.1, 0.2, 0.3]], dtype=torch.float32)
    out_ids, out_w = sm12x_replace_moe_topk_sentinels(ids, weights)
    assert torch.equal(out_ids, torch.tensor([[3, 5, 0], [0, 0, 0]], dtype=torch.int32))
    assert torch.equal(
        out_w, torch.tensor([[0.5, 0.5, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32)
    )
    filled = sm12x_replace_negative_indices(ids, mode="fill0")
    assert filled is not None
    assert torch.equal(filled, out_ids)


def test_sm12x_replace_moe_topk_sentinels_noop_off_sm12x(monkeypatch):
    from vllm.utils import sm12x as sm12x_utils

    monkeypatch.setattr(
        sm12x_utils.current_platform,
        "is_device_capability_family",
        lambda fam: False,
    )
    ids = torch.tensor([[1, -1]], dtype=torch.int32)
    weights = torch.ones(1, 2)
    out_ids, out_w = sm12x_replace_moe_topk_sentinels(ids, weights)
    assert torch.equal(out_ids, ids)
    assert torch.equal(out_w, weights)


def test_sm12x_replace_negative_indices_modes(monkeypatch):
    """One helper: SWA repeats last valid; MoE fills expert 0."""
    from vllm.utils import sm12x as sm12x_utils

    monkeypatch.setattr(
        sm12x_utils.current_platform,
        "is_device_capability_family",
        lambda fam: fam == 120,
    )
    rows = torch.tensor([[3, 5, -1], [-1, -1, -1]], dtype=torch.int32)
    filled0 = sm12x_replace_negative_indices(rows, mode="fill0")
    assert filled0 is not None
    assert torch.equal(filled0, torch.tensor([[3, 5, 0], [0, 0, 0]], dtype=torch.int32))
    last = sm12x_replace_negative_indices(rows, mode="repeat_last")
    assert last is not None
    assert torch.equal(last, torch.tensor([[3, 5, 5], [0, 0, 0]], dtype=torch.int32))


def test_sm12x_moe_alignment_pad_zeros_extra_weights(monkeypatch):
    """Alignment pad repeats last topk row, then caller zeros extra weights."""
    from vllm.utils import sm12x as sm12x_utils

    monkeypatch.setattr(
        sm12x_utils.current_platform,
        "is_device_capability_family",
        lambda fam: fam == 120,
    )
    ids = torch.tensor([[3, 5], [1, 2]], dtype=torch.int32)
    weights = torch.tensor([[0.6, 0.4], [0.7, 0.3]], dtype=torch.float32)
    padded_ids, _ = sm12x_pad_token_rows(ids)
    padded_w, orig = sm12x_pad_token_rows(weights)
    padded_w[orig:] = 0
    assert orig == 2
    assert padded_ids.shape[0] == 16
    assert torch.equal(padded_ids[:2], ids)
    assert torch.equal(padded_ids[2:], ids[-1:].expand(14, -1))
    assert torch.equal(padded_w[:2], weights)
    assert torch.count_nonzero(padded_w[2:]) == 0


def test_sm12x_replace_swa_index_sentinels_repeats_last_valid(monkeypatch):
    """A 2-token first-prefill leaves 126–127 ``-1``s in each 128-wide row."""
    from vllm.utils import sm12x as sm12x_utils

    monkeypatch.setattr(
        sm12x_utils.current_platform,
        "is_device_capability_family",
        lambda fam: fam == 120,
    )
    rows = torch.full((2, 128), -1, dtype=torch.int32)
    rows[0, 0] = 7
    rows[1, 0] = 7
    rows[1, 1] = 8
    filled = sm12x_replace_swa_index_sentinels(rows)
    assert filled is not None
    assert filled.shape == (2, 128)
    assert torch.all(filled >= 0)
    assert torch.equal(filled[0], torch.full((128,), 7, dtype=torch.int32))
    expect1 = torch.full((128,), 8, dtype=torch.int32)
    expect1[0] = 7
    assert torch.equal(filled[1], expect1)

    batched = rows.unsqueeze(1)
    filled3 = sm12x_replace_swa_index_sentinels(batched)
    assert filled3 is not None
    assert filled3.shape == (2, 1, 128)
    assert torch.all(filled3 >= 0)
    assert torch.equal(filled3[:, 0], filled)

    empty_row = torch.full((1, 128), -1, dtype=torch.int32)
    zeros = sm12x_replace_swa_index_sentinels(empty_row)
    assert zeros is not None
    assert torch.equal(zeros, torch.zeros((1, 128), dtype=torch.int32))

    clean = torch.arange(8, dtype=torch.int32).view(2, 4)
    assert torch.equal(sm12x_replace_swa_index_sentinels(clean), clean)
    assert sm12x_replace_swa_index_sentinels(None) is None


def test_sm12x_replace_swa_index_sentinels_noop_off_sm12x(monkeypatch):
    from vllm.utils import sm12x as sm12x_utils

    monkeypatch.setattr(
        sm12x_utils.current_platform,
        "is_device_capability_family",
        lambda fam: False,
    )
    rows = torch.tensor([[10, -1, -1], [11, 12, -1]], dtype=torch.int32)
    assert torch.equal(sm12x_replace_swa_index_sentinels(rows), rows)


def test_sm12x_o_proj_pads_small_batches(monkeypatch):
    """o_proj ran on 2 tokens after SWA-only [1, 6]."""
    from vllm.models.deepseek_v4.nvidia.ops import o_proj as o_proj_mod
    from vllm.utils import sm12x as sm12x_utils

    monkeypatch.setattr(
        sm12x_utils.current_platform,
        "is_device_capability_family",
        lambda fam: fam == 120,
    )
    seen: dict[str, int] = {}

    def _fake_rope(o, positions, *args, **kwargs):
        seen["o"] = o.shape[0]
        seen["pos"] = positions.shape[0]
        return torch.zeros_like(o), torch.ones(o.shape[0], 1)

    class _W(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.zeros(1)
            self.weight_scale = torch.ones(1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            seen["wo"] = x.shape[0]
            return x

    monkeypatch.setattr(o_proj_mod, "fused_inv_rope_fp8_quant", _fake_rope)
    monkeypatch.setattr(
        o_proj_mod, "_use_deepseek_v4_sm12x_triton_fp8_einsum", lambda *a, **k: True
    )
    monkeypatch.setattr(o_proj_mod, "deepseek_v4_fp8_einsum", lambda *a, **k: None)
    wo = _W()
    out = o_proj_mod.deep_gemm_fp8_o_proj(
        torch.zeros(2, 4, 8),
        torch.arange(2),
        torch.zeros(1),
        wo,
        wo,
        n_groups=1,
        heads_per_group=4,
        nope_dim=4,
        rope_dim=4,
        o_lora_rank=8,
        einsum_recipe=(1, 128, 128),
        tma_aligned_scales=False,
    )
    assert seen == {"o": 16, "pos": 16, "wo": 16}
    assert out.shape[0] == 2


def _dspark_full_decode_capture_sizes(
    decode_query_len: int,
    capture_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 24, 32, 36),
    max_num_reqs: int = 6,
) -> list[int]:
    """Mirror CudaGraphManager FULL_DECODE_ONLY rounding (largest first)."""
    max_decode = max_num_reqs * decode_query_len
    max_cg = max(capture_sizes)
    sizes: list[int] = []
    for num_tokens in capture_sizes:
        rounded = (
            (num_tokens + decode_query_len - 1) // decode_query_len
        ) * decode_query_len
        reqs = rounded // decode_query_len
        if rounded > max_decode or rounded > max_cg or reqs > max_num_reqs:
            continue
        if rounded not in sizes:
            sizes.append(rounded)
    sizes.sort(reverse=True)
    return sizes


def test_sm12x_dspark_capture_avoids_q_len_5_dummy(monkeypatch):
    """DSpark sample_from_anchor is q_len=5; capture must use aligned 6.

    q_len=5 first graph is 25=5×5 (IMA before FlashInfer pad).
    q_len=6 first graph is 36=6×6 (main capture already green).
    """
    from vllm.utils import sm12x as sm12x_utils

    monkeypatch.setattr(
        sm12x_utils.current_platform,
        "is_device_capability_family",
        lambda fam: fam == 120,
    )
    assert sm12x_align_decode_q_len(5) == 6
    assert _dspark_full_decode_capture_sizes(5) == [25, 20, 10, 5]
    aligned = sm12x_align_decode_q_len(5)
    assert _dspark_full_decode_capture_sizes(aligned) == [36, 24, 18, 12, 6]
    assert 25 not in _dspark_full_decode_capture_sizes(aligned)
    sizes = [1, 2, 4, 8, 16, 24, 32, 36]
    assert sm12x_dspark_capture_sizes(sizes, aligned) == [6, 12, 18, 24, 36]
    assert 25 not in sm12x_dspark_capture_sizes(sizes, aligned)
    # Vision-Exp k=3 is DSpark q_len=3 (sample_from_anchor), not k+1=4.
    # 3 snaps to 6. Autotune must cover both runner q=4 and aligned q=6.
    vision_sizes = [1, 2, 4, 8, 12, 16, 24]
    assert sm12x_align_decode_q_len(3) == 6
    assert sm12x_align_decode_q_len(4) == 4
    assert sm12x_flashinfer_autotune_query_lens(4) == [4, 6]
    assert sm12x_dspark_capture_sizes(vision_sizes, 4) == [4, 8, 12, 16, 24]
    assert sm12x_dspark_capture_sizes(vision_sizes, 6) == [6, 12, 18, 24]
    assert sm12x_dspark_capture_sizes(sizes, 6) == [6, 12, 18, 24, 36]
    assert sm12x_allow_full_decode_capture(36, aligned) is True
    assert sm12x_allow_full_decode_capture(24, aligned) is True
    assert sm12x_allow_full_decode_capture(18, aligned) is True
    assert sm12x_allow_full_decode_capture(12, aligned) is True
    assert sm12x_allow_full_decode_capture(6, aligned) is True
    assert sm12x_flashinfer_decode_tune_sizes(sizes, aligned) == [
        6,
        12,
        18,
        24,
        36,
    ]
    assert sm12x_kernel_warmup_prefill_len(6) == 8
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
    from vllm.v1.worker.gpu.warmup import warmup_kernels

    assert "sm12x_allow_full_decode_capture" in (
        CudaGraphManager._init_candidates.__code__.co_names
    )
    assert "sm12x_kernel_warmup_prefill_len" in (
        warmup_kernels.__wrapped__.__code__.co_names
    )
    from vllm.model_executor.warmup import flashinfer_sparse_mla_warmup as fi_warmup

    assert "_autotune_sm12x_decode_sizes" in (
        fi_warmup._run_flashinfer_sparse_mla_decode_autotune.__code__.co_names
    )
    from contextlib import nullcontext

    seen: list[tuple[int, bool]] = []

    class _Runner:
        vllm_config = SimpleNamespace(
            compilation_config=SimpleNamespace(
                cudagraph_capture_sizes=[1, 2, 4, 8, 16, 24, 32, 36]
            )
        )
        decode_query_len = 6

        def _dummy_run(self, num_tokens: int, **kwargs: object) -> None:
            seen.append((num_tokens, bool(kwargs.get("uniform_decode"))))

    monkeypatch.setattr(fi_warmup, "flashinfer_autotune", lambda *a, **k: nullcontext())
    fi_warmup._autotune_sm12x_decode_sizes(_Runner(), "/tmp/x", is_leader=True)
    assert seen == [
        (6, True),
        (12, True),
        (18, True),
        (24, True),
        (36, True),
    ]

    seen.clear()

    class _VisionRunner:
        vllm_config = SimpleNamespace(
            compilation_config=SimpleNamespace(
                cudagraph_capture_sizes=[1, 2, 4, 8, 12, 16, 24]
            )
        )
        decode_query_len = 4

        def _dummy_run(self, num_tokens: int, **kwargs: object) -> None:
            seen.append((num_tokens, bool(kwargs.get("uniform_decode"))))

    fi_warmup._autotune_sm12x_decode_sizes(_VisionRunner(), "/tmp/x", is_leader=True)
    tokens = [n for n, _ in seen]
    assert 6 in tokens
    assert 4 in tokens


def test_dspark_init_cudagraph_manager_copies_capture_sizes(monkeypatch):
    """Vision k=3 capture override must copy configs; missing import copy crashed."""
    from vllm.config.compilation import CUDAGraphMode
    from vllm.utils import sm12x as sm12x_utils
    from vllm.v1.attention.backend import AttentionCGSupport
    from vllm.v1.worker.gpu.spec_decode.dflash import speculator as spec_mod

    monkeypatch.setattr(
        sm12x_utils.current_platform,
        "is_device_capability_family",
        lambda fam: fam == 120,
    )

    orig_sizes = [1, 2, 4, 8, 16, 24, 32, 36]
    compilation_config = SimpleNamespace(cudagraph_capture_sizes=list(orig_sizes))
    vllm_config = SimpleNamespace(compilation_config=compilation_config)
    captured: dict[str, object] = {}

    class _Mgr:
        def __init__(
            self,
            cfg: object,
            device: object,
            cudagraph_mode: object,
            decode_query_len: int,
        ) -> None:
            captured["sizes"] = list(cfg.compilation_config.cudagraph_capture_sizes)
            captured["q"] = decode_query_len
            captured["cfg_id"] = id(cfg)
            captured["comp_id"] = id(cfg.compilation_config)

    monkeypatch.setattr(spec_mod, "DFlashCudaGraphManager", _Mgr)

    spec = spec_mod.DFlashSpeculator.__new__(spec_mod.DFlashSpeculator)
    spec.vllm_config = vllm_config
    spec.device = "cpu"
    spec.num_query_per_req = 3
    spec._speculator_name = "DSpark"
    spec.attn_cg_support = SimpleNamespace(
        min_cg_support=AttentionCGSupport.UNIFORM_BATCH,
        min_cg_attn_backend="x",
    )
    spec.init_cudagraph_manager(CUDAGraphMode.FULL)

    assert captured["q"] == 6
    assert captured["sizes"] == [6, 12, 18, 24, 36]
    assert captured["cfg_id"] != id(vllm_config)
    assert captured["comp_id"] != id(compilation_config)
    assert compilation_config.cudagraph_capture_sizes == orig_sizes


def test_fused_qnorm_insert_out_is_optional():
    """later-main _C_stable_libtorch has no _out; do not attribute-access it."""
    from vllm.models.deepseek_v4.attention import DeepseekV4Attention

    names = DeepseekV4Attention._fused_qnorm_rope_kv_insert.__code__.co_names
    assert "getattr" in names
    assert "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert" in names


def test_sm12x_align_tokens_unchanged_off_sm12x(monkeypatch):
    from vllm.utils import sm12x as sm12x_utils

    monkeypatch.setattr(
        sm12x_utils.current_platform,
        "is_device_capability_family",
        lambda fam: False,
    )
    assert sm12x_align_tokens(8) == 8
    assert sm12x_mixed_warmup_decode_prompt_len() == 2
    assert sm12x_mixed_warmup_prefill_len(15) == 15
    assert sm12x_align_prefill_q_len(2) == 2
    assert sm12x_treat_short_extends_as_decodes() is True
    assert sm12x_disable_attn_aux_streams() is False
    assert sm12x_disable_eager_scratch_pool() is False
    sizes = [1, 2, 4, 8, 16, 24, 32, 36]
    assert sm12x_dspark_capture_sizes(sizes, 6) == sizes
    assert sm12x_allow_full_decode_capture(18, 6) is True
    assert sm12x_flashinfer_decode_tune_sizes(sizes, 6) == []
    assert sm12x_kernel_warmup_prefill_len(6) == 7


def test_sm12x_pad_token_rows(monkeypatch):
    from vllm.utils import sm12x as sm12x_utils

    monkeypatch.setattr(
        sm12x_utils.current_platform,
        "is_device_capability_family",
        lambda fam: fam == 120,
    )
    x = torch.arange(8 * 4, dtype=torch.float32).view(8, 4)
    padded, orig = sm12x_pad_token_rows(x, what="MoE")
    assert orig == 8
    assert padded.shape == (16, 4)
    assert torch.equal(padded[:8], x)
    assert torch.equal(padded[8:], x[-1:].expand(8, -1))
    assert pad_token_rows(x, 8).shape == (8, 4)


def test_extend_padding_mask_grows_with_true():
    """topk_softplus_sqrt requires is_padding.numel() == padded token count."""
    mask = torch.tensor([False, False, True, True, False, False, True, True])
    extended = extend_padding_mask(mask, 16)
    assert extended is not None
    assert extended.shape == (16,)
    assert torch.equal(extended[:8], mask)
    assert torch.all(extended[8:])
    assert extend_padding_mask(mask, 4).shape == (4,)
    assert extend_padding_mask(None, 16) is None


def test_sm12x_keeps_q_len_2_seed_on_prefill_split(monkeypatch):
    """DSpark threshold=6 must not swallow a 2-token first prefill as decode."""
    from vllm.utils import sm12x as sm12x_utils
    from vllm.v1.attention.backends.utils import split_decodes_and_prefills

    monkeypatch.setattr(
        sm12x_utils.current_platform,
        "is_device_capability_family",
        lambda fam: fam == 120,
    )
    cam = SimpleNamespace(
        max_query_len=2,
        num_reqs=1,
        num_actual_tokens=2,
        query_start_loc_cpu=torch.tensor([0, 2]),
        is_prefilling=torch.tensor([True]),
    )
    assert split_decodes_and_prefills(
        cam,
        decode_threshold=6,
        treat_short_extends_as_decodes=True,
    ) == (1, 0, 2, 0)
    assert split_decodes_and_prefills(
        cam,
        decode_threshold=6,
        treat_short_extends_as_decodes=sm12x_treat_short_extends_as_decodes(),
    ) == (0, 1, 0, 2)


def test_sm12x_dspark_capture_dummy_splits_as_all_decodes(monkeypatch):
    """DSpark capture dummies are q_len=6 decodes; missing is_prefilling
    AssertionError'd at 0/5. make_dummy zeros the flag."""
    from vllm.utils import sm12x as sm12x_utils
    from vllm.v1.attention.backends.utils import split_decodes_and_prefills

    monkeypatch.setattr(
        sm12x_utils.current_platform,
        "is_device_capability_family",
        lambda fam: fam == 120,
    )
    cam = SimpleNamespace(
        max_query_len=6,
        num_reqs=6,
        num_actual_tokens=36,
        query_start_loc_cpu=torch.arange(0, 37, 6, dtype=torch.int32),
        is_prefilling=torch.zeros(6, dtype=torch.bool),
    )
    assert split_decodes_and_prefills(
        cam,
        decode_threshold=6,
        treat_short_extends_as_decodes=sm12x_treat_short_extends_as_decodes(),
    ) == (6, 0, 36, 0)
    cam_missing = SimpleNamespace(**{**cam.__dict__, "is_prefilling": None})
    assert split_decodes_and_prefills(
        cam_missing,
        decode_threshold=6,
        treat_short_extends_as_decodes=False,
    ) == (6, 0, 36, 0)


def test_sm12x_full_cg_pad_aligns_unpadded_is_prefilling(monkeypatch):
    """FULL-CG pads 2 DSpark decodes to 4 slots; is_prefilling stays [2].

    Live query_start_loc is ``[0, 6, 12, 12, 12]`` (extra slots q_len=0).
    ``is_prefill |= prefilling`` then crashed:
    ``The size of tensor a (4) must match the size of tensor b (2)``.
    """
    from vllm.utils import sm12x as sm12x_utils
    from vllm.v1.attention.backends.utils import split_decodes_and_prefills

    monkeypatch.setattr(
        sm12x_utils.current_platform,
        "is_device_capability_family",
        lambda fam: fam == 120,
    )
    cam = SimpleNamespace(
        max_query_len=6,
        num_reqs=4,
        num_actual_tokens=24,
        query_start_loc_cpu=torch.tensor([0, 6, 12, 12, 12], dtype=torch.int32),
        is_prefilling=torch.tensor([False, False]),
    )
    assert split_decodes_and_prefills(
        cam,
        decode_threshold=6,
        treat_short_extends_as_decodes=sm12x_treat_short_extends_as_decodes(),
    ) == (4, 0, 24, 0)
    cam_mixed = SimpleNamespace(
        **{**cam.__dict__, "is_prefilling": torch.tensor([False, True])}
    )
    assert split_decodes_and_prefills(
        cam_mixed,
        decode_threshold=6,
        treat_short_extends_as_decodes=False,
    ) == (1, 3, 6, 18)


def test_sm12x_c128a_split_matches_swa_for_mixed_warmup(monkeypatch):
    """C128A must use the same treat_short flag as SWA.

    Default treat_short=True plus DSpark threshold (6 or 11) classifies
    the 2-token seed and the mixed 1+2 step as all-decode. FlashInfer
    follows SWA (prefill) and asserts c128a_prefill_topk_indices.
    """
    from vllm.models.deepseek_v4.sparse_mla import (
        DeepseekV4SparseMLAMetadataBuilder,
    )
    from vllm.utils import sm12x as sm12x_utils
    from vllm.v1.attention.backends.utils import split_decodes_and_prefills

    monkeypatch.setattr(
        sm12x_utils.current_platform,
        "is_device_capability_family",
        lambda fam: fam == 120,
    )

    def _split(cam, threshold=6):
        # Same kwargs as DeepseekV4SparseMLAMetadataBuilder._build_c128a_metadata.
        return split_decodes_and_prefills(
            cam,
            decode_threshold=threshold,
            treat_short_extends_as_decodes=sm12x_treat_short_extends_as_decodes(),
        )

    seed = SimpleNamespace(
        max_query_len=2,
        num_reqs=1,
        num_actual_tokens=2,
        query_start_loc_cpu=torch.tensor([0, 2]),
        is_prefilling=torch.tensor([True]),
    )
    mixed = SimpleNamespace(
        max_query_len=2,
        num_reqs=2,
        num_actual_tokens=3,
        query_start_loc_cpu=torch.tensor([0, 1, 3]),
        is_prefilling=torch.tensor([False, True]),
    )
    assert _split(seed) == (0, 1, 0, 2)
    assert _split(mixed) == (1, 1, 1, 2)
    assert _split(seed, threshold=11) == (0, 1, 0, 2)
    assert split_decodes_and_prefills(seed, decode_threshold=6) == (1, 0, 2, 0)
    assert split_decodes_and_prefills(mixed, decode_threshold=6) == (2, 0, 3, 0)
    names = DeepseekV4SparseMLAMetadataBuilder._build_c128a_metadata.__code__.co_names
    assert "sm12x_treat_short_extends_as_decodes" in names


def test_sm12x_dspark_dummy_syncs_only_on_eager_warmup(monkeypatch):
    """synchronize inside torch.cuda.graph aborted DSpark 0/3."""
    from vllm.v1.worker.gpu.spec_decode.dflash import cudagraph as dflash_cg

    synced: list[bool] = []
    monkeypatch.setattr(
        dflash_cg.current_platform,
        "is_device_capability_family",
        lambda fam: fam == 120,
    )
    monkeypatch.setattr(
        dflash_cg.torch.cuda, "synchronize", lambda: synced.append(True)
    )
    dflash_cg._sync_after_eager_dspark_dummy(False)
    assert synced == []
    dflash_cg._sync_after_eager_dspark_dummy(True)
    assert synced == [True]
