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
    sm12x_align_prefill_q_len,
    sm12x_extend_prefill_slots,
    sm12x_pad_prefill_token_rows,
    sm12x_align_tokens,
    sm12x_disable_attn_aux_streams,
    sm12x_dspark_capture_sizes,
    sm12x_mixed_warmup_decode_prompt_len,
    sm12x_mixed_warmup_prefill_len,
    sm12x_pad_token_rows,
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


def test_sm12x_extend_prefill_slots_stays_in_block():
    slots = torch.tensor([10, 11], dtype=torch.int64)
    ext = sm12x_extend_prefill_slots(slots, 6, block_size=256)
    assert torch.equal(ext, torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int64))
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
    """Spark 18:13: tokens=4 decode dummy must not fill during capture."""
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
    """23:42: 6-token C4A write IMA'd after SWA-only [1, 6]."""
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
    """Spark 23:54: o_proj ran on 2 tokens after SWA-only [1, 6]."""
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

    q_len=5 first graph is 25=5×5 (16:26 IMA before FlashInfer pad).
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
    assert sm12x_dspark_capture_sizes(sizes, aligned) == [6, 24, 36]
    assert 18 not in sm12x_dspark_capture_sizes(sizes, aligned)
    assert 12 not in sm12x_dspark_capture_sizes(sizes, aligned)
    assert 25 not in sm12x_dspark_capture_sizes(sizes, aligned)


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
    sizes = [1, 2, 4, 8, 16, 24, 32, 36]
    assert sm12x_dspark_capture_sizes(sizes, 6) == sizes


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
    assert torch.count_nonzero(padded[8:]) == 0
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
    AssertionError'd at 0/5 (16:41). make_dummy zeros the flag."""
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
    """Spark 16:53: synchronize inside torch.cuda.graph aborted DSpark 0/3."""
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
