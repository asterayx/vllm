# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CED prefill gating for DeepSeek-V4.1. CPU only: no kernel launch."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v41.ced_prefill import (
    CED_PREFILL_ENV,
    CED_PREFILL_ROWS_ENV,
    CedPrefillConfig,
    CedRefusal,
    apply_ced_compaction,
    build_ced_prefill,
    ced_prefill_enabled,
    ced_runtime_block_reason,
    decide_ced_rank,
    full_resolution_boundary,
    plan_prefill_keep,
    reset_ced_prefill_logs,
    restore_token_rows,
)

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _clear_ced_logs():
    reset_ced_prefill_logs()
    yield
    reset_ced_prefill_logs()


def _flash_hf(**overrides):
    ratios = [0, 0] + [2] * 18 + [1] * 20
    config = dict(
        num_hidden_layers=40,
        compress_ratios=ratios,
        kv_source_layer_ids=[2, 8, 14, 20],
        index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
        engram_layer_ids=[1, 14],
        sliding_window=128,
        candidate_source_layer_id=20,
        candidate_topk_blocks=2048,
        candidate_block_size=8,
    )
    config.update(overrides)
    return SimpleNamespace(**config)


def _vllm_config(hf):
    parallel = SimpleNamespace(
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf),
        parallel_config=parallel,
    )


def _rank_kwargs(hf, **overrides):
    base = dict(
        boundary=full_resolution_boundary(hf),
        num_hidden_layers=hf.num_hidden_layers,
        start_layer=0,
        end_layer=hf.num_hidden_layers,
        tail_rows=128,
        use_sequence_parallel=False,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        use_mega_moe=False,
        engram_layer_ids=tuple(hf.engram_layer_ids),
    )
    base.update(overrides)
    return base


def test_flag_off_is_inactive(monkeypatch):
    monkeypatch.delenv(CED_PREFILL_ENV, raising=False)
    monkeypatch.delenv(CED_PREFILL_ROWS_ENV, raising=False)
    assert ced_prefill_enabled() is False
    # Flag off does not inspect topology, so a refuse-worthy config is ignored.
    assert (
        build_ced_prefill(
            None,
            start_layer=0,
            end_layer=40,
            use_sequence_parallel=True,
            use_mega_moe=True,
        )
        is None
    )


def test_flag_on_attaches_at_flash_boundary(monkeypatch):
    monkeypatch.setenv(CED_PREFILL_ENV, "1")
    hf = _flash_hf()
    assert full_resolution_boundary(hf) == 20
    # 2048 blocks x 8 positions is the existing indexer pool. CED does not
    # resize it; this pins the number the PR reports.
    assert hf.candidate_topk_blocks * hf.candidate_block_size == 16384

    built = build_ced_prefill(
        _vllm_config(hf),
        start_layer=0,
        end_layer=40,
        use_sequence_parallel=False,
        use_mega_moe=False,
    )
    assert built is not None
    assert built.boundary_layer == 20
    assert built.tail_rows == 128
    assert built.reindex_after_boundary is True


def test_tail_rows_knob_defaults_and_overrides(monkeypatch):
    monkeypatch.setenv(CED_PREFILL_ENV, "1")
    hf = _flash_hf()
    monkeypatch.setenv(CED_PREFILL_ROWS_ENV, "64")
    built = build_ced_prefill(
        _vllm_config(hf),
        start_layer=0,
        end_layer=40,
        use_sequence_parallel=False,
        use_mega_moe=False,
    )
    assert built is not None and built.tail_rows == 64

    monkeypatch.setenv(CED_PREFILL_ROWS_ENV, "nope")
    reset_ced_prefill_logs()
    assert (
        build_ced_prefill(
            _vllm_config(hf),
            start_layer=0,
            end_layer=40,
            use_sequence_parallel=False,
            use_mega_moe=False,
        )
        is None
    )


def test_pp2_attaches_only_on_the_decoder_rank():
    hf = _flash_hf()
    # Even split of 40 layers: rank 0 is the encoder, rank 1 owns layer 20
    # and every later decoder layer.
    encoder = decide_ced_rank(**_rank_kwargs(hf, start_layer=0, end_layer=20))
    decoder = decide_ced_rank(**_rank_kwargs(hf, start_layer=20, end_layer=40))
    assert encoder == ("skip", None)
    assert decoder[0] == "attach"


def test_refusal_reasons():
    hf = _flash_hf()
    split = decide_ced_rank(**_rank_kwargs(hf, start_layer=20, end_layer=30))
    assert split[0] == "refuse"
    assert "PP layers" in split[1]

    later_only = decide_ced_rank(**_rank_kwargs(hf, start_layer=30, end_layer=40))
    assert later_only[0] == "refuse"
    assert "do not own boundary" in later_only[1]

    sp = decide_ced_rank(**_rank_kwargs(hf, use_sequence_parallel=True))
    assert sp[0] == "refuse" and "sequence parallel" in sp[1]

    cp = decide_ced_rank(**_rank_kwargs(hf, prefill_context_parallel_size=2))
    assert cp[0] == "refuse" and "context parallel" in cp[1]

    mega = decide_ced_rank(**_rank_kwargs(hf, use_mega_moe=True))
    assert mega[0] == "refuse" and "MegaMoE" in mega[1]

    engram = decide_ced_rank(**_rank_kwargs(hf, engram_layer_ids=(1, 30)))
    assert engram[0] == "refuse" and "engram" in engram[1]

    no_boundary = decide_ced_rank(**_rank_kwargs(hf, boundary=None))
    assert no_boundary[0] == "refuse" and "no full-resolution" in no_boundary[1]

    last = _flash_hf(num_hidden_layers=21, compress_ratios=[0, 0] + [2] * 18 + [1])
    # boundary 20 is the last layer: nothing after it to compact.
    assert full_resolution_boundary(last) == 20
    refused = decide_ced_rank(**_rank_kwargs(last, end_layer=21, num_hidden_layers=21))
    assert refused[0] == "refuse" and "last layer" in refused[1]


def test_non_index_ratio1_is_not_the_boundary():
    hf = _flash_hf(index_source_layer_ids=[2, 8, 14])
    assert full_resolution_boundary(hf) is None


def test_keep_plan_last_128_after_decode_prefix():
    # 4 decode tokens (DSpark verify) then one 200-row prefill chunk.
    qsl = [0, 4, 204]
    plan = plan_prefill_keep(qsl, num_decodes=1, num_decode_tokens=4, tail_rows=128)
    assert not isinstance(plan, str)
    assert plan.dropped_prefill_rows == 200 - 128
    assert plan.num_prefill_tokens == 128
    assert plan.keep_indices[:4] == (0, 1, 2, 3)
    assert plan.keep_indices[4:] == tuple(range(204 - 128, 204))
    assert plan.query_lens == (4, 128)


def test_short_prefill_and_decode_drop_nothing():
    short = plan_prefill_keep(
        [0, 64], num_decodes=0, num_decode_tokens=0, tail_rows=128
    )
    assert not isinstance(short, str)
    assert short.dropped_prefill_rows == 0
    assert short.keep_indices == tuple(range(64))

    decode = plan_prefill_keep(
        [0, 4, 8], num_decodes=2, num_decode_tokens=8, tail_rows=128
    )
    assert not isinstance(decode, str)
    assert decode.dropped_prefill_rows == 0


def test_each_prefill_keeps_its_own_tail():
    # decode [0, 2), prefill A [2, 10), prefill B [10, 14). Tail 3.
    plan = plan_prefill_keep(
        [0, 2, 10, 14], num_decodes=1, num_decode_tokens=2, tail_rows=3
    )
    assert not isinstance(plan, str)
    assert plan.keep_indices == (0, 1, 7, 8, 9, 11, 12, 13)
    assert plan.query_lens == (2, 3, 3)
    assert plan.prefill_local == (5, 6, 7, 9, 10, 11)


@dataclass
class _Chunk:
    cu_seqlen_ks: torch.Tensor
    cu_seqlen_ke: torch.Tensor
    token_to_seq: torch.Tensor
    token_start: int
    token_end: int
    pcp_deinterleave_idx: torch.Tensor | None = None


@dataclass
class _Prefill:
    chunks: list


@dataclass
class _Indexer:
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int
    slot_mapping: torch.Tensor
    prefill: _Prefill
    seq_lens: torch.Tensor


def _batch(tail_rows: int = 3):
    """2 decode tokens + 8 prefill rows. Boundary is layer 20."""
    full = 10
    qsl = torch.tensor([0, 2, full], dtype=torch.int32)
    swa = SimpleNamespace(
        num_decodes=1,
        num_prefills=1,
        num_decode_tokens=2,
        num_prefill_tokens=8,
        query_start_loc_cpu=qsl.clone(),
        query_start_loc=qsl.clone(),
        slot_mapping=torch.arange(full, dtype=torch.int64),
        is_valid_token=torch.ones(full, dtype=torch.bool),
        token_to_req_indices=torch.tensor([0, 0, 1, 1, 1, 1, 1, 1, 1, 1]),
        prefill_swa_indices=torch.arange(8).view(8, 1),
        prefill_swa_lens=torch.arange(8),
        prefill_left_visible=None,
        prefill_right_visible=None,
        prefill_query_lens_cpu=torch.tensor([8], dtype=torch.int32),
        prefill_seq_lens_cpu=torch.tensor([8], dtype=torch.int32),
        prefill_gather_lens=torch.tensor([8], dtype=torch.int32),
        prefill_window_size=128,
        prefill_seq_lens_cpu_id=None,
        flashinfer_sparse_index_cache={"stale": (1, 2)},
    )
    # Mark the seq-lens tensor so the test can see it was not replaced.
    swa.prefill_seq_lens_cpu_id = swa.prefill_seq_lens_cpu
    ks = torch.arange(8, dtype=torch.int32)
    indexer = _Indexer(
        num_decodes=1,
        num_decode_tokens=2,
        num_prefills=1,
        num_prefill_tokens=8,
        slot_mapping=torch.arange(full, dtype=torch.int64),
        prefill=_Prefill(
            chunks=[
                _Chunk(
                    cu_seqlen_ks=ks.clone(),
                    cu_seqlen_ke=ks.clone() + 8,
                    token_to_seq=torch.zeros(8, dtype=torch.int32),
                    token_start=2,
                    token_end=10,
                )
            ]
        ),
        seq_lens=torch.tensor([2, 8], dtype=torch.int32),
    )
    hidden = torch.arange(full, dtype=torch.float32).view(full, 1)
    topk = torch.arange(full, dtype=torch.int32).view(full, 1)
    candidates = torch.arange(full, dtype=torch.int32).view(full, 1).repeat(1, 4)
    metadata = {
        "model.layers.20.attn.swa_cache": swa,  # shared object, also below
        "model.layers.21.attn.swa_cache": swa,
        "model.layers.20.attn.indexer.k_cache": indexer,
        "model.layers.24.attn.swa_cache": swa,
    }
    config = CedPrefillConfig(
        boundary_layer=20, tail_rows=tail_rows, reindex_after_boundary=True
    )
    return config, metadata, swa, indexer, hidden, topk, candidates


def test_flag_on_compacts_to_last_n_and_scatter_restores_tail():
    config, metadata, swa, indexer, hidden, topk, candidates = _batch(tail_rows=3)
    seq_lens = swa.prefill_seq_lens_cpu
    indexer_seq = indexer.seq_lens
    original_ks = indexer.prefill.chunks[0].cu_seqlen_ks.clone()
    result = apply_ced_compaction(
        config,
        metadata,
        hidden_states=hidden,
        residual=hidden.clone(),
        post_mix=None,
        res_mix=None,
        pre_mix=hidden.clone(),
        positions=torch.arange(10),
        input_ids=torch.arange(10),
        topk_indices_buffer=topk,
        candidate_block_buffer=candidates,
    )
    assert not isinstance(result, CedRefusal)
    assert result is not None
    # Decode rows 0,1 plus prefill abs 7,8,9 (local rows 5,6,7).
    assert result.keep.compact_tokens == 5
    assert result.keep.full_prefill_tokens == 8
    assert result.keep.kept_prefill_tokens == 3
    expected = torch.tensor([0, 1, 7, 8, 9], dtype=torch.float32)
    torch.testing.assert_close(result.hidden_states.view(-1), expected)
    assert swa.num_prefill_tokens == 3
    assert tuple(int(v) for v in swa.query_start_loc_cpu.tolist()) == (0, 2, 5)
    torch.testing.assert_close(
        swa.prefill_swa_indices.view(-1), torch.tensor([5, 6, 7])
    )
    # Full context length stays; gather grows by the dropped prefix inside the window.
    assert swa.prefill_seq_lens_cpu is seq_lens
    assert int(swa.prefill_gather_lens[0]) == 3 + min(8 - 3, 127)
    assert swa.flashinfer_sparse_index_cache == {}
    # Shared SWA object narrowed once, not twice (still 3, not the tail of 3).
    assert metadata["model.layers.20.attn.swa_cache"].num_prefill_tokens == 3

    # Indexer context bounds of the kept rows are preserved; the span retargets.
    chunk = indexer.prefill.chunks[0]
    assert chunk.token_start == 2 and chunk.token_end == 5
    torch.testing.assert_close(chunk.cu_seqlen_ks, original_ks[5:])
    torch.testing.assert_close(chunk.cu_seqlen_ke, (original_ks + 8)[5:])
    assert indexer.seq_lens is indexer_seq
    assert indexer.num_prefill_tokens == 3
    torch.testing.assert_close(
        topk[:5], torch.tensor([[0], [1], [7], [8], [9]], dtype=torch.int32)
    )
    # Pool width is unchanged; only the row set moved.
    assert candidates.shape[1] == 4
    torch.testing.assert_close(
        candidates[:5, 0], torch.tensor([0, 1, 7, 8, 9], dtype=torch.int32)
    )

    restored = restore_token_rows(result.hidden_states, result.keep)
    assert restored.shape[0] == 10
    torch.testing.assert_close(
        restored.view(-1),
        torch.tensor([0, 1, 0, 0, 0, 0, 0, 7, 8, 9], dtype=torch.float32),
    )


def test_short_chunk_does_not_compact():
    config, metadata, swa, _, hidden, topk, candidates = _batch(tail_rows=128)
    before = swa.slot_mapping.clone()
    result = apply_ced_compaction(
        config,
        metadata,
        hidden_states=hidden,
        residual=None,
        post_mix=None,
        res_mix=None,
        pre_mix=None,
        positions=torch.arange(10),
        input_ids=None,
        topk_indices_buffer=topk,
        candidate_block_buffer=candidates,
    )
    assert result is None
    torch.testing.assert_close(swa.slot_mapping, before)
    assert swa.num_prefill_tokens == 8


def test_unrecognized_metadata_refuses_without_mutating():
    config, metadata, swa, _, hidden, topk, _candidates = _batch(tail_rows=3)
    metadata["model.layers.22.attn.extra"] = object()
    before_slots = swa.slot_mapping.clone()
    before_topk = topk.clone()
    result = apply_ced_compaction(
        config,
        metadata,
        hidden_states=hidden,
        residual=None,
        post_mix=None,
        res_mix=None,
        pre_mix=None,
        positions=torch.arange(10),
        input_ids=None,
        topk_indices_buffer=topk,
        candidate_block_buffer=None,
    )
    assert isinstance(result, CedRefusal)
    assert "unrecognized" in result.reason
    torch.testing.assert_close(swa.slot_mapping, before_slots)
    torch.testing.assert_close(topk, before_topk)
    assert swa.num_prefill_tokens == 8


def test_missing_indexer_metadata_refuses():
    config, metadata, swa, _, hidden, topk, _candidates = _batch(tail_rows=3)
    del metadata["model.layers.20.attn.indexer.k_cache"]
    before = swa.num_prefill_tokens
    result = apply_ced_compaction(
        config,
        metadata,
        hidden_states=hidden,
        residual=None,
        post_mix=None,
        res_mix=None,
        pre_mix=None,
        positions=torch.arange(10),
        input_ids=None,
        topk_indices_buffer=topk,
        candidate_block_buffer=None,
    )
    assert isinstance(result, CedRefusal)
    assert "indexer" in result.reason
    assert swa.num_prefill_tokens == before


def test_runtime_block_reasons():
    assert (
        ced_runtime_block_reason(
            metadata_kind="dict", capturing=False, cudagraph_mode_value=0
        )
        is None
    )
    capture = ced_runtime_block_reason(
        metadata_kind="dict", capturing=True, cudagraph_mode_value=0
    )
    assert capture is not None and "capture" in capture
    graph = ced_runtime_block_reason(
        metadata_kind="dict", capturing=False, cudagraph_mode_value=1
    )
    assert graph is not None and "cudagraph" in graph
    ubatch = ced_runtime_block_reason(
        metadata_kind="list", capturing=False, cudagraph_mode_value=0
    )
    assert ubatch is not None and "ubatch" in ubatch
    missing = ced_runtime_block_reason(
        metadata_kind="missing", capturing=False, cudagraph_mode_value=0
    )
    assert missing is not None and "missing" in missing
