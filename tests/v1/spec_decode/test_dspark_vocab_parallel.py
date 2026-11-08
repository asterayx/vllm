# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vocab-parallel DSpark sample/verify reductions.

PP/grammar: this path does not change draft-token bookkeeping. Structured
output still uses gathered logits so grammar bitmasks apply on the stock
verify path (open PR #12 persists per-request drafts under V2+PP).
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.sample.vocab_parallel import (
    apply_row_temperature,
    combine_logsumexp,
    gather_token_logits,
    mask_rejected_token,
    proposal_tokens_from_draft_sampled,
    reduce_max_idx,
    shard_logsumexp,
    vocab_parallel_sample_enabled,
    vocab_shard_argmax,
)


def _cat_last(parts: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat(parts, dim=-1)


def _fake_all_gather(parts: list[torch.Tensor]):
    """Last-dim all-gather stand-in: ignore the caller tensor, concat shards."""

    def fn(local: torch.Tensor) -> torch.Tensor:
        del local
        return _cat_last(parts)

    return fn


def test_flag_off_by_default():
    assert vocab_parallel_sample_enabled() is False


def test_flag_opt_in(monkeypatch):
    monkeypatch.setenv("DSV41_EXP_VOCAB_PARALLEL_SAMPLE", "1")
    assert vocab_parallel_sample_enabled() is True
    monkeypatch.setenv("DSV41_EXP_VOCAB_PARALLEL_SAMPLE", "0")
    assert vocab_parallel_sample_enabled() is False


def test_reduce_max_idx_tp1_is_identity():
    vals = torch.tensor([1.5, -0.2, 3.0])
    idx = torch.tensor([7, 1, 99])
    out = reduce_max_idx(vals, idx, tp_size=1)
    assert out.tolist() == [7, 1, 99]


def test_greedy_vocab_parallel_reduction_matches_concat_argmax():
    """Greedy draft/verify: shard argmax + (max, idx) reduce == full argmax."""
    torch.manual_seed(0)
    rows, shard, tp = 5, 8, 2
    shards = [torch.randn(rows, shard) for _ in range(tp)]
    full = torch.cat(shards, dim=-1)
    expected = full.argmax(dim=-1)

    pairs = []
    for rank, shard_logits in enumerate(shards):
        vmax, vidx = shard_logits.max(dim=-1)
        pairs.append(torch.stack([vmax.float(), (vidx + rank * shard).float()], dim=-1))

    out = reduce_max_idx(
        shards[0].max(dim=-1).values,
        shards[0].max(dim=-1).indices,
        tp_size=tp,
        all_gather_fn=_fake_all_gather(pairs),
    )
    assert out.tolist() == expected.tolist()


def test_sampled_vocab_parallel_reduction_matches_concat_argmax():
    """Temperature>0 path: Gumbel-max is argmax of noised shard logits."""
    torch.manual_seed(1)
    rows, shard, tp = 4, 6, 2
    shards = [torch.randn(rows, shard) for _ in range(tp)]
    noise = [torch.randn(rows, shard) for _ in range(tp)]
    noised = [s + n for s, n in zip(shards, noise)]
    full_noised = torch.cat(noised, dim=-1)
    expected = full_noised.argmax(dim=-1)

    pairs = []
    for rank, shard_noised in enumerate(noised):
        vmax, vidx = shard_noised.max(dim=-1)
        pairs.append(torch.stack([vmax.float(), (vidx + rank * shard).float()], dim=-1))

    out = reduce_max_idx(
        noised[0].max(dim=-1).values,
        noised[0].max(dim=-1).indices,
        tp_size=tp,
        all_gather_fn=_fake_all_gather(pairs),
    )
    assert out.tolist() == expected.tolist()


def test_logsumexp_reduction_matches_concat():
    torch.manual_seed(2)
    rows, shard, tp = 6, 7, 4
    shards = [torch.randn(rows, shard) for _ in range(tp)]
    full = torch.cat(shards, dim=-1)
    expected = torch.logsumexp(full.float(), dim=-1)

    maxes = []
    sumexps = []
    for shard_logits in shards:
        mx = shard_logits.float().max(dim=-1).values
        se = torch.exp(shard_logits.float() - mx.unsqueeze(-1)).sum(dim=-1)
        maxes.append(mx)
        sumexps.append(se)
    gathered_max = torch.stack(maxes, dim=-1)
    gathered_se = torch.stack(sumexps, dim=-1)
    got = combine_logsumexp(gathered_max, gathered_se)
    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)


def test_shard_logsumexp_tp1_matches_torch():
    logits = torch.tensor([[1.0, 0.0, -5.0], [0.0, 0.0, 0.0]])
    got = shard_logsumexp(logits, tp_size=1)
    torch.testing.assert_close(got, torch.logsumexp(logits, dim=-1))


def test_gather_token_logits_owned_and_missing():
    local = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    # Rank 0 owns [0, 3). Token 1 is local; token 5 is not; -1 is invalid.
    tokens = torch.tensor([1, 5])
    got = gather_token_logits(local, tokens, vocab_start=0, n_org=3, tp_size=1)
    assert got[0].item() == 2.0
    assert torch.isneginf(got[1])

    tokens_invalid = torch.tensor([-1, 0])
    got = gather_token_logits(local, tokens_invalid, vocab_start=0, n_org=3, tp_size=1)
    assert torch.isneginf(got[0])
    assert got[1].item() == 4.0


def test_gather_token_logits_tp_reduce_takes_owner():
    rank0 = torch.tensor([[10.0, 11.0]])
    rank1 = torch.tensor([[20.0, 21.0]])
    # token 3 lives on rank 1 (start=2). Fake all-gather of the 1-wide values.
    token = torch.tensor([3])
    v0 = gather_token_logits(rank0, token, 0, 2, tp_size=1)
    v1 = gather_token_logits(rank1, token, 2, 2, tp_size=1)
    assert torch.isneginf(v0)
    assert v1.item() == 21.0
    gathered = torch.stack([v0, v1], dim=-1)
    assert gathered.max(dim=-1).values.item() == 21.0


def test_one_hot_leviathan_accepts_until_ratio_fails():
    """Sampled verify: accept while log p(draft) > log u, then stop."""
    log_p = torch.tensor([-0.1, -0.2, -4.0])
    u = torch.tensor([0.2, 0.3, 0.5])
    accepted = 0
    for lp, uu in zip(log_p.tolist(), u.tolist()):
        if lp <= torch.tensor(uu).log().item():
            break
        accepted += 1
    assert accepted == 2


def test_greedy_one_hot_accepts_while_draft_equals_argmax():
    drafts = torch.tensor([4, 4, 7])
    top = torch.tensor([4, 4, 1])
    accepted = 0
    for d, t in zip(drafts.tolist(), top.tolist()):
        if d != t:
            break
        accepted += 1
    assert accepted == 2


def test_residual_mask_zeros_rejected_token_then_argmax():
    logits = torch.tensor([[0.1, 5.0, 0.2, 0.3]])
    masked = mask_rejected_token(logits, torch.tensor([1]), vocab_start=0, n_org=4)
    assert torch.isneginf(masked[0, 1])
    assert masked.argmax(dim=-1).item() == 3


def test_proposal_tokens_skip_bonus_rows():
    # 2 requests, K=2 -> 3 logit rows each.
    draft_sampled = torch.tensor([9, 1, 2, 8, 3, 4])
    cu = torch.tensor([0, 3, 6])
    got = proposal_tokens_from_draft_sampled(draft_sampled, cu)
    assert got.tolist() == [1, 2, -1, 3, 4, -1]


def test_apply_row_temperature_skips_zero_and_one():
    logits = torch.tensor([[2.0, 4.0], [2.0, 4.0], [2.0, 4.0]])
    idx = torch.tensor([0, 1, 2])
    temp = torch.tensor([0.0, 1.0, 2.0])
    scaled = apply_row_temperature(logits, idx, temp)
    torch.testing.assert_close(scaled[0], logits[0].float())
    torch.testing.assert_close(scaled[1], logits[1].float())
    torch.testing.assert_close(scaled[2], logits[2].float() / 2.0)


def test_vocab_shard_argmax_tp1():
    logits = torch.tensor([[0.0, 3.0, 1.0], [5.0, 0.0, 0.1]])
    got = vocab_shard_argmax(logits, vocab_start=10, tp_size=1)
    assert got.tolist() == [11, 10]


def _make_sampler_for_eligibility(
    *,
    temperature: float,
    top_p: float = 1.0,
    top_k: int | None = None,
    min_p: float = 0.0,
    use_penalty: bool = False,
    vocab_size: int = 128,
):
    if top_k is None:
        top_k = vocab_size
    states = SimpleNamespace(
        temperature=SimpleNamespace(np=np.array([temperature, temperature])),
        top_p=SimpleNamespace(np=np.array([top_p, top_p])),
        top_k=SimpleNamespace(np=np.array([top_k, top_k])),
        min_p=SimpleNamespace(np=np.array([min_p, min_p])),
        vocab_size=vocab_size,
        max_num_logprobs=lambda idx: -1,
    )
    sampler = SimpleNamespace(
        sampling_states=states,
        penalties_state=SimpleNamespace(use_penalty=np.array([use_penalty, False])),
        thinking_budget_state=SimpleNamespace(
            enabled=False, use_thinking_budget=np.array([False, False])
        ),
        logits_processors=[],
        compute_nans=False,
        trace_replay_state=None,
        use_fp64_gumbel=False,
    )
    return sampler


def test_can_vocab_parallel_verify_temp_gt0_without_top_p(monkeypatch):
    monkeypatch.setenv("DSV41_EXP_VOCAB_PARALLEL_SAMPLE", "1")
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

    rs = RejectionSampler.__new__(RejectionSampler)
    rs.watermark_key = None
    rs.synthetic_conditional_rates = None
    rs.use_block_verification = False
    rs.enable_adaptive_verification = False
    rs.sampler = _make_sampler_for_eligibility(temperature=0.7)
    batch = SimpleNamespace(idx_mapping_np=np.array([0]))
    assert rs.can_vocab_parallel_verify(batch, draft_logits=None) is True


def test_can_vocab_parallel_verify_rejects_top_p(monkeypatch):
    monkeypatch.setenv("DSV41_EXP_VOCAB_PARALLEL_SAMPLE", "1")
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

    rs = RejectionSampler.__new__(RejectionSampler)
    rs.watermark_key = None
    rs.synthetic_conditional_rates = None
    rs.use_block_verification = False
    rs.enable_adaptive_verification = False
    rs.sampler = _make_sampler_for_eligibility(temperature=0.7, top_p=0.95)
    batch = SimpleNamespace(idx_mapping_np=np.array([0]))
    assert rs.can_vocab_parallel_verify(batch, draft_logits=None) is False


def test_can_vocab_parallel_verify_flag_off_even_at_temp_gt0(monkeypatch):
    monkeypatch.setenv("DSV41_EXP_VOCAB_PARALLEL_SAMPLE", "0")
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

    rs = RejectionSampler.__new__(RejectionSampler)
    rs.watermark_key = None
    rs.synthetic_conditional_rates = None
    rs.use_block_verification = False
    rs.enable_adaptive_verification = False
    rs.sampler = _make_sampler_for_eligibility(temperature=0.8)
    batch = SimpleNamespace(idx_mapping_np=np.array([0]))
    assert rs.can_vocab_parallel_verify(batch, draft_logits=None) is False


def test_can_vocab_parallel_verify_rejects_probabilistic_drafts(monkeypatch):
    monkeypatch.setenv("DSV41_EXP_VOCAB_PARALLEL_SAMPLE", "1")
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

    rs = RejectionSampler.__new__(RejectionSampler)
    rs.watermark_key = None
    rs.synthetic_conditional_rates = None
    rs.use_block_verification = False
    rs.enable_adaptive_verification = False
    rs.sampler = _make_sampler_for_eligibility(temperature=0.7)
    batch = SimpleNamespace(idx_mapping_np=np.array([0]))
    assert rs.can_vocab_parallel_verify(batch, draft_logits=torch.zeros(1)) is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Gumbel kernel")
def test_vocab_shard_gumbel_matches_full_vocab_gumbel():
    from vllm.v1.worker.gpu.sample.gumbel import (
        gumbel_sample,
        gumbel_sample_with_values,
    )

    device = "cuda"
    torch.manual_seed(3)
    vocab, shard, rows = 64, 32, 5
    full = torch.randn(rows, vocab, device=device)
    idx_map = torch.zeros(rows, dtype=torch.int32, device=device)
    temp = torch.tensor([0.8], dtype=torch.float32, device=device)
    seed = torch.tensor([123], dtype=torch.int64, device=device)
    pos = torch.arange(rows, dtype=torch.int64, device=device)
    expected = gumbel_sample(
        full,
        idx_map,
        temp,
        seed,
        pos,
        apply_temperature=True,
        is_drafting=True,
        key_offset=0,
    )
    pairs = []
    for rank in range(2):
        local = full[:, rank * shard : (rank + 1) * shard].contiguous()
        ids, vals = gumbel_sample_with_values(
            local,
            idx_map,
            temp,
            seed,
            pos,
            apply_temperature=True,
            is_drafting=True,
            key_offset=rank * shard,
        )
        pairs.append(torch.stack([vals.float(), (ids + rank * shard).float()], dim=-1))
    got = reduce_max_idx(
        pairs[0][:, 0],
        pairs[0][:, 1],
        tp_size=2,
        all_gather_fn=_fake_all_gather(pairs),
    )
    assert got.tolist() == expected.tolist()
