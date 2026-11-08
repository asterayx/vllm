# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vocab-sharded Gumbel-max / logsumexp reductions for DSpark sample+verify.

Each TP rank keeps its lm_head shard and exchanges O(batch) pairs instead of
all-gathering the full vocabulary. Gated by DSV41_EXP_VOCAB_PARALLEL_SAMPLE.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import torch

from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample_with_values, tl_rand32

AllGatherFn = Callable[[torch.Tensor], torch.Tensor]


def vocab_parallel_sample_enabled() -> bool:
    """True when the opt-in DSpark vocab-parallel sample/verify path is on."""
    return os.environ.get("DSV41_EXP_VOCAB_PARALLEL_SAMPLE", "0") == "1"


def all_gather_last_dim(tensor: torch.Tensor) -> torch.Tensor:
    return tensor_model_parallel_all_gather(tensor, dim=-1)


def reduce_max_idx(
    local_max: torch.Tensor,
    global_idx: torch.Tensor,
    tp_size: int | None = None,
    all_gather_fn: AllGatherFn | None = None,
) -> torch.Tensor:
    """Global argmax from per-rank (max, global_idx) pairs.

    Args:
        local_max: Per-row shard maximum, any floating dtype.
        global_idx: Global vocab index of that maximum.
        tp_size: TP world size. Defaults to the live TP group.
        all_gather_fn: Optional last-dim all-gather. Tests inject a fake.

    Returns:
        Global vocab ids, int64, same leading shape as ``local_max``.

    """
    if tp_size is None:
        tp_size = get_tensor_model_parallel_world_size()
    if tp_size <= 1:
        return global_idx.to(torch.int64)
    gather = all_gather_fn or all_gather_last_dim
    pair = torch.stack([local_max.float(), global_idx.float()], dim=-1)
    gathered = gather(pair).view(*local_max.shape, tp_size, 2)
    best = gathered[..., 0].argmax(dim=-1, keepdim=True)
    return gathered[..., 1].gather(-1, best).squeeze(-1).to(torch.int64)


def combine_logsumexp(
    gathered_max: torch.Tensor,
    gathered_sumexp: torch.Tensor,
) -> torch.Tensor:
    """Reduce per-rank (max, sumexp) along the last dim to a global logsumexp.

    Args:
        gathered_max: ``[..., tp_size]`` shard maxima.
        gathered_sumexp: ``[..., tp_size]`` sum-exp relative to those maxes.

    Returns:
        Global log-sum-exp, same leading shape as a single rank's max.

    """
    gmax = gathered_max.max(dim=-1).values
    stable = gathered_sumexp * torch.exp(gathered_max - gmax.unsqueeze(-1))
    summed = stable.sum(dim=-1)
    return torch.where(
        torch.isfinite(gmax) & (summed > 0),
        gmax + summed.log(),
        gmax,
    )


def shard_logsumexp(
    local_logits: torch.Tensor,
    tp_size: int | None = None,
    all_gather_fn: AllGatherFn | None = None,
) -> torch.Tensor:
    """Vocab-parallel log-sum-exp of ``local_logits`` along the last dim."""
    if tp_size is None:
        tp_size = get_tensor_model_parallel_world_size()
    local_max = local_logits.float().max(dim=-1).values
    local_sumexp = torch.exp(local_logits.float() - local_max.unsqueeze(-1)).sum(dim=-1)
    if tp_size <= 1:
        return torch.where(
            torch.isfinite(local_max) & (local_sumexp > 0),
            local_max + local_sumexp.log(),
            local_max,
        )
    gather = all_gather_fn or all_gather_last_dim
    pair = torch.stack([local_max, local_sumexp], dim=-1)
    gathered = gather(pair).view(*local_max.shape, tp_size, 2)
    return combine_logsumexp(gathered[..., 0], gathered[..., 1])


def gather_token_logits(
    local_logits: torch.Tensor,
    token_ids: torch.Tensor,
    vocab_start: int,
    n_org: int,
    tp_size: int | None = None,
    all_gather_fn: AllGatherFn | None = None,
) -> torch.Tensor:
    """Return ``logits[i, token_ids[i]]`` with ownership reduced over TP.

    Out-of-range or negative ``token_ids`` yield ``-inf``.
    """
    if tp_size is None:
        tp_size = get_tensor_model_parallel_world_size()
    invalid = token_ids < 0
    local_idx = (token_ids - vocab_start).clamp(min=0, max=max(n_org - 1, 0))
    in_range = (token_ids >= vocab_start) & (token_ids < vocab_start + n_org) & ~invalid
    vals = local_logits.gather(-1, local_idx.unsqueeze(-1)).squeeze(-1).float()
    vals = torch.where(in_range, vals, vals.new_full((), float("-inf")))
    if tp_size <= 1:
        return vals
    gather = all_gather_fn or all_gather_last_dim
    return gather(vals.unsqueeze(-1)).max(dim=-1).values


def vocab_shard_argmax(
    local_logits: torch.Tensor,
    vocab_start: int,
    tp_size: int | None = None,
    all_gather_fn: AllGatherFn | None = None,
) -> torch.Tensor:
    """Greedy global argmax of vocab-sharded logits."""
    local_max, local_idx = local_logits.max(dim=-1)
    return reduce_max_idx(
        local_max,
        local_idx + vocab_start,
        tp_size=tp_size,
        all_gather_fn=all_gather_fn,
    )


def vocab_shard_gumbel_sample(
    local_logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seed: torch.Tensor,
    pos: torch.Tensor,
    vocab_start: int,
    *,
    apply_temperature: bool,
    is_drafting: bool,
    use_fp64: bool = False,
    tp_size: int | None = None,
    all_gather_fn: AllGatherFn | None = None,
) -> torch.Tensor:
    """Gumbel-max sample on a vocab shard, then reduce (max, idx) over TP.

    Noise keys are global vocab ids (``local_col + vocab_start``), matching a
    full-vocabulary ``gumbel_sample`` with the same seeds.
    """
    if tp_size is None:
        tp_size = get_tensor_model_parallel_world_size()
    local_ids, local_values = gumbel_sample_with_values(
        local_logits,
        expanded_idx_mapping,
        temperature,
        seed,
        pos,
        apply_temperature=apply_temperature,
        is_drafting=is_drafting,
        use_fp64=use_fp64,
        key_offset=vocab_start,
    )
    return reduce_max_idx(
        local_values,
        local_ids + vocab_start,
        tp_size=tp_size,
        all_gather_fn=all_gather_fn,
    )


def apply_row_temperature(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
) -> torch.Tensor:
    """FP32 copy of ``logits`` with per-row temperature, matching the sampler.

    Rows with temperature 0 or 1 are left unscaled. Other rows are divided
    on the copy.
    """
    scaled = torch.empty_like(logits, dtype=torch.float32).copy_(logits)
    safe_idx = expanded_idx_mapping.clamp(min=0)
    temp = temperature[safe_idx].to(torch.float32)
    need = (expanded_idx_mapping >= 0) & (temp != 0.0) & (temp != 1.0)
    if torch.any(need):
        scaled[need] = scaled[need] / temp[need].unsqueeze(-1)
    return scaled


def mask_rejected_token(
    local_logits: torch.Tensor,
    token_ids: torch.Tensor,
    vocab_start: int,
    n_org: int,
) -> torch.Tensor:
    """Set owned ``token_ids`` to ``-inf`` (one-hot-draft residual)."""
    out = local_logits.clone()
    invalid = token_ids < 0
    in_range = (token_ids >= vocab_start) & (token_ids < vocab_start + n_org) & ~invalid
    if not torch.any(in_range):
        return out
    rows = torch.arange(out.shape[0], device=out.device)
    local_idx = (token_ids - vocab_start).clamp(min=0)
    out[rows[in_range], local_idx[in_range]] = float("-inf")
    return out


def proposal_tokens_from_draft_sampled(
    draft_sampled: torch.Tensor,
    cu_num_logits: torch.Tensor,
) -> torch.Tensor:
    """Target-row draft ids: ``draft_sampled[row+1]``, ``-1`` on bonus rows."""
    verify_tokens = draft_sampled.new_full(draft_sampled.shape, -1)
    if draft_sampled.numel() == 0:
        return verify_tokens
    verify_tokens[:-1] = draft_sampled[1:]
    last_rows = cu_num_logits[1:] - 1
    verify_tokens[last_rows] = -1
    return verify_tokens


@triton.jit
def _one_hot_ratio_verify_kernel(
    sampled_ptr,
    sampled_stride,
    num_sampled_ptr,
    resample_row_ptr,
    resample_draft_ptr,
    top_tokens_ptr,
    proposal_logit_ptr,
    target_lse_ptr,
    draft_sampled_ptr,
    cu_num_logits_ptr,
    idx_mapping_ptr,
    pos_ptr,
    temp_ptr,
    seed_ptr,
):
    """Leviathan one-hot-q verify: greedy argmax or p(x)>u ratio test."""
    req_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx).to(tl.int64)
    start_idx = tl.load(cu_num_logits_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1).to(tl.int64)
    num_draft_tokens = end_idx - start_idx - 1
    seed = tl.load(seed_ptr + req_state_idx)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    is_greedy = temp == 0.0
    accepted_length = tl.zeros((), tl.int32)
    verifying = True
    for i in range(num_draft_tokens):
        if verifying:
            logit_idx = start_idx + i
            draft = tl.load(draft_sampled_ptr + logit_idx + 1).to(tl.int64)
            is_valid = draft >= 0
            if is_greedy:
                target = tl.load(top_tokens_ptr + logit_idx).to(tl.int64)
                accepted = (target == draft) & is_valid
                verifying = accepted
                accepted_length += accepted.to(tl.int32)
                tl.store(
                    sampled_ptr + req_idx * sampled_stride + i,
                    tl.where(accepted, draft, target),
                )
            else:
                verifying &= is_valid
                if verifying:
                    pos = tl.load(pos_ptr + logit_idx)
                    u = tl_rand32(seed, pos, includes_zero=False)
                    logit = tl.load(proposal_logit_ptr + logit_idx).to(tl.float32)
                    lse = tl.load(target_lse_ptr + logit_idx).to(tl.float32)
                    accepted = logit - lse > tl.log(u)
                    verifying = accepted
                    accepted_length += accepted.to(tl.int32)
                    tl.store(sampled_ptr + req_idx * sampled_stride + i, draft)
    if is_greedy:
        if accepted_length == num_draft_tokens:
            bonus = tl.load(top_tokens_ptr + end_idx - 1).to(tl.int64)
            tl.store(
                sampled_ptr + req_idx * sampled_stride + num_draft_tokens,
                bonus,
            )
        tl.store(num_sampled_ptr + req_idx, accepted_length + 1)
        tl.store(resample_row_ptr + req_idx, -1)
        tl.store(resample_draft_ptr + req_idx, -1)
    else:
        resample_idx = start_idx + accepted_length
        is_bonus = resample_idx == end_idx - 1
        rejected_draft = tl.load(
            draft_sampled_ptr + resample_idx + 1,
            mask=not is_bonus,
            other=-1,
        ).to(tl.int64)
        tl.store(num_sampled_ptr + req_idx, accepted_length)
        tl.store(resample_row_ptr + req_idx, resample_idx)
        tl.store(resample_draft_ptr + req_idx, rejected_draft)


def one_hot_ratio_verify(
    top_tokens: torch.Tensor,
    proposal_logit: torch.Tensor,
    target_lse: torch.Tensor,
    draft_sampled: torch.Tensor,
    pos: torch.Tensor,
    cu_num_logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seed: torch.Tensor,
    num_speculative_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the one-hot Leviathan kernel.

    Returns:
        sampled, num_sampled (greedy already includes the bonus/reject
        replacement; sampled rows do not), resample_row, resample_draft.

    """
    num_reqs = cu_num_logits.shape[0] - 1
    sampled = draft_sampled.new_empty(
        num_reqs, num_speculative_steps + 1, dtype=torch.int64
    )
    num_sampled = sampled.new_empty(num_reqs, dtype=torch.int32)
    resample_row = sampled.new_empty(num_reqs, dtype=torch.int64)
    resample_draft = sampled.new_empty(num_reqs, dtype=torch.int64)
    _one_hot_ratio_verify_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        resample_row,
        resample_draft,
        top_tokens,
        proposal_logit,
        target_lse,
        draft_sampled,
        cu_num_logits,
        idx_mapping,
        pos,
        temperature,
        seed,
        num_warps=1,
    )
    return sampled, num_sampled, resample_row, resample_draft
