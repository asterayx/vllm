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
    sm12x_align_tokens,
    sm12x_disable_attn_aux_streams,
    sm12x_dspark_capture_sizes,
    sm12x_mixed_warmup_decode_prompt_len,
    sm12x_mixed_warmup_prefill_len,
    sm12x_pad_token_rows,
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
