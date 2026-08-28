# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the V2 model runner's InputBatch (vllm.v1.worker.gpu.input_batch)."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.input_batch import (
    InputBatch,
    InputBuffers,
    uniform_dummy_num_reqs,
)

DEVICE = current_platform.device_type


@pytest.mark.parametrize(
    "num_reqs,num_tokens",
    [
        (256, 496),  # remainder 240: previously gave the last request 241 tokens
        (128, 512),  # no remainder
        (3, 8),
        (1, 7),
    ],
)
def test_make_dummy_distributes_remainder(num_reqs: int, num_tokens: int):
    """No dummy request may exceed ceil(num_tokens / num_reqs) tokens.

    Dumping the remainder on a single request can produce a dummy request with
    seq_len > max_model_len, which the block tables cannot back; attention
    kernels running on the dummy batch during cudagraph capture then read
    block-table entries out of bounds (https://github.com/vllm-project/vllm/pull/49364
    CI failure).
    """
    buffers = InputBuffers(
        max_num_reqs=num_reqs, max_num_tokens=num_tokens, device=torch.device(DEVICE)
    )
    batch = InputBatch.make_dummy(num_reqs, num_tokens, buffers)

    max_per_req = -(-num_tokens // num_reqs)
    assert batch.num_scheduled_tokens.sum() == num_tokens
    assert batch.num_scheduled_tokens.max() == max_per_req
    assert batch.num_scheduled_tokens.min() >= num_tokens // num_reqs
    # Requests with an extra token are placed at the end of the batch.
    assert (batch.num_scheduled_tokens[:-1] <= batch.num_scheduled_tokens[1:]).all()

    # seq_len == query_len for the dummy prefill-shaped batch, on GPU and CPU.
    query_lens = batch.query_start_loc_np[1:] - batch.query_start_loc_np[:-1]
    assert (query_lens == batch.num_scheduled_tokens).all()
    assert torch.equal(
        batch.seq_lens, torch.from_numpy(batch.num_scheduled_tokens).to(DEVICE)
    )
    assert batch.query_start_loc_np[-1] == num_tokens
    assert torch.equal(
        batch.query_start_loc.cpu(), torch.from_numpy(batch.query_start_loc_np)
    )


@pytest.mark.parametrize(
    "num_tokens,max_num_seqs,expected_reqs",
    [
        (32, 6, 4),  # DSpark capture size that used to assert 32/6
        (16, 6, 4),
        (8, 6, 4),
        (24, 6, 6),
        (36, 6, 6),
        (1, 6, 1),
        (2, 6, 2),
        (4, 6, 4),
    ],
)
def test_uniform_dummy_num_reqs_keeps_dspark_capture_sizes_divisible(
    num_tokens, max_num_seqs, expected_reqs
):
    """PIECEWISE dummy of size 32 with max_num_seqs=6 must not be 6 ragged reqs."""
    num_reqs = uniform_dummy_num_reqs(num_tokens, max_num_seqs)
    assert num_reqs == expected_reqs
    assert num_tokens % num_reqs == 0


def test_piecewise_dummy_batch_32x6_is_uniform_spec_decode_safe():
    """The crashing dummy (32 tokens, 6 seqs) must become a uniform layout."""
    num_tokens = 32
    max_num_seqs = 6
    num_reqs = uniform_dummy_num_reqs(num_tokens, max_num_seqs)
    buffers = InputBuffers(max_num_seqs, num_tokens, torch.device("cpu"))
    batch = InputBatch.make_dummy(num_reqs, num_tokens, buffers)
    query_lens = batch.query_start_loc_np[1:] - batch.query_start_loc_np[:-1]

    assert num_reqs == 4
    assert int(batch.num_scheduled_tokens.sum()) == num_tokens
    assert len(query_lens) == num_reqs
    assert query_lens.min() == query_lens.max() == 8
    assert num_tokens % num_reqs == 0
