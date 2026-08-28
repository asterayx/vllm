# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the max_num_reqs gate on the V2 mixed prefill+decode warmup."""

from types import SimpleNamespace

import pytest
import torch

from vllm.utils.sm12x import SM12X_SPARSE_PREFILL_MIN_TOKENS
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec
from vllm.v1.worker.gpu.warmup import run_mixed_prefill_decode_warmup


def _fail(*args, **kwargs):
    raise AssertionError("worker callback must not run when warmup is skipped")


def _sm12x_mixed_runner() -> SimpleNamespace:
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    return SimpleNamespace(
        is_pooling_model=False,
        max_num_reqs=4,
        max_model_len=1024,
        model_state=SimpleNamespace(max_encoder_len=0),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[KVCacheGroupSpec(["layer"], spec)],
            num_blocks=1024,
        ),
        vllm_config=SimpleNamespace(num_lookahead_tokens=6),
        kv_connector=SimpleNamespace(set_disabled=lambda disabled: None),
    )


@pytest.mark.parametrize("max_num_reqs", [1, 0])
def test_mixed_warmup_skipped_for_single_seq(max_num_reqs):
    """A mixed prefill+decode step needs >=2 requests; with max_num_reqs < 2
    the warmup must be skipped without touching the worker callbacks."""
    runner = SimpleNamespace(is_pooling_model=False, max_num_reqs=max_num_reqs)

    assert (
        run_mixed_prefill_decode_warmup(
            runner,
            worker_execute_model=_fail,
            worker_sample_tokens=_fail,
            num_tokens=128,
        )
        is False
    )


def test_mixed_warmup_sm12x_seeds_at_safe_prefill_width(monkeypatch):
    """SM12x must run mixed warmup with a >64-token seed, not skip or q_len=2/4."""
    from vllm.utils import sm12x as sm12x_utils

    monkeypatch.setattr(
        sm12x_utils.current_platform,
        "is_device_capability_family",
        lambda fam: fam == 120,
    )
    scheduled: list[dict[str, int]] = []

    def _record_execute(scheduler_output) -> None:
        scheduled.append(dict(scheduler_output.num_scheduled_tokens))

    assert run_mixed_prefill_decode_warmup(
        _sm12x_mixed_runner(),
        worker_execute_model=_record_execute,
        worker_sample_tokens=lambda grammar_output=None: None,
        num_tokens=16,
    )
    assert scheduled[0] == {"_v2_mixed_warmup_decode_": SM12X_SPARSE_PREFILL_MIN_TOKENS}
    assert scheduled[1] == {
        "_v2_mixed_warmup_decode_": 1,
        "_v2_mixed_warmup_prefill_": SM12X_SPARSE_PREFILL_MIN_TOKENS,
    }
