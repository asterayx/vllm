# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-side contract for the opt-in SM120 indexer logits shrink.

No GPU: these tests lock the row width and the fallback log. They do not
launch the Triton kernel or report timings.
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.model_executor.layers import sm120_paged_mqa_logits as shrink
from vllm.model_executor.layers.sm120_paged_mqa_logits import (
    INDEXER_LOGITS_SHRINK_ENV,
    commit_indexer_logits_shrink,
    deepgemm_paged_logits_width,
    resolve_sm120_paged_logits_width,
    sm120_paged_mqa_logits_fp8,
)

_MAX_MODEL_LEN = 1_048_576
_USED = 8_000


def _decision(**kwargs):
    params = dict(
        enabled=True,
        capturing=False,
        out_width=None,
    )
    params.update(kwargs)
    return resolve_sm120_paged_logits_width(_MAX_MODEL_LEN, _USED, **params)


def test_flag_off_keeps_full_width_and_does_not_log():
    decision = _decision(enabled=False, capturing=True, out_width=128)
    assert decision.width == _MAX_MODEL_LEN
    assert decision.active is False
    assert decision.reason == "disabled"
    assert decision.log_line is None


def test_shrink_width_covers_used_positions_and_logs():
    decision = _decision()
    assert decision.active is True
    assert decision.reason == "shrunk"
    assert decision.width >= _USED
    assert decision.width < _MAX_MODEL_LEN
    assert decision.width % 128 == 0
    assert decision.log_line is not None
    assert "DSV41_EXP_INDEXER_LOGITS_SHRINK active" in decision.log_line
    assert f"width={decision.width}" in decision.log_line
    assert f"used_len={_USED}" in decision.log_line


def test_capture_falls_back_to_full_width_with_a_log():
    decision = _decision(capturing=True)
    assert decision.width == _MAX_MODEL_LEN
    assert decision.active is False
    assert decision.reason == "cuda_graph_capture"
    assert decision.log_line is not None
    assert "DSV41_EXP_INDEXER_LOGITS_SHRINK fallback" in decision.log_line
    assert "cuda graph capture" in decision.log_line


@pytest.mark.parametrize(
    ("used_len", "reason"),
    [
        (None, "missing_used_len"),
        (0, "invalid_used_len"),
        (-4, "invalid_used_len"),
        (True, "missing_used_len"),
        (_MAX_MODEL_LEN + 1, "used_len_exceeds_max_model_len"),
    ],
)
def test_bad_used_len_falls_back_with_a_log(used_len, reason):
    decision = resolve_sm120_paged_logits_width(
        _MAX_MODEL_LEN, used_len, enabled=True, capturing=False
    )
    assert decision.width == _MAX_MODEL_LEN
    assert decision.active is False
    assert decision.reason == reason
    assert decision.log_line is not None
    assert "fallback" in decision.log_line


def test_bound_that_already_covers_max_model_len_is_not_a_shrink():
    decision = resolve_sm120_paged_logits_width(
        _MAX_MODEL_LEN, _MAX_MODEL_LEN - 10, enabled=True
    )
    assert decision.width == _MAX_MODEL_LEN
    assert decision.active is False
    assert decision.reason == "full_bound"
    assert decision.log_line is None


def test_preallocated_full_buffer_refuses_shrink_with_a_log():
    decision = _decision(out_width=_MAX_MODEL_LEN)
    assert decision.active is False
    assert decision.reason == "preallocated_out"
    assert decision.width == _MAX_MODEL_LEN
    assert decision.log_line is not None
    assert "preallocated" in decision.log_line


def test_numpy_integer_used_len_shrinks():
    numpy = pytest.importorskip("numpy")
    decision = resolve_sm120_paged_logits_width(
        _MAX_MODEL_LEN, numpy.int64(_USED), enabled=True
    )
    assert decision.active is True
    assert decision.width == _decision().width


def _empty_inputs(batch: int, width_pages: int = 4):
    heads, dim, page = 4, 16, 128
    q = torch.empty(batch, 1, heads, dim)
    kv = torch.empty(width_pages, page, dim + 4, dtype=torch.uint8)
    weights = torch.empty(batch, heads)
    context = torch.ones(batch, 1, dtype=torch.int32)
    block_table = torch.zeros(max(batch, 1), 4, dtype=torch.int32)
    return q, kv, weights, context, block_table


def test_allocation_width_follows_the_decision(monkeypatch):
    monkeypatch.delenv(INDEXER_LOGITS_SHRINK_ENV, raising=False)
    q, kv, weights, context, block_table = _empty_inputs(0)
    full = sm120_paged_mqa_logits_fp8(
        q, kv, weights, context, block_table, _MAX_MODEL_LEN, used_len=_USED
    )
    assert full.shape == (0, _MAX_MODEL_LEN)

    monkeypatch.setenv(INDEXER_LOGITS_SHRINK_ENV, "1")
    shrunk = sm120_paged_mqa_logits_fp8(
        q,
        kv,
        weights,
        context,
        block_table,
        _MAX_MODEL_LEN,
        used_len=_USED,
        capturing=False,
    )
    expected = resolve_sm120_paged_logits_width(
        _MAX_MODEL_LEN, _USED, enabled=True
    ).width
    assert shrunk.shape == (0, expected)
    assert expected < _MAX_MODEL_LEN

    with pytest.raises(RuntimeError, match="cuda graph capture"):
        sm120_paged_mqa_logits_fp8(
            q,
            kv,
            weights,
            context,
            block_table,
            _MAX_MODEL_LEN,
            used_len=_USED,
            capturing=True,
        )


def test_launch_width_matches_allocation_and_is_not_silently_full(monkeypatch):
    monkeypatch.setenv(INDEXER_LOGITS_SHRINK_ENV, "1")
    seen: dict[str, int] = {}

    def _fake_launch(*args, **kwargs):
        seen["width"] = args[6]
        seen["rows"] = args[5].shape[0]

    monkeypatch.setattr(shrink, "_launch_sm120_paged_mqa_logits", _fake_launch)
    q, kv, weights, context, block_table = _empty_inputs(2)
    out = sm120_paged_mqa_logits_fp8(
        q,
        kv,
        weights,
        context,
        block_table,
        _MAX_MODEL_LEN,
        used_len=_USED,
        capturing=False,
    )
    assert out.shape[0] == 2
    assert out.shape[1] == seen["width"]
    assert seen["rows"] == 2
    assert seen["width"] < _MAX_MODEL_LEN
    assert seen["width"] % 128 == 0
    assert seen["width"] >= _USED


def test_flag_off_launch_stays_full_width_even_if_used_len_is_short(monkeypatch):
    monkeypatch.delenv(INDEXER_LOGITS_SHRINK_ENV, raising=False)
    seen: dict[str, int] = {}

    def _fake_launch(*args, **kwargs):
        seen["width"] = args[6]

    monkeypatch.setattr(shrink, "_launch_sm120_paged_mqa_logits", _fake_launch)
    q, kv, weights, context, block_table = _empty_inputs(1)
    out = sm120_paged_mqa_logits_fp8(
        q, kv, weights, context, block_table, 4096, used_len=100
    )
    assert out.shape == (1, 4096)
    assert seen["width"] == 4096


def test_active_path_emits_the_attach_line(monkeypatch):
    monkeypatch.setenv(INDEXER_LOGITS_SHRINK_ENV, "1")
    lines: list[str] = []
    monkeypatch.setattr(shrink.logger, "info_once", lines.append)
    q, kv, weights, context, block_table = _empty_inputs(0)
    sm120_paged_mqa_logits_fp8(
        q,
        kv,
        weights,
        context,
        block_table,
        _MAX_MODEL_LEN,
        used_len=_USED,
        capturing=False,
    )
    assert lines
    assert lines[0].startswith("DSV41_EXP_INDEXER_LOGITS_SHRINK active:")


def test_capture_path_refuses_full_width_instead_of_staying_silent(monkeypatch):
    monkeypatch.setenv(INDEXER_LOGITS_SHRINK_ENV, "1")
    errors: list[str] = []
    monkeypatch.setattr(shrink.logger, "error", errors.append)
    q, kv, weights, context, block_table = _empty_inputs(0)
    with pytest.raises(RuntimeError, match="cuda graph capture") as raised:
        sm120_paged_mqa_logits_fp8(
            q,
            kv,
            weights,
            context,
            block_table,
            _MAX_MODEL_LEN,
            used_len=_USED,
            capturing=True,
        )
    assert "DSV41_EXP_INDEXER_LOGITS_SHRINK fallback" in str(raised.value)
    assert errors
    assert "cuda graph capture" in errors[0]


def test_deepgemm_pages_shrink_when_the_flag_is_on():
    """SM120 V4.1 indexer pages are 64, so decode stays on DeepGEMM."""
    off = deepgemm_paged_logits_width(
        _MAX_MODEL_LEN, _USED, 64, enabled=False, capturing=False
    )
    assert off.width == _MAX_MODEL_LEN
    assert off.active is False
    assert off.log_line is None

    on = deepgemm_paged_logits_width(
        _MAX_MODEL_LEN, _USED, 64, enabled=True, capturing=False
    )
    assert on.active is True
    assert on.width >= _USED
    assert on.width < _MAX_MODEL_LEN
    assert on.width % 64 == 0
    assert on.log_line is not None
    assert on.log_line.startswith(
        "DSV41_EXP_INDEXER_LOGITS_SHRINK active: DeepGEMM fp8_fp4_paged_mqa_logits "
    )
    assert f"width={on.width}" in on.log_line
    assert "was not used" not in on.log_line

    # A context that ends mid-page still covers the last DeepGEMM page.
    partial = deepgemm_paged_logits_width(
        _MAX_MODEL_LEN, 129, 64, enabled=True, capturing=False
    )
    assert partial.width >= 192
    assert partial.width % 64 == 0


def test_deepgemm_shrink_refuses_cuda_graph_capture():
    decision = deepgemm_paged_logits_width(
        _MAX_MODEL_LEN, _USED, 64, enabled=True, capturing=True
    )
    with pytest.raises(RuntimeError, match="DeepGEMM fp8_fp4_paged_mqa_logits"):
        commit_indexer_logits_shrink(decision)


def test_unshrinkable_backend_raises_only_when_the_flag_is_set(monkeypatch):
    from vllm.model_executor.layers.sparse_attn_indexer import (
        _refuse_unshrinkable_indexer_logits,
    )

    monkeypatch.delenv(INDEXER_LOGITS_SHRINK_ENV, raising=False)
    _refuse_unshrinkable_indexer_logits("DeepGEMM fp8_fp4_paged_mqa_logits")
    monkeypatch.setenv(INDEXER_LOGITS_SHRINK_ENV, "1")
    with pytest.raises(RuntimeError, match="refusing to keep the full-width"):
        _refuse_unshrinkable_indexer_logits("XPU paged MQA logits")


def test_indexer_wires_shrink_to_max_seq_len_and_full_graph_break():
    from vllm.model_executor.layers import sparse_attn_indexer as indexer

    source = inspect.getsource(indexer)
    assert "used_len=int(attn_metadata_narrowed.max_seq_len)" in source
    assert "break_on_full=_indexer_logits_shrink_breaks_full_graph" in source
    assert "deepgemm_paged_logits_width" in source
    assert "was not used (DeepGEMM fp8_fp4_paged_mqa_logits)" not in source
    assert "max_model_len=logits_width" in source
    assert indexer._indexer_logits_shrink_breaks_full_graph() is (
        shrink.indexer_logits_shrink_enabled()
    )


def test_full_cudagraph_breaks_only_when_the_predicate_says_so(monkeypatch):
    import vllm.envs as envs
    from vllm.compilation import breakable_cudagraph as graphs
    from vllm.config import CUDAGraphMode

    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    envs.disable_envs_cache()

    def _invoke(break_on_full):
        @graphs.eager_break_during_capture(break_on_full=break_on_full)
        def op(value):
            return value + 1

        capture = MagicMock()
        capture._capturing = True
        capture.add_eager.side_effect = lambda fn: fn()
        ctx = MagicMock()
        ctx.cudagraph_runtime_mode = CUDAGraphMode.FULL
        with (
            patch.object(
                graphs.BreakableCUDAGraphCapture,
                "current",
                return_value=capture,
            ),
            patch.object(graphs, "is_forward_context_available", return_value=True),
            patch.object(graphs, "get_forward_context", return_value=ctx),
        ):
            result = op(3)
        return result, capture.add_eager.called

    assert _invoke(None) == (4, False)
    assert _invoke(lambda: False) == (4, False)
    assert _invoke(lambda: True) == (4, True)


def test_breakable_disabled_ignores_full_graph_predicate(monkeypatch):
    import vllm.envs as envs
    from vllm.compilation import breakable_cudagraph as graphs

    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "0")
    envs.disable_envs_cache()

    @graphs.eager_break_during_capture(break_on_full=lambda: True)
    def op(value):
        return value + 1

    assert op(3) == 4
