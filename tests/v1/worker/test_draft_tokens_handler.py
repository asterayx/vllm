# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler

pytestmark = pytest.mark.cpu_test


def _unstructured_batch(req_ids: list[str]) -> SimpleNamespace:
    return SimpleNamespace(req_ids=req_ids, has_structured_output_reqs=False)


def _get_drafts(handler: DraftTokensHandler) -> dict[str, list[int]]:
    drafts = handler.get_draft_tokens()
    assert drafts is not None
    return dict(zip(drafts.req_ids, drafts.draft_token_ids, strict=True))


def test_later_microbatch_keeps_earlier_request_drafts():
    handler = DraftTokensHandler(device=torch.device("cpu"))
    handler._draft_tokens_by_req["a"] = np.array([11, 12], dtype=np.int32)
    handler._draft_tokens_by_req["b"] = np.array([21, 22], dtype=np.int32)

    handler.set_draft_tokens(
        _unstructured_batch(["plain"]),
        torch.tensor([[31, 32]], dtype=torch.int32),
    )
    assert _get_drafts(handler) == {
        "a": [11, 12],
        "b": [21, 22],
        "plain": [-1, -1],
    }

    handler.discard("b")
    assert _get_drafts(handler) == {
        "a": [11, 12],
        "plain": [-1, -1],
    }
    handler.discard("a")
    handler.discard("plain")
    assert _get_drafts(handler) == {}


def test_set_draft_tokens_merges_by_request_id():
    handler = DraftTokensHandler(device=torch.device("cpu"))
    handler.set_draft_tokens(
        _unstructured_batch(["a", "b"]),
        torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
    )
    handler.set_draft_tokens(
        _unstructured_batch(["c"]),
        torch.tensor([[5, 6, 7]], dtype=torch.int32),
    )
    drafts = _get_drafts(handler)
    assert drafts["a"] == [-1, -1]
    assert drafts["b"] == [-1, -1]
    assert drafts["c"] == [-1, -1, -1]
    assert set(drafts) == {"a", "b", "c"}
