# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu.async_utils import AsyncOutput
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


def test_get_copied_draft_tokens_skips_width_placeholders():
    handler = DraftTokensHandler(device=torch.device("cpu"))
    handler._draft_tokens_by_req["a"] = np.array([11, 12], dtype=np.int32)
    handler._draft_tokens_by_req["plain"] = 2

    drafts = handler.get_copied_draft_tokens(["a", "plain", "missing"])
    assert drafts is not None
    assert drafts.req_ids == ["a"]
    assert drafts.draft_token_ids == [[11, 12]]
    assert handler.get_copied_draft_tokens(["plain", "missing"]) is None


def test_async_output_attaches_copied_drafts():
    """get_output() carries this step's real drafts on ModelRunnerOutput."""
    handler = DraftTokensHandler(device=torch.device("cpu"))
    handler._draft_tokens_by_req["a"] = np.array([11, 12, 13], dtype=np.int32)
    handler._draft_tokens_by_req["plain"] = 3

    model_runner_output = ModelRunnerOutput(
        req_ids=["a", "plain"],
        req_id_to_index={"a": 0, "plain": 1},
    )
    async_output = AsyncOutput.__new__(AsyncOutput)
    async_output.model_runner_output = model_runner_output
    async_output.draft_tokens_handler = handler
    async_output.copy_event = MagicMock()
    async_output.sampled_token_ids = np.array([[10], [20]])
    async_output.num_sampled_tokens_np = np.array([1, 1])
    async_output.sampling_mask_tensors = None
    async_output.num_nans = None
    async_output.logprobs_tensors = None
    async_output.prompt_logprobs_dict = {}
    async_output.prompt_token_id_logprobs_dict = {}
    async_output.pending_aux_output = None
    async_output._has_fault = None

    result = async_output.get_output()
    assert result.draft_token_ids is not None
    assert result.draft_token_ids.req_ids == ["a"]
    assert result.draft_token_ids.draft_token_ids == [[11, 12, 13]]
