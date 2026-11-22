# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable
from typing import Any, NamedTuple

import numpy as np
import torch

from vllm.config import LoadConfig, replace
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import DraftTokenIds
from vllm.v1.worker.gpu.async_utils import async_copy_to_np, new_d2h_stream
from vllm.v1.worker.gpu.input_batch import InputBatch

logger = init_logger(__name__)


class DraftCopySnapshot(NamedTuple):
    """Per-step CPU draft copy. Owned by that step's AsyncOutput."""

    req_ids: list[str]
    draft_tokens_np: np.ndarray
    copy_event: Any | None


def max_num_tokens_after_draft_slots(
    max_num_tokens: int, num_reqs: int, max_num_new_slots_per_req: int
) -> int:
    """Return the input-token budget after reserving each request's draft slots."""
    return max_num_tokens - num_reqs * max_num_new_slots_per_req


def get_drafter_hidden_states(
    model: torch.nn.Module, hidden_states: torch.Tensor
) -> torch.Tensor:
    """Hidden states fed to the drafter.

    Targets such as DeepSeek V4 expose the pre-hc_head residual through
    get_mtp_target_hidden_states(). The buffer is sized at
    max_num_batched_tokens and only allocated for drafters that consume
    target hidden states, so None means "use the regular hidden states".
    """
    get_target_hidden_states = getattr(model, "get_mtp_target_hidden_states", None)
    if get_target_hidden_states is None:
        return hidden_states
    target_hidden_states = get_target_hidden_states()
    if target_hidden_states is None:
        return hidden_states
    return target_hidden_states[: hidden_states.shape[0]]


def get_pp_safe_draft_load_config(load_config: LoadConfig) -> LoadConfig:
    """Avoid collectives that include PP ranks without a draft model."""
    if get_pp_group().world_size > 1 and load_config.load_format == "fastsafetensors":
        logger.warning_once(
            "fastsafetensors cannot load a draft model instantiated on only "
            "one pipeline stage; falling back to the standard safetensors "
            "loader for the draft model. The target model still uses "
            "fastsafetensors."
        )
        return replace(load_config, load_format="auto")
    return load_config


class DraftTokensHandler:
    def __init__(self, device: torch.device | None = None):
        self.device = device
        # Lazily created so CPU unit tests can exercise the per-request store.
        self.copy_stream: torch.cuda.Stream | None = None
        self.copy_event: torch.cuda.Event | None = None
        self._copy_pending = False

        # Latest drafts per live request. A numpy row is a view into an
        # async-copied array (synced in get_draft_tokens). An int is the draft
        # width when the scheduler does not need the token ids.
        self._draft_tokens_by_req: dict[str, np.ndarray | int] = {}

        self._deferred_req_ids: Iterable[str] | None = None

    def begin_step(self, scheduler_output: SchedulerOutput) -> None:
        # get_draft_tokens is only called for the most recently executed
        # scheduler output.
        self._deferred_req_ids = (
            scheduler_output.scheduled_spec_decode_tokens.keys()
            if scheduler_output.pending_structured_output_tokens
            else None
        )

    def _ensure_copy_resources(self) -> None:
        if self.copy_stream is None:
            self.copy_stream = new_d2h_stream(self.device)

    def set_draft_tokens(
        self, input_batch: InputBatch, draft_tokens: torch.Tensor
    ) -> DraftCopySnapshot | None:
        if not input_batch.has_structured_output_reqs:
            # No draft token validation needs to be performed by
            # the scheduler for this batch. Record widths only so a later
            # structured microbatch cannot drop these request ids.
            num_draft_tokens = int(draft_tokens.shape[1])
            for req_id in input_batch.req_ids:
                self._draft_tokens_by_req[req_id] = num_draft_tokens
            return None

        # For spec decoding + structured outputs, we must transfer the
        # draft tokens back to the scheduler for grammar validation.
        is_cpu = self.device is None or self.device.type == "cpu"
        if is_cpu:
            draft_tokens_np = draft_tokens.detach().cpu().numpy().copy()
            copy_event = None
        else:
            self._ensure_copy_resources()
            assert self.copy_stream is not None
            current_stream = torch.cuda.current_stream(self.device)
            self.copy_stream.wait_stream(current_stream)
            # Per-step event so the output thread never waits on a later
            # sample_tokens recording of the shared copy_event.
            copy_event = torch.cuda.Event(blocking=True)
            with torch.cuda.stream(self.copy_stream):
                draft_tokens_np = async_copy_to_np(draft_tokens)
                # draft_tokens is a temporary allocation on the main stream
                # and read here on copy_stream; without record_stream, the
                # caching allocator may reuse its memory before the async
                # copy executes.
                draft_tokens.record_stream(self.copy_stream)
                copy_event.record()
            self.copy_event = copy_event
            self._copy_pending = True

        for req_id, row in zip(input_batch.req_ids, draft_tokens_np, strict=True):
            self._draft_tokens_by_req[req_id] = row
        return DraftCopySnapshot(list(input_batch.req_ids), draft_tokens_np, copy_event)

    def discard(self, req_id: str) -> None:
        self._draft_tokens_by_req.pop(req_id, None)

    def get_draft_tokens(self) -> DraftTokenIds | None:
        """Return the latest drafts needed for the current step's deferred
        grammar bitmask, or for all live requests if sampling isn't deferred.
        Requests without retained drafts are omitted."""
        if self._copy_pending:
            assert self.copy_event is not None
            self.copy_event.synchronize()
            self._copy_pending = False

        drafts_by_req = self._draft_tokens_by_req
        req_ids = self._deferred_req_ids
        if req_ids is None:
            out_req_ids = list(drafts_by_req)
        else:
            out_req_ids = [req_id for req_id in req_ids if req_id in drafts_by_req]
        draft_token_ids: list[list[int]] = []
        for req_id in out_req_ids:
            draft_tokens = drafts_by_req[req_id]
            if isinstance(draft_tokens, int):
                # Construct placeholders from draft token width
                draft_tokens = [-1] * draft_tokens
            else:
                # Convert numpy view to list
                draft_tokens = draft_tokens.tolist()
            draft_token_ids.append(draft_tokens)

        return DraftTokenIds(out_req_ids, draft_token_ids)


def get_parallel_drafting_token_id(hf_config) -> int:
    """Resolve the mask token id used for parallel drafting slots.

    Checks (in order): `dflash_config.mask_token_id`, top-level `mask_token_id`,
    `dspark_noise_token_id`, `pard_token`, `ptd_token_id`. Raises ValueError if
    none are present.
    """
    dflash_config = getattr(hf_config, "dflash_config", None) or {}
    if "mask_token_id" in dflash_config:
        return int(dflash_config["mask_token_id"])
    if getattr(hf_config, "mask_token_id", None) is not None:
        return int(hf_config.mask_token_id)
    if hasattr(hf_config, "dspark_noise_token_id"):
        return int(hf_config.dspark_noise_token_id)
    if hasattr(hf_config, "pard_token"):
        return int(hf_config.pard_token)
    if hasattr(hf_config, "ptd_token_id"):
        return int(hf_config.ptd_token_id)
    raise ValueError(
        "Model config must specify `dflash_config.mask_token_id`,"
        " `mask_token_id`, `dspark_noise_token_id`, `pard_token`, or"
        " `ptd_token_id` for parallel drafting."
    )
