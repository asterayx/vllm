# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, NamedTuple

import numpy as np
import torch

from vllm.config import LoadConfig, replace
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.v1.outputs import DraftTokenIds
from vllm.v1.worker.gpu.async_utils import async_copy_to_np
from vllm.v1.worker.gpu.input_batch import InputBatch

logger = init_logger(__name__)


class DraftCopySnapshot(NamedTuple):
    """Per-step CPU draft copy. Owned by that step's AsyncOutput."""

    req_ids: list[str]
    draft_tokens_np: np.ndarray
    copy_event: Any | None


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

    def _ensure_copy_resources(self) -> None:
        if self.copy_stream is None:
            self.copy_stream = torch.cuda.Stream(self.device)

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
            draft_tokens_np = np.ascontiguousarray(draft_tokens.detach().cpu().numpy())
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
        """Return the latest retained drafts for every live request.

        Under pipeline parallelism the most recent microbatch may not include a
        request whose drafts were produced earlier; those rows stay in the
        per-request store until overwritten or discarded.
        """
        if self._copy_pending:
            assert self.copy_event is not None
            self.copy_event.synchronize()
            self._copy_pending = False

        out_req_ids = list(self._draft_tokens_by_req)
        draft_token_ids: list[list[int]] = []
        for req_id in out_req_ids:
            draft_tokens = self._draft_tokens_by_req[req_id]
            if isinstance(draft_tokens, int):
                draft_tokens = [-1] * draft_tokens
            else:
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
