# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in Causal Encoder-Decoder (CED) prefill for DeepSeek-V4.1.

CED, as used by the V4.1 design: the boundary layer (the last ratio-1 KV
source that also writes index keys) still runs on every row of the current
prefill chunk, so the full-resolution global KV and index keys land in the
cache. Later decoder layers then run on a dependency-preserving tail of each
prefill request. Long-context attention still reads the global cache; only
the queries shrink.

The tail is wider than the sliding window. Causal SWA of width ``W`` at each
of the ``D`` compact layers needs ``W - 1`` extra positions of the previous
layer, and decode reads the last ``W`` keys at the final layer, so the first
compact layer must see

    W + (D - 1) * (W - 1)

rows ending at the chunk. That is 2414 on Flash (``W=128``, 19 layers after
layer 20). ``DSV41_EXP_CED_PREFILL_ROWS``, when set, replaces that width.

Unset ``DSV41_EXP_CED_PREFILL`` leaves the full-chunk prefill untouched.

The hierarchical indexer pool is a separate, already-wired mechanism:
``candidate_source_layer_id`` publishes ``candidate_topk_blocks`` blocks of
``candidate_block_size`` compressed positions (Flash: 2048 x 8 = 16384), and
later reindex layers score that pool before the top-512. This path does not
change it. Reindex queries are the compacted tail, so they read the candidate
rows the boundary wrote for those same tokens.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace
from typing import Any

import torch

from vllm.forward_context import (
    get_forward_context,
    is_forward_context_available,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

CED_PREFILL_ENV = "DSV41_EXP_CED_PREFILL"
CED_PREFILL_ROWS_ENV = "DSV41_EXP_CED_PREFILL_ROWS"
# Used only when the checkpoint does not set ``sliding_window``.
DEFAULT_CED_WINDOW = 128

_LAYER_ID = re.compile(r"\.layers\.(\d+)\.")
_logged: set[str] = set()


def _log_once(level: str, reason: str) -> None:
    if reason in _logged:
        return
    _logged.add(reason)
    getattr(logger, level)("%s", reason)


def reset_ced_prefill_logs() -> None:
    """Clear one-shot log keys. Tests use this."""
    _logged.clear()


def ced_prefill_enabled() -> bool:
    """True only for the explicit opt-in. Unset and ``0`` stay off."""
    return os.environ.get(CED_PREFILL_ENV, "0") == "1"


def ced_rows_override() -> tuple[int | None, str | None]:
    """Explicit tail width, or ``(None, None)`` when the knob is unset.

    Unset means the dependency-preserving halo, not the bare sliding window.
    """
    raw = os.environ.get(CED_PREFILL_ROWS_ENV)
    if raw is None or raw == "":
        return None, None
    try:
        rows = int(raw)
    except ValueError:
        return None, f"{CED_PREFILL_ROWS_ENV}={raw!r} is not an integer"
    if rows <= 0:
        return None, f"{CED_PREFILL_ROWS_ENV}={rows} must be positive"
    return rows, None


def sliding_window_of(hf_config: Any) -> int:
    window = getattr(hf_config, "sliding_window", None)
    if isinstance(window, int) and window > 0:
        return window
    return DEFAULT_CED_WINDOW


def compact_layer_count(num_hidden_layers: int, boundary: int) -> int:
    """Decoder layers that run after the boundary and therefore see the tail."""
    return max(0, num_hidden_layers - boundary - 1)


def dependency_halo_rows(window: int, compact_layers: int) -> int:
    """Trailing rows whose final-layer states match a full prefill.

    Args:
        window: Causal SWA width (``sliding_window``).
        compact_layers: How many decoder layers consume the tail.

    Returns:
        ``window + (compact_layers - 1) * (window - 1)`` when both are
        positive, else 0. The last ``window`` outputs of the final compact
        layer then have a complete local cone, which is what the sampled
        token and the next decode step's SWA read.

    """
    if window <= 0 or compact_layers <= 0:
        return 0
    return window + (compact_layers - 1) * (window - 1)


def full_resolution_boundary(hf_config: Any) -> int | None:
    """Last layer that writes full-resolution (ratio-1) global KV and index keys.

    Encoder KV sources pool tokens (ratio 2 on Flash) and are not this boundary.
    """
    ratios = getattr(hf_config, "compress_ratios", None)
    kv_sources = getattr(hf_config, "kv_source_layer_ids", None) or ()
    index_sources = {
        int(layer_id)
        for layer_id in (getattr(hf_config, "index_source_layer_ids", None) or ())
    }
    if not ratios or not kv_sources or not index_sources:
        return None
    found: list[int] = []
    for layer_id in kv_sources:
        layer_id = int(layer_id)
        if layer_id < 0 or layer_id >= len(ratios):
            continue
        if int(ratios[layer_id]) != 1 or layer_id not in index_sources:
            continue
        found.append(layer_id)
    return max(found) if found else None


def _reindex_after(hf_config: Any, boundary: int) -> bool:
    sources = getattr(hf_config, "index_source_layer_ids", None) or ()
    return any(int(layer_id) > boundary for layer_id in sources)


@dataclass(frozen=True)
class CedPrefillConfig:
    """Active CED prefill on this PP rank. Absent when the flag is off."""

    boundary_layer: int
    tail_rows: int
    reindex_after_boundary: bool
    window: int = DEFAULT_CED_WINDOW
    compact_layers: int = 0
    formula_rows: int = 0
    rows_overridden: bool = False


def ced_attach_message(config: CedPrefillConfig) -> str:
    """One-time attach line. Names the halo unless a row override is set."""
    if config.rows_overridden:
        return (
            "DSV41 CED prefill attached: after layer "
            f"{config.boundary_layer} writes full-resolution global KV and "
            f"index keys, later decoder layers compute the last "
            f"{config.tail_rows} prefill rows ({CED_PREFILL_ROWS_ENV} "
            f"overrides the dependency-preserving halo of "
            f"{config.formula_rows}). Unset {CED_PREFILL_ENV} keeps full "
            "prefill."
        )
    lookback_layers = max(0, config.compact_layers - 1)
    return (
        "DSV41 CED prefill attached: after layer "
        f"{config.boundary_layer} writes full-resolution global KV and index "
        "keys, later decoder layers compute a dependency-preserving tail of "
        f"{config.tail_rows} rows per prefill chunk (sliding window "
        f"{config.window} plus {lookback_layers} layers of "
        f"{config.window - 1}-token left context). Unset {CED_PREFILL_ENV} "
        "keeps full prefill."
    )


def ced_compact_message(
    config: CedPrefillConfig, full_prefill_tokens: int, kept_prefill_tokens: int
) -> str:
    """First-compaction line. The boundary write count stays in the message."""
    wrote = (
        "boundary wrote full-resolution global KV and index keys for all "
        f"{full_prefill_tokens} prefill rows; decoder layers see "
        f"{kept_prefill_tokens}"
    )
    if config.rows_overridden:
        return (
            "DSV41 CED prefill compacted to the last "
            f"{config.tail_rows} prefill rows after layer {config.boundary_layer} "
            f"({CED_PREFILL_ROWS_ENV} overrides the dependency-preserving halo "
            f"of {config.formula_rows}; {wrote})."
        )
    return (
        "DSV41 CED prefill compacted to a dependency-preserving tail of "
        f"{config.tail_rows} rows after layer {config.boundary_layer} "
        f"(sliding window {config.window}, {config.compact_layers} decoder "
        f"layers after the boundary; {wrote})."
    )


def decide_ced_rank(
    *,
    boundary: int | None,
    num_hidden_layers: int,
    start_layer: int,
    end_layer: int,
    tail_rows: int,
    use_sequence_parallel: bool,
    prefill_context_parallel_size: int,
    decode_context_parallel_size: int,
    use_mega_moe: bool,
    engram_layer_ids: tuple[int, ...] | list[int],
) -> tuple[str, str | None]:
    """Return ``("attach"|"skip"|"refuse", reason)``.

    ``skip`` is an encoder-only PP rank: compaction happens on the rank that
    owns the boundary and every later layer. ``refuse`` keeps full prefill.
    """
    if boundary is None:
        reason = (
            "no full-resolution KV/index boundary "
            "(need a ratio-1 kv source that is also an index source)"
        )
        return "refuse", reason
    if tail_rows <= 0:
        return "refuse", "tail rows must be positive"
    if use_sequence_parallel:
        return "refuse", "sequence parallel shards the token axis"
    if prefill_context_parallel_size > 1 or decode_context_parallel_size > 1:
        return "refuse", "context parallel is not supported"
    if use_mega_moe:
        return "refuse", "MegaMoE precomputes routing for the full chunk"
    later_engram = [int(layer) for layer in engram_layer_ids if int(layer) > boundary]
    if later_engram:
        reason = (
            f"engram layers {later_engram} sit after boundary {boundary} "
            "and need the full token stream"
        )
        return "refuse", reason
    local_ids = range(start_layer, end_layer)
    has_local_later = any(layer_id > boundary for layer_id in local_ids)
    owns_boundary = start_layer <= boundary < end_layer
    owns_all_later = end_layer == num_hidden_layers
    has_later = boundary + 1 < num_hidden_layers
    if not has_local_later:
        if owns_boundary and not has_later:
            reason = (
                f"boundary layer {boundary} is the last layer; "
                "no subsequent decoder layers to compact"
            )
            return "refuse", reason
        return "skip", None
    if not (owns_boundary and owns_all_later and has_later):
        reason = (
            f"PP layers [{start_layer}, {end_layer}) do not own boundary "
            f"layer {boundary} and every later decoder layer "
            f"(num_hidden_layers={num_hidden_layers})"
        )
        return "refuse", reason
    return "attach", None


def build_ced_prefill(
    vllm_config: Any,
    *,
    start_layer: int,
    end_layer: int,
    use_sequence_parallel: bool,
    use_mega_moe: bool,
) -> CedPrefillConfig | None:
    """Resolve the opt-in for this rank. ``None`` means full prefill."""
    if not ced_prefill_enabled():
        return None
    hf_config = vllm_config.model_config.hf_config
    parallel = vllm_config.parallel_config
    override, tail_error = ced_rows_override()
    if tail_error is not None:
        _log_once("warning", f"DSV41 CED prefill refused: {tail_error}")
        return None
    boundary = full_resolution_boundary(hf_config)
    window = sliding_window_of(hf_config)
    compact_layers = (
        compact_layer_count(int(hf_config.num_hidden_layers), boundary)
        if boundary is not None
        else 0
    )
    formula_rows = dependency_halo_rows(window, compact_layers)
    if override is None:
        # A zero formula means there is no compact stack. Pass a positive
        # placeholder so decide_ced_rank can report the real reason.
        tail_rows = formula_rows if formula_rows > 0 else 1
        rows_overridden = False
    else:
        tail_rows = override
        rows_overridden = True
    engram_ids = getattr(hf_config, "engram_layer_ids", None) or ()
    decision, reason = decide_ced_rank(
        boundary=boundary,
        num_hidden_layers=int(hf_config.num_hidden_layers),
        start_layer=start_layer,
        end_layer=end_layer,
        tail_rows=tail_rows,
        use_sequence_parallel=use_sequence_parallel,
        prefill_context_parallel_size=int(
            getattr(parallel, "prefill_context_parallel_size", 1) or 1
        ),
        decode_context_parallel_size=int(
            getattr(parallel, "decode_context_parallel_size", 1) or 1
        ),
        use_mega_moe=use_mega_moe,
        engram_layer_ids=engram_ids,
    )
    if decision == "skip":
        logger.debug(
            "DSV41 CED prefill is not applied on PP layers [%d, %d); "
            "boundary layer %s is outside this rank's post-boundary span.",
            start_layer,
            end_layer,
            boundary,
        )
        return None
    if decision == "refuse":
        _log_once("warning", f"DSV41 CED prefill refused: {reason}")
        return None
    assert boundary is not None
    config = CedPrefillConfig(
        boundary_layer=boundary,
        tail_rows=tail_rows,
        reindex_after_boundary=_reindex_after(hf_config, boundary),
        window=window,
        compact_layers=compact_layers,
        formula_rows=formula_rows,
        rows_overridden=rows_overridden,
    )
    logger.info("%s", ced_attach_message(config))
    return config


@dataclass(frozen=True)
class PrefillKeepPlan:
    """Rows kept after the boundary, in batch order (decodes, then tails)."""

    keep_indices: tuple[int, ...]
    prefill_local: tuple[int, ...]
    query_lens: tuple[int, ...]
    num_decodes: int
    num_prefills: int
    num_decode_tokens: int
    num_prefill_tokens: int
    full_prefill_tokens: int
    dropped_prefill_rows: int

    @property
    def prefill_query_lens(self) -> tuple[int, ...]:
        return self.query_lens[self.num_decodes :]


def plan_prefill_keep(
    query_start_loc: list[int],
    num_decodes: int,
    num_decode_tokens: int,
    tail_rows: int,
) -> PrefillKeepPlan | str:
    """Plan the compact batch. A plan with ``dropped_prefill_rows == 0`` is a no-op.

    Decode requests (DSpark / speculative verify included) stay whole. Each
    prefill request keeps its last ``tail_rows`` query rows. Callers pass the
    dependency-preserving halo as that width unless an env override is set.
    The tail still ends at the original context length, so long-context
    reads stay valid.
    """
    if tail_rows <= 0:
        return "tail rows must be positive"
    if not query_start_loc or query_start_loc[0] != 0:
        return "query_start_loc is empty or does not start at 0"
    num_reqs = len(query_start_loc) - 1
    if num_decodes < 0 or num_decodes > num_reqs:
        return "decode request count is outside the batch"
    if query_start_loc[num_decodes] != num_decode_tokens:
        return (
            f"decode tokens ({num_decode_tokens}) do not match "
            f"query_start_loc ({query_start_loc[num_decodes]})"
        )
    keep: list[int] = []
    query_lens: list[int] = []
    for req in range(num_reqs):
        start = int(query_start_loc[req])
        end = int(query_start_loc[req + 1])
        if end < start:
            return "query_start_loc is not monotonic"
        qlen = end - start
        if req < num_decodes:
            kept = qlen
            keep.extend(range(start, end))
        else:
            kept = qlen if qlen <= tail_rows else tail_rows
            keep.extend(range(end - kept, end))
        query_lens.append(kept)
    full_prefill = int(query_start_loc[-1]) - num_decode_tokens
    kept_prefill = sum(query_lens[num_decodes:])
    prefill_local = tuple(
        index - num_decode_tokens for index in keep if index >= num_decode_tokens
    )
    return PrefillKeepPlan(
        keep_indices=tuple(keep),
        prefill_local=prefill_local,
        query_lens=tuple(query_lens),
        num_decodes=num_decodes,
        num_prefills=num_reqs - num_decodes,
        num_decode_tokens=num_decode_tokens,
        num_prefill_tokens=kept_prefill,
        full_prefill_tokens=full_prefill,
        dropped_prefill_rows=full_prefill - kept_prefill,
    )


def _layer_id(key: str) -> int | None:
    match = _LAYER_ID.search(key)
    if match is None:
        return None
    return int(match.group(1))


def _kind(metadata: Any) -> str:
    if getattr(metadata, "query_start_loc_cpu", None) is not None and hasattr(
        metadata, "num_prefill_tokens"
    ):
        return "swa"
    if (
        hasattr(metadata, "prefill")
        and hasattr(metadata, "num_prefill_tokens")
        and hasattr(metadata, "num_decode_tokens")
    ):
        return "indexer"
    return "unknown"


def _gather(tensor: torch.Tensor, index: torch.Tensor) -> torch.Tensor | None:
    if tensor.ndim < 1 or tensor.shape[0] <= int(index[-1].item()):
        return None
    if index.device != tensor.device:
        index = index.to(device=tensor.device)
    return tensor.index_select(0, index)


def _cpu_index(
    indices: tuple[int, ...], *, dtype: torch.dtype = torch.long
) -> torch.Tensor:
    return torch.tensor(indices, dtype=dtype)


def _prepare_swa(metadata: Any, plan: PrefillKeepPlan) -> dict[str, Any] | str:
    if int(metadata.num_prefills) != plan.num_prefills:
        return "SWA metadata request split does not match the keep plan"
    if int(metadata.num_prefill_tokens) != plan.full_prefill_tokens:
        return "SWA metadata prefill width does not match query_start_loc"
    keep = _cpu_index(plan.keep_indices)
    local = _cpu_index(plan.prefill_local)
    updates: dict[str, Any] = {
        "num_prefill_tokens": plan.num_prefill_tokens,
        "num_decode_tokens": plan.num_decode_tokens,
        "num_decodes": plan.num_decodes,
        "num_prefills": plan.num_prefills,
    }
    for name in ("slot_mapping", "is_valid_token", "token_to_req_indices"):
        tensor = getattr(metadata, name, None)
        if tensor is None:
            return f"SWA metadata is missing {name}"
        gathered = _gather(tensor, keep)
        if gathered is None:
            return f"SWA {name} does not cover the kept rows"
        updates[name] = gathered
    for name in ("prefill_swa_indices", "prefill_swa_lens"):
        tensor = getattr(metadata, name, None)
        if tensor is None:
            return f"SWA metadata is missing {name}"
        gathered = _gather(tensor, local)
        if gathered is None:
            return f"SWA {name} does not cover the kept prefill rows"
        updates[name] = gathered
    for name in ("prefill_left_visible", "prefill_right_visible"):
        tensor = getattr(metadata, name, None)
        if tensor is None:
            continue
        if tensor.shape[0] > int(keep[-1].item()):
            gathered = _gather(tensor, keep)
        else:
            gathered = _gather(tensor, local)
        if gathered is None:
            return f"SWA {name} does not cover the kept rows"
        updates[name] = gathered
    query_lens = _cpu_index(plan.query_lens, dtype=metadata.query_start_loc_cpu.dtype)
    cu_cpu = torch.empty(query_lens.shape[0] + 1, dtype=query_lens.dtype)
    cu_cpu[0] = 0
    torch.cumsum(query_lens, dim=0, out=cu_cpu[1:])
    updates["query_start_loc_cpu"] = cu_cpu
    if getattr(metadata, "query_start_loc", None) is None:
        return "SWA metadata is missing query_start_loc"
    updates["query_start_loc"] = cu_cpu.to(
        device=metadata.query_start_loc.device,
        dtype=metadata.query_start_loc.dtype,
    )
    query_cpu = getattr(metadata, "prefill_query_lens_cpu", None)
    seq_cpu = getattr(metadata, "prefill_seq_lens_cpu", None)
    if query_cpu is not None:
        new_query = _cpu_index(plan.prefill_query_lens, dtype=query_cpu.dtype)
        updates["prefill_query_lens_cpu"] = new_query
        if seq_cpu is None or seq_cpu.shape[0] != new_query.shape[0]:
            return "SWA prefill seq lens do not match the prefill requests"
        prefix = seq_cpu.to(dtype=torch.int64, device="cpu") - new_query.to(torch.int64)
        window = int(getattr(metadata, "prefill_window_size", 0) or 0)
        if window > 0:
            extra = torch.clamp(prefix, min=0, max=window - 1)
        else:
            extra = torch.zeros_like(prefix)
        gather_cpu = (new_query.to(torch.int64) + extra).to(dtype=query_cpu.dtype)
        updates["prefill_query_lens_cpu"] = new_query
        gather = getattr(metadata, "prefill_gather_lens", None)
        if gather is not None:
            updates["prefill_gather_lens"] = gather_cpu.to(
                device=gather.device, dtype=gather.dtype
            )
    cache = getattr(metadata, "flashinfer_sparse_index_cache", None)
    if isinstance(cache, dict):
        # Layer 20 may have cached full-batch sparse indices. Later layers
        # must rebuild them from the compacted top-k rows.
        updates["flashinfer_sparse_index_cache"] = {}
    return updates


def _prepare_indexer(metadata: Any, plan: PrefillKeepPlan) -> dict[str, Any] | str:
    if getattr(metadata, "sparse_prefill", None) is not None:
        return (
            "sparse-logits indexer metadata is not retargeted by CED prefill "
            "(SM100 sparse MQA path)"
        )
    prefill = metadata.prefill
    if prefill is None or not getattr(prefill, "chunks", None):
        return "indexer prefill chunks are missing"
    keep_set = set(plan.keep_indices)
    grouped: dict[int, list[int]] = {}
    covered: set[int] = set()
    for chunk_id, chunk in enumerate(prefill.chunks):
        if getattr(chunk, "pcp_deinterleave_idx", None) is not None:
            return "context-parallel indexer chunks are not supported"
        n_rows = int(chunk.cu_seqlen_ks.shape[0])
        rows: list[int] = []
        for row in range(n_rows):
            absolute = int(chunk.token_start) + row
            covered.add(absolute)
            if absolute in keep_set:
                rows.append(row)
        if rows:
            grouped[chunk_id] = rows
    # Decode rows are not in the prefill chunks. Prefill keeps must be.
    missing = [
        index
        for index in plan.keep_indices
        if index >= plan.num_decode_tokens and index not in covered
    ]
    if missing:
        return "indexer prefill chunks do not cover the kept prefill rows"
    new_chunks = []
    cursor = plan.num_decode_tokens
    for chunk_id, chunk in enumerate(prefill.chunks):
        rows = grouped.get(chunk_id)
        if not rows:
            continue
        row_index = torch.tensor(rows, dtype=torch.long)
        try:
            new_chunk = replace(
                chunk,
                cu_seqlen_ks=_gather(chunk.cu_seqlen_ks, row_index),
                cu_seqlen_ke=_gather(chunk.cu_seqlen_ke, row_index),
                token_to_seq=_gather(chunk.token_to_seq, row_index),
                token_start=cursor,
                token_end=cursor + len(rows),
            )
        except (AttributeError, TypeError) as exc:
            return f"indexer chunk could not be retargeted ({exc})"
        if (
            new_chunk.cu_seqlen_ks is None
            or new_chunk.cu_seqlen_ke is None
            or new_chunk.token_to_seq is None
        ):
            return "indexer chunk rows do not cover the kept queries"
        new_chunks.append(new_chunk)
        cursor += len(rows)
    if cursor - plan.num_decode_tokens != plan.num_prefill_tokens:
        return "indexer kept rows do not match the compacted prefill width"
    slot = getattr(metadata, "slot_mapping", None)
    try:
        new_prefill = replace(prefill, chunks=new_chunks)
    except TypeError as exc:
        return f"indexer prefill metadata could not be retargeted ({exc})"
    updates: dict[str, Any] = {
        "num_prefill_tokens": plan.num_prefill_tokens,
        "prefill": new_prefill,
    }
    if slot is not None:
        gathered = _gather(slot, _cpu_index(plan.keep_indices))
        if gathered is None:
            return "indexer slot_mapping does not cover the kept rows"
        updates["slot_mapping"] = gathered
    return updates


def _select_metadata(
    attn_metadata: dict[str, Any], config: CedPrefillConfig
) -> tuple[list[tuple[Any, str]], str | None]:
    """Metadata objects that post-boundary layers will read, each once."""
    selected: list[tuple[Any, str]] = []
    seen: set[int] = set()
    found_swa = False
    found_indexer = False
    for key, metadata in attn_metadata.items():
        if metadata is None:
            continue
        layer_id = _layer_id(key)
        if layer_id is None:
            continue
        is_later = layer_id > config.boundary_layer
        is_boundary_indexer = layer_id == config.boundary_layer and "indexer" in key
        if not is_later and not is_boundary_indexer:
            continue
        if id(metadata) in seen:
            if is_later and _kind(metadata) == "swa":
                found_swa = True
            if is_boundary_indexer and _kind(metadata) == "indexer":
                found_indexer = True
            continue
        kind = _kind(metadata)
        if kind == "unknown":
            return [], (
                f"unrecognized attention metadata {type(metadata).__name__} at {key}"
            )
        seen.add(id(metadata))
        selected.append((metadata, kind))
        found_swa = found_swa or (is_later and kind == "swa")
        found_indexer = found_indexer or (is_boundary_indexer and kind == "indexer")
    if not found_swa:
        return [], "no post-boundary SWA metadata to retarget"
    if config.reindex_after_boundary and not found_indexer:
        return [], (
            "reindex layers after the boundary need the boundary indexer "
            "metadata, and it was not found"
        )
    return selected, None


def _assign(metadata: Any, updates: dict[str, Any]) -> None:
    for name, value in updates.items():
        setattr(metadata, name, value)


def _stage_buffer(
    buffer: torch.Tensor | None, keep: torch.Tensor, full_tokens: int
) -> torch.Tensor | None | str:
    if buffer is None:
        return None
    if buffer.ndim < 1 or buffer.shape[0] < full_tokens:
        return "token buffer is shorter than the batch"
    gathered = _gather(buffer, keep)
    if gathered is None:
        return "token buffer does not cover the kept rows"
    return gathered


@dataclass
class CedKeep:
    keep_index: torch.Tensor
    full_tokens: int
    compact_tokens: int
    boundary_layer: int
    tail_rows: int
    full_prefill_tokens: int
    kept_prefill_tokens: int


@dataclass
class CedCompaction:
    keep: CedKeep
    hidden_states: torch.Tensor
    residual: torch.Tensor | None
    post_mix: torch.Tensor | None
    res_mix: torch.Tensor | None
    pre_mix: torch.Tensor | None
    positions: torch.Tensor
    input_ids: torch.Tensor | None


@dataclass(frozen=True)
class CedRefusal:
    reason: str


def apply_ced_compaction(
    config: CedPrefillConfig,
    attn_metadata: dict[str, Any],
    *,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    post_mix: torch.Tensor | None,
    res_mix: torch.Tensor | None,
    pre_mix: torch.Tensor | None,
    positions: torch.Tensor,
    input_ids: torch.Tensor | None,
    topk_indices_buffer: torch.Tensor | None,
    candidate_block_buffer: torch.Tensor | None,
) -> CedCompaction | CedRefusal | None:
    """Compact post-boundary state. ``None`` means nothing to drop.

    Metadata is replaced only after every retarget succeeds, so a refusal
    leaves the full-prefill tensors and metadata as they were.
    """
    swa_for_plan = None
    for key, metadata in attn_metadata.items():
        layer_id = _layer_id(key) if isinstance(key, str) else None
        if (
            layer_id is not None
            and layer_id > config.boundary_layer
            and metadata is not None
            and _kind(metadata) == "swa"
        ):
            swa_for_plan = metadata
            break
    if swa_for_plan is None:
        return CedRefusal("no post-boundary SWA metadata to retarget")
    qsl_cpu = swa_for_plan.query_start_loc_cpu
    planned = plan_prefill_keep(
        [int(value) for value in qsl_cpu.tolist()],
        int(swa_for_plan.num_decodes),
        int(swa_for_plan.num_decode_tokens),
        config.tail_rows,
    )
    if isinstance(planned, str):
        return CedRefusal(planned)
    if planned.dropped_prefill_rows == 0:
        return None
    selected, error = _select_metadata(attn_metadata, config)
    if error is not None:
        return CedRefusal(error)
    prepared: list[tuple[Any, dict[str, Any]]] = []
    for metadata, kind in selected:
        if kind == "swa":
            updates = _prepare_swa(metadata, planned)
        else:
            updates = _prepare_indexer(metadata, planned)
        if isinstance(updates, str):
            return CedRefusal(updates)
        prepared.append((metadata, updates))

    full_tokens = int(hidden_states.shape[0])
    if not planned.keep_indices or planned.keep_indices[-1] >= full_tokens:
        return CedRefusal("kept rows fall outside the hidden-state batch")
    if positions.ndim != 1 or int(positions.shape[0]) != full_tokens:
        return CedRefusal("positions are not a 1D token axis matching hidden states")
    for name, tensor in (
        ("residual", residual),
        ("post_mix", post_mix),
        ("res_mix", res_mix),
        ("pre_mix", pre_mix),
        ("input_ids", input_ids),
    ):
        if tensor is not None and (
            tensor.ndim < 1 or int(tensor.shape[0]) != full_tokens
        ):
            return CedRefusal(f"{name} token axis does not match hidden states")
    keep = _cpu_index(planned.keep_indices).to(device=hidden_states.device)
    if topk_indices_buffer is None:
        return CedRefusal("topk index buffer is missing")
    staged_topk = _stage_buffer(topk_indices_buffer, keep, full_tokens)
    staged_candidates = _stage_buffer(candidate_block_buffer, keep, full_tokens)
    if isinstance(staged_topk, str):
        return CedRefusal(staged_topk)
    if isinstance(staged_candidates, str):
        return CedRefusal(staged_candidates)
    assert staged_topk is not None

    def _take(tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None:
            return None
        gathered = _gather(tensor, keep)
        assert gathered is not None
        return gathered

    for metadata, updates in prepared:
        _assign(metadata, updates)
    topk_indices_buffer[: staged_topk.shape[0]].copy_(staged_topk)
    if candidate_block_buffer is not None and staged_candidates is not None:
        candidate_block_buffer[: staged_candidates.shape[0]].copy_(staged_candidates)
    _log_once(
        "info",
        ced_compact_message(
            config, planned.full_prefill_tokens, planned.num_prefill_tokens
        ),
    )
    return CedCompaction(
        keep=CedKeep(
            keep_index=keep,
            full_tokens=full_tokens,
            compact_tokens=int(keep.shape[0]),
            boundary_layer=config.boundary_layer,
            tail_rows=config.tail_rows,
            full_prefill_tokens=planned.full_prefill_tokens,
            kept_prefill_tokens=planned.num_prefill_tokens,
        ),
        hidden_states=_take(hidden_states),
        residual=_take(residual),
        post_mix=_take(post_mix),
        res_mix=_take(res_mix),
        pre_mix=_take(pre_mix),
        positions=_take(positions),
        input_ids=_take(input_ids),
    )


def restore_token_rows(compact: torch.Tensor, keep: CedKeep) -> torch.Tensor:
    """Scatter compacted rows back to their original batch positions.

    Dropped prefill rows are zeros. Sampling reads the last token of each
    request, which is always inside the kept tail (or a kept decode row).
    """
    if compact.shape[0] == keep.full_tokens:
        return compact
    restored = compact.new_zeros((keep.full_tokens, *compact.shape[1:]))
    index = keep.keep_index
    if index.device != restored.device:
        index = index.to(device=restored.device)
    restored.index_copy_(0, index, compact)
    return restored


def restore_ced_outputs(
    keep: CedKeep,
    hidden_states: torch.Tensor,
    pre_mix: torch.Tensor | None,
    aux_hidden_by_layer: dict[int, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor | None, dict[int, torch.Tensor]]:
    """Put CED outputs back on the runner's original token axis."""
    hidden_states = restore_token_rows(hidden_states, keep)
    if isinstance(pre_mix, torch.Tensor) and pre_mix.shape[0] == keep.compact_tokens:
        pre_mix = restore_token_rows(pre_mix, keep)
    for layer_id, value in list(aux_hidden_by_layer.items()):
        if isinstance(value, torch.Tensor) and value.shape[0] == keep.compact_tokens:
            aux_hidden_by_layer[layer_id] = restore_token_rows(value, keep)
    return hidden_states, pre_mix, aux_hidden_by_layer


def ced_runtime_block_reason(
    *,
    metadata_kind: str,
    capturing: bool,
    cudagraph_mode_value: int,
) -> str | None:
    """Why an otherwise compactable step must stay on full prefill."""
    if metadata_kind == "missing":
        return "attention metadata is missing"
    if metadata_kind == "list":
        return "ubatch/DBO attention metadata is not supported"
    if capturing:
        return "CUDA graph capture keeps full prefill"
    if cudagraph_mode_value != 0:
        return f"cudagraph runtime mode {cudagraph_mode_value} keeps full prefill"
    return None


def _cuda_capturing() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except RuntimeError:
        return False


def _cudagraph_mode_value(mode: Any) -> int:
    value = mode.value
    if isinstance(value, tuple):
        return -1
    return int(value)


def _swa_for_plan(
    config: CedPrefillConfig, attn_metadata: dict[str, Any]
) -> Any | None:
    for key, metadata in attn_metadata.items():
        layer_id = _layer_id(key) if isinstance(key, str) else None
        if (
            layer_id is not None
            and layer_id > config.boundary_layer
            and metadata is not None
            and _kind(metadata) == "swa"
        ):
            return metadata
    return None


def compact_after_boundary(
    config: CedPrefillConfig,
    *,
    hidden_states: Any,
    residual: torch.Tensor | None,
    post_mix: torch.Tensor | None,
    res_mix: torch.Tensor | None,
    pre_mix: torch.Tensor | None,
    positions: torch.Tensor,
    input_ids: torch.Tensor | None,
    topk_indices_buffer: torch.Tensor | None,
    candidate_block_buffer: torch.Tensor | None,
    mega_gate_metadata: Any,
) -> CedCompaction | None:
    """Hook used by the decoder loop. Refusals are logged and keep full prefill.

    Decode steps and short prefill chunks drop nothing, so they stay on the
    current path with no log line. A chunk that would compact is refused,
    with a log line, when the step cannot change token width safely.
    """
    if not isinstance(hidden_states, torch.Tensor):
        _log_once(
            "warning",
            "DSV41 CED prefill refused: boundary output is not a dense tensor",
        )
        return None
    if mega_gate_metadata is not None:
        _log_once(
            "warning",
            "DSV41 CED prefill refused: MegaMoE routing metadata is bound "
            "to the full chunk",
        )
        return None
    if not is_forward_context_available():
        _log_once(
            "warning",
            "DSV41 CED prefill refused: attention metadata is missing",
        )
        return None
    ctx = get_forward_context()
    metadata = ctx.attn_metadata
    if isinstance(metadata, dict):
        kind = "dict"
    elif isinstance(metadata, list):
        kind = "list"
    else:
        kind = "missing"
    if kind != "dict":
        block = ced_runtime_block_reason(
            metadata_kind=kind,
            capturing=False,
            cudagraph_mode_value=0,
        )
        if block is not None:
            _log_once("warning", f"DSV41 CED prefill refused: {block}")
        return None

    swa = _swa_for_plan(config, metadata)
    if swa is None:
        _log_once(
            "warning",
            "DSV41 CED prefill refused: no post-boundary SWA metadata to retarget",
        )
        return None
    planned = plan_prefill_keep(
        [int(value) for value in swa.query_start_loc_cpu.tolist()],
        int(swa.num_decodes),
        int(swa.num_decode_tokens),
        config.tail_rows,
    )
    if isinstance(planned, str):
        _log_once("warning", f"DSV41 CED prefill refused: {planned}")
        return None
    if planned.dropped_prefill_rows == 0:
        return None
    block = ced_runtime_block_reason(
        metadata_kind="dict",
        capturing=_cuda_capturing(),
        cudagraph_mode_value=_cudagraph_mode_value(ctx.cudagraph_runtime_mode),
    )
    if block is not None:
        _log_once("warning", f"DSV41 CED prefill refused: {block}")
        return None
    result = apply_ced_compaction(
        config,
        metadata,
        hidden_states=hidden_states,
        residual=residual,
        post_mix=post_mix,
        res_mix=res_mix,
        pre_mix=pre_mix,
        positions=positions,
        input_ids=input_ids,
        topk_indices_buffer=topk_indices_buffer,
        candidate_block_buffer=candidate_block_buffer,
    )
    if isinstance(result, CedRefusal):
        _log_once("warning", f"DSV41 CED prefill refused: {result.reason}")
        return None
    return result
