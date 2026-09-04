# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM12x (GB10 / SM120/SM121) small-batch alignment helpers."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal

import torch

from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

# Size-16 PIECEWISE dummies succeed on GB10; size 8 then IMA'd in MHC / b12x
# MoE (same compiled kernels, dynamic token count). Pad small batches up.
SM12X_SAFE_MIN_TOKENS = 16

# Per-request FlashInfer decode-form q_len. FlashInfer SM120 keeps
# num_tokens <= 64 on decode kernels.
# Decode dummies: 4 real tokens as [1, 4] survived capture (size 16 = 4×4).
# Do not pad those to [1, 6] by repeating the last row — that IMA'd
# (async, reported at DSpark capture start).
# Prefill: 2 and 4 real first-prefills IMA'd at [1, 4]; pad to 6.
# 2/3 must not snap to 4 (mixed-warmup 2→4 IMA'd).
SM12X_SAFE_DECODE_Q_LENS = (1, 4, 6, 8, 16, 24, 32, 36)
SM12X_SAFE_PREFILL_DECODE_Q_LENS = (1, 6, 8, 16, 24, 32, 36)
SM12X_UNSAFE_PER_REQUEST_Q_LENS = (2, 3)
# DSpark FULL dummies. k=5 aligns q_len 5→6 (6/12/18/24/36). k=3 is
# native q_len=4 (4/8/12/16/24). Keep 12 and 18 so a 2- or 3-req
# DSpark batch does not pad to 4/6 slots. Dropping them forced 12→24
# and crashed: is_prefilling length 2 vs query_lens 4. Extra 4/8/16
# are ignored for q_len=6 (not divisible).
SM12X_DSPARK_SAFE_CAPTURE_TOKENS = (4, 6, 8, 12, 16, 18, 24, 36)


def sm12x_align_tokens(num_tokens: int, min_tokens: int = SM12X_SAFE_MIN_TOKENS) -> int:
    """Return a safe token count for SM12x kernels. Unchanged off SM12x."""
    if (
        num_tokens > 0
        and num_tokens < min_tokens
        and current_platform.is_device_capability_family(120)
    ):
        return min_tokens
    return num_tokens


def sm12x_mixed_warmup_decode_prompt_len() -> int:
    """Prefill length used to seed mixed-warmup's decode request.

    Keep the historical 2-token seed. SM12x pads that ``q_len=2`` prefill
    to one decode-form launch instead of splitting it or growing the seed
    into the untested ``num_tokens > 64`` prefill orchestrator.
    """
    return 2


def sm12x_mixed_warmup_prefill_len(requested_prefill: int) -> int:
    """Scheduled mixed-step prefill tokens.

    On SM12x keep ``q_len=2`` so FlashInfer pads one decode launch.
    Growing the scheduled prefill to 16 makes a 16-token *real* prefill,
    which is not the pad path and already IMA'd at 4 real tokens.
    """
    if requested_prefill <= 0:
        return requested_prefill
    if current_platform.is_device_capability_family(120):
        return sm12x_mixed_warmup_decode_prompt_len()
    return requested_prefill


def sm12x_dspark_capture_sizes(
    capture_sizes: list[int] | None,
    decode_query_len: int,
) -> list[int]:
    """Restrict SM12x DSpark FULL capture to proven token counts.

    Off SM12x, or when no safe size is divisible by ``decode_query_len``,
    return the original list.
    """
    sizes = list(capture_sizes or [])
    if not sizes or not current_platform.is_device_capability_family(120):
        return sizes
    max_cg = max(sizes)
    allow = [
        n
        for n in SM12X_DSPARK_SAFE_CAPTURE_TOKENS
        if n <= max_cg and n % decode_query_len == 0
    ]
    return allow or sizes


def sm12x_allow_full_decode_capture(num_tokens: int, decode_query_len: int) -> bool:
    """Whether a rounded FULL-decode dummy may be captured on SM12x.

    DSpark ``decode_query_len=6`` rounds capture sizes to multiples of
    6. Keep the proven set ``6, 12, 18, 24, 36``. Off SM12x, or when
    ``decode_query_len<=1``, keep every size.
    """
    if decode_query_len <= 1:
        return True
    if not current_platform.is_device_capability_family(120):
        return True
    return (
        num_tokens in SM12X_DSPARK_SAFE_CAPTURE_TOKENS
        and num_tokens % decode_query_len == 0
    )


def sm12x_flashinfer_decode_tune_sizes(
    capture_sizes: list[int] | None,
    decode_query_len: int,
) -> list[int]:
    """DSpark FULL token counts FlashInfer should autotune on SM12x."""
    if decode_query_len <= 1:
        return []
    if not current_platform.is_device_capability_family(120):
        return []
    return sm12x_dspark_capture_sizes(capture_sizes, decode_query_len)


def sm12x_flashinfer_autotune_query_lens(decode_query_len: int) -> list[int]:
    """FlashInfer dummy-run q_lens for SM12x DSpark.

    The target runner's ``decode_query_len`` is ``k + 1``. DSpark
    ``sample_from_anchor`` uses ``q_len = k``. Vision ``k=3`` therefore
    launches ``q=3`` which snaps to 6, while the runner reports 4.
    Autotuning only 4 misses the hot ``q=6`` decode (tactic=-1).
    """
    if decode_query_len <= 0:
        return []
    if decode_query_len <= 1 or not current_platform.is_device_capability_family(120):
        return [decode_query_len]
    qs = {
        decode_query_len,
        sm12x_align_decode_q_len(decode_query_len),
        sm12x_align_decode_q_len(decode_query_len - 1),
    }
    return sorted(q for q in qs if q > 1)


def sm12x_kernel_warmup_prefill_len(decode_query_len: int) -> int:
    """V2 kernel-warmup prefill length.

    Stock uses ``decode_query_len + 1`` so the step is not classified
    as uniform decode. On SM12x that is 7 for DSpark, which FlashInfer
    pads to 8. Schedule 8 directly (still ``> decode_query_len``).
    """
    prompt_len = decode_query_len + 1
    if not current_platform.is_device_capability_family(120):
        return prompt_len
    aligned = sm12x_align_prefill_q_len(prompt_len)
    return aligned if aligned > decode_query_len else prompt_len


def sm12x_align_decode_q_len(q_len: int) -> int:
    """Snap a FlashInfer *decode dummy* q_len to a GB10-safe width.

    Keep 4 (CUDA-graph dummy). Snap 2/3 to 6, not 4. Unchanged off SM12x.
    """
    if q_len <= 0 or not current_platform.is_device_capability_family(120):
        return q_len
    for safe in SM12X_SAFE_DECODE_Q_LENS:
        if safe == 4 and q_len < 4:
            continue
        if q_len <= safe:
            return safe
    return q_len


def sm12x_treat_short_extends_as_decodes() -> bool:
    """Whether short first prefills may ride the decode split.

    DSpark sets ``decode_threshold = k+1 = 6``. A 2-token mixed-warmup
    seed then becomes an all-decode batch and FlashInfer launches
    ``[1, 4]`` (decode align), the width that IMA'd. On SM12x keep those
    requests on the prefill path so ``q_len=2`` pads to one ``[1, 6]``.
    """
    return not current_platform.is_device_capability_family(120)


def sm12x_disable_attn_aux_streams() -> bool:
    """Do not overlap indexer/compressor GEMMs on SM12x aux streams.

    ``maybe_execute_in_parallel`` already drops aux during breakable
    CUDA-graph capture. Mixed warmup is eager after capture and is the
    first time those aux streams run; Marlin already IMA'd on aux.
    Shared experts disable their aux stream on SM12x for the same reason.
    """
    return current_platform.is_device_capability_family(120)


def sm12x_disable_eager_scratch_pool() -> bool:
    """Skip v0.28.0 eager-scratch ``_out`` insert on SM12x.

    later-main never allocated this pool and used allocating
    ``fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert``. Host-mounted
    later-main ``_C_stable_libtorch`` does not register ``_out``, which
    AttributeError'd mixed warmup after the 2→6 prefill pad.
    """
    return current_platform.is_device_capability_family(120)


def reject_sm12x_unsafe_decode_query(query: torch.Tensor) -> None:
    """Refuse SM12x per-request FlashInfer shapes that IMA'd on GB10.

    ``[1, 2]`` / ``[1, 3]`` IMA'd. ``[1, 4]`` is a valid decode dummy
    (size-16 capture); do not reject it.
    """
    if (
        query.ndim == 4
        and query.shape[0] == 1
        and int(query.shape[1]) in SM12X_UNSAFE_PER_REQUEST_Q_LENS
        and current_platform.is_device_capability_family(120)
    ):
        raise RuntimeError(
            "SM12x FlashInfer refused per-request query shape "
            f"{tuple(query.shape)}; pad q_len to 6"
        )


def sm12x_align_prefill_q_len(q_len: int) -> int:
    """Pad a short SM12x prefill to one decode-form launch.

    FlashInfer SM120 routes ``num_tokens <= 64`` to decode kernels. Do not
    split ``q_len=2`` into two ``q_len=1`` launches. Skip width 4: a
    4-token real seed IMA'd after capture.
    """
    if q_len <= 0 or not current_platform.is_device_capability_family(120):
        return q_len
    for safe in SM12X_SAFE_PREFILL_DECODE_Q_LENS:
        if q_len <= safe:
            return safe
    return q_len


def sm12x_is_capturing() -> bool:
    """Host query: current CUDA stream is capturing a graph."""
    is_capturing = getattr(torch.cuda, "is_current_stream_capturing", None)
    return bool(is_capturing and is_capturing())


def sm12x_should_fill_prefill_slots(
    num_tokens: int, num_decode_tokens: int | None = 0
) -> bool:
    """Whether to write extra first-prefill SWA KV slots for a [1, 6] pad.

    Eager all-prefill only. The compressor must not run this on a
    tokens=4 decode dummy during PIECEWISE capture; ``bool(torch.all)``
    host-syncs and aborts the graph. C4A writes stay at the real token
    count — see ``sm12x_should_fill_compressed_prefill_slots``.
    """
    if num_tokens not in (2, 4, 5):
        return False
    if num_decode_tokens:
        return False
    if sm12x_align_prefill_q_len(num_tokens) == num_tokens:
        return False
    return not sm12x_is_capturing()


def sm12x_should_fill_compressed_prefill_slots(
    num_tokens: int, num_decode_tokens: int | None = 0
) -> bool:
    """Do not pad C4A writes to the SWA [1, 6] width.

    A 6-token C4A Triton write plus unused ``compute_global_topk`` on a
    2-token indexer IMA'd after FlashInfer already dropped extra cache.
    Keep SWA insert. C4A stays off only when ``is_prefill`` and
    ``launch_len != q_len``; long prefills and DSpark decode ``5→6``
    keep C4A.
    """
    return False


# FlashInfer SM120 DSV4 dual-cache prefill (C4A extra KV) is only
# instantiated for SWA topk=128. Vision-Exp allocates
# window + vision_max_n_token (128+384=512) even on text dummy rows.
FLASHINFER_SM120_DSV4_DUAL_PREFILL_SWA_TOPK = 128


def sm12x_align_flashinfer_dual_prefill(
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    extra_kv: torch.Tensor | None,
    extra_indices: torch.Tensor | None,
    extra_lens: torch.Tensor | None,
    *,
    has_image: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Make a >64-token FlashInfer prefill hit a compiled cubin.

    Dual-cache prefill accepts SWA topk=128 only. Vision-Exp widens the
    allocated SWA row to ``window + vision_max_n_token``. Text dummy
    rows still have that 512-wide buffer, which FlashInfer reads as
    ``topk=512`` and rejects (``Unsupported sparse-MLA prefill
    configuration``). Slice text rows to 128 and keep C4A. Image rows
    need the widened SWA, so drop C4A and use the single-cache 512
    cubin. 0731 (width 128) is unchanged.
    """
    if extra_kv is None or not current_platform.is_device_capability_family(120):
        return swa_indices, swa_lens, extra_kv, extra_indices, extra_lens
    topk = int(swa_indices.shape[-1])
    want = FLASHINFER_SM120_DSV4_DUAL_PREFILL_SWA_TOPK
    if topk == want:
        return swa_indices, swa_lens, extra_kv, extra_indices, extra_lens
    if has_image:
        return swa_indices, swa_lens, None, None, None
    return (
        swa_indices[..., :want],
        swa_lens.clamp(max=want),
        extra_kv,
        extra_indices,
        extra_lens,
    )


def sm12x_skip_padded_prefill_c4a(query_lens: list[int]) -> bool:
    """True when every prefill span will launch SWA-only [1, 6].

    C4A is dropped iff ``is_prefill and launch_len != q_len``. Do not
    restore C4A on that padded first-prefill path (extra-sparse and
    ``compute_global_topk`` IMA'd). Long prefills and DSpark decode
    ``5→6`` keep C4A.
    """
    if not query_lens or not current_platform.is_device_capability_family(120):
        return False
    for q_len in query_lens:
        if q_len <= 0:
            continue
        if q_len > 64 or sm12x_align_prefill_q_len(q_len) == q_len:
            return False
    return True


def sm12x_replace_negative_indices(
    indices: torch.Tensor | None,
    *,
    mode: Literal["repeat_last", "fill0"] = "repeat_last",
) -> torch.Tensor | None:
    """Replace ``-1`` gather sentinels for SM12x kernels.

    FlashInfer SM120 decode cubins gather the full top-k width first.
    A 2-token first-prefill has ``swa_len`` 1–2, so most of a 128-wide
    SWA row is ``-1``. Those addresses IMA. ``VLLM_MOE_SKIP_PADDING``
    writes the same sentinel on alignment rows; b12x SiLU gathers it.

    ``repeat_last`` (SWA): last valid index per row; all-``-1`` rows
    become 0. ``swa_topk_lens`` still masks softmax.
    ``fill0`` (MoE): every ``-1`` becomes expert 0. Callers zero those
    weights. Capture-safe: no ``.item()`` / ``bool(tensor)``.
    """
    if indices is None or indices.numel() == 0:
        return indices
    if not current_platform.is_device_capability_family(120):
        return indices
    if mode == "fill0":
        return indices.clamp(min=0)
    width = indices.shape[-1]
    if width == 0:
        return indices
    rows = indices.reshape(-1, width)
    valid = rows >= 0
    pos = torch.arange(width, device=rows.device)
    last_pos = torch.where(valid, pos, pos.new_zeros(())).amax(dim=-1)
    last_slot = rows.gather(-1, last_pos.unsqueeze(-1)).squeeze(-1)
    last_slot = torch.where(valid.any(dim=-1), last_slot, last_slot.new_zeros(()))
    filled = torch.where(valid, rows, last_slot.unsqueeze(-1))
    return filled.reshape(indices.shape)


def sm12x_replace_swa_index_sentinels(
    indices: torch.Tensor | None,
) -> torch.Tensor | None:
    """Replace SWA ``-1`` tails with the last valid slot per row."""
    return sm12x_replace_negative_indices(indices, mode="repeat_last")


def sm12x_extend_prefill_slots(
    slot_mapping: torch.Tensor, target: int, block_size: int
) -> torch.Tensor:
    """Grow first-prefill slots to the decode-form pad width.

    Extra query rows repeat the last real token. Extra KV slots reuse
    that last slot so chunked prefill cannot occupy a later token's
    address. ``block_size`` is unused but kept for callers.

    Capture-safe: no ``.item()`` / ``bool(tensor)`` host sync.
    """
    del block_size
    n = slot_mapping.shape[0]
    if n <= 0 or n >= target:
        return slot_mapping
    extra = target - n
    reuse = slot_mapping[-1].expand(extra)
    return torch.cat((slot_mapping, reuse), dim=0)


def sm12x_pad_prefill_token_rows(t: torch.Tensor, target: int) -> torch.Tensor:
    """Repeat the last real row so KV insert matches a [1, 6] launch."""
    n = t.shape[0]
    if n <= 0 or n >= target:
        return t
    extra = target - n
    return torch.cat((t, t[-1:].expand(extra, *t.shape[1:])), dim=0)


def pad_token_rows(t: torch.Tensor, target: int) -> torch.Tensor:
    """Pad ``t`` along dim 0 with zeros up to ``target`` rows."""
    n = t.shape[0]
    if n >= target:
        return t
    return torch.cat((t, t.new_zeros((target - n, *t.shape[1:]))), dim=0)


def sm12x_pad_token_rows(
    t: torch.Tensor,
    min_tokens: int = SM12X_SAFE_MIN_TOKENS,
    *,
    what: str | None = None,
) -> tuple[torch.Tensor, int]:
    """Pad ``t`` to ``sm12x_align_tokens`` and return ``(padded, orig_rows)``."""
    orig = t.shape[0]
    target = sm12x_align_tokens(orig, min_tokens)
    if target == orig:
        return t, orig
    if what is not None:
        logger.info_once(
            "SM12x %s: padding token dim %d -> %d to avoid small-batch IMA",
            what,
            orig,
            target,
        )
    return sm12x_pad_prefill_token_rows(t, target), orig


def sm12x_replace_moe_topk_sentinels(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace router ``-1`` pad experts before the b12x SiLU kernel.

    ``VLLM_MOE_SKIP_PADDING`` writes ``topk_ids=-1`` on SM12x alignment
    rows. FlashInfer SWA can mask those; b12x ``MoEDynamicKernelSilu``
    gathers the expert id. Capture dummies mark every row padding, so
    all ids are ``-1`` and the kernel no-ops. Mixed-warmup has 2 real
    rows plus 14 ``-1``s. Map sentinels to expert 0 and zero their
    weights. Capture-safe: no ``.item()`` / ``bool(tensor)``.
    """
    if not current_platform.is_device_capability_family(120):
        return topk_ids, topk_weights
    invalid = topk_ids < 0
    filled = sm12x_replace_negative_indices(topk_ids, mode="fill0")
    assert filled is not None
    return filled, topk_weights.masked_fill(invalid, 0)


def extend_padding_mask(
    is_padding: torch.Tensor | None, num_tokens: int
) -> torch.Tensor | None:
    """Slice or grow ``is_padding`` so it has exactly ``num_tokens`` rows.

    Extra rows from SM12x kernel alignment are marked True (not real tokens).
    """
    if is_padding is None:
        return None
    n = is_padding.shape[0]
    if n >= num_tokens:
        return is_padding[:num_tokens]
    extra = num_tokens - n
    return torch.cat((is_padding, is_padding.new_ones((extra,), dtype=torch.bool)))


@contextmanager
def sm12x_align_is_padding(num_tokens: int) -> Iterator[None]:
    """Temporarily grow ForwardContext.is_padding to ``num_tokens``."""
    if not is_forward_context_available():
        yield
        return
    ctx = get_forward_context()
    orig = ctx.is_padding
    if orig is not None and orig.shape[0] != num_tokens:
        ctx.is_padding = extend_padding_mask(orig, num_tokens)
    try:
        yield
    finally:
        ctx.is_padding = orig
