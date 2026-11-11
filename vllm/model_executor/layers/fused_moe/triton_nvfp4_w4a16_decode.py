# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in small-M NVFP4 W4A16 decode MoE on top of Marlin.

``DSV41_EXP_FUSED_W4A16_MOE=1`` keeps a view of checkpoint-layout NVFP4
weights *after* Marlin conversion has allocated its shuffled tensors, then
for M <= MAX_M runs the in-tree Triton NVFP4 weight-only GEMM
(``fused_moe_nvfp4_emulation_kernel``) **without** activation QDQ. That is
W4A16: bf16 activations, NVFP4 weights dequantized as
``e2m1 * block_scale * global_scale``, then Marlin's SwiGLU clamp.

The pack shares checkpoint storage (no clone). Cloning the full expert pack
*before* Marlin permute/scale conversion is a startup OOM on 72GB / 0.91:
peak = clone + conversion temporaries. Marlin does not write back into the
checkpoint tensors, so a view is enough for the small-M path.

Unset / default stays bit-for-bit Marlin (no extra view, no extra peak).
Large-M stays Marlin. This is not the W4A4 ``triton_nvfp4_decode`` overlay
(``DSV41_EXP_TRITON_MOE`` / b12x).

The overlay refuses to enable when ``swiglu_limit`` is unset so the
checkpoint clamp is never skipped.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.utils.math_utils import cdiv

logger = init_logger(__name__)

ENV_FLAG = "DSV41_EXP_FUSED_W4A16_MOE"
ENV_MAX_M = "DSV41_EXP_FUSED_W4A16_MOE_MAX_M"
ENV_TILE = "DSV41_EXP_FUSED_W4A16_MOE_TILE"
LAYER_PACK_ATTR = "_dsv41_nvfp4_w4a16_decode_pack"

# Decode-sized default: 16 seqs * DSpark k=3 (1+3 tokens/seq) = 64.
_DEFAULT_MAX_M = 64
_DEFAULT_TILE = dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64, GROUP_SIZE_M=1)
_NVFP4_GROUP = 16
_RTX_PRO_5000_SMS = 110

# DSv4.1-Flash TP4 shapes used for occupancy logging / tests.
DSV41_FLASH_TP4_H = 5120
DSV41_FLASH_TP4_I = 576


def fused_w4a16_decode_enabled() -> bool:
    return os.environ.get(ENV_FLAG, "0") == "1"


def fused_w4a16_decode_max_m() -> int:
    raw = os.environ.get(ENV_MAX_M)
    if raw is None or raw == "":
        return _DEFAULT_MAX_M
    try:
        value = int(raw)
    except ValueError:
        logger.warning("%s=%r is not an int; using %d", ENV_MAX_M, raw, _DEFAULT_MAX_M)
        return _DEFAULT_MAX_M
    return value


def decode_moe_tile_config() -> dict[str, int]:
    cfg = dict(_DEFAULT_TILE)
    raw = os.environ.get(ENV_TILE)
    if not raw:
        return cfg
    aliases = {
        "BM": "BLOCK_SIZE_M",
        "BN": "BLOCK_SIZE_N",
        "BK": "BLOCK_SIZE_K",
        "GM": "GROUP_SIZE_M",
    }
    try:
        for part in raw.split(","):
            key, value = part.split("=", 1)
            key = key.strip()
            cfg[aliases.get(key, key)] = int(value.strip())
    except ValueError:
        logger.warning("%s=%r is invalid; using defaults %s", ENV_TILE, raw, cfg)
        return dict(_DEFAULT_TILE)
    return cfg


def _expert_scale(scale: torch.Tensor, num_experts: int) -> torch.Tensor:
    t = scale.detach().float().reshape(-1)
    if t.numel() == 1:
        t = t.expand(num_experts)
    elif t.numel() == num_experts * 2:
        t = t.view(num_experts, 2)[:, 0]
    elif t.numel() != num_experts:
        raise ValueError(
            f"expected {num_experts} expert scales, got {tuple(scale.shape)}"
        )
    return t.contiguous()


def _hold_checkpoint_tensor(t: torch.Tensor) -> torch.Tensor:
    """Keep checkpoint-layout storage without cloning the full pack.

    Marlin conversion allocates new shuffled tensors and does not mutate
    these. A clone here was the 72GB startup OOM (clone + permute/scale
    temps). ``contiguous()`` is a no-op (view) when the tensor already is.
    """
    held = t.detach()
    if not held.is_contiguous():
        held = held.contiguous()
    return held


@dataclass
class Nvfp4W4A16DecodePack:
    """Checkpoint-layout NVFP4 tensors (linear block scales, not Marlin)."""

    w13: torch.Tensor
    w13_scale: torch.Tensor
    w13_scale_2: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor
    w2_scale_2: torch.Tensor

    @property
    def num_experts(self) -> int:
        return int(self.w13.shape[0])

    @property
    def hidden_size(self) -> int:
        return int(self.w2.shape[1])

    @property
    def intermediate_size(self) -> int:
        return int(self.w2.shape[2]) * 2

    def shares_weight_storage(
        self,
        w13: torch.Tensor,
        w13_scale: torch.Tensor,
        w2: torch.Tensor,
        w2_scale: torch.Tensor,
    ) -> bool:
        """True when the pack is a view of the checkpoint weight/scale tensors."""
        return (
            self.w13.data_ptr() == w13.data_ptr()
            and self.w13_scale.data_ptr() == w13_scale.data_ptr()
            and self.w2.data_ptr() == w2.data_ptr()
            and self.w2_scale.data_ptr() == w2_scale.data_ptr()
        )


def maybe_snapshot_nvfp4_w4a16_decode_weights(
    layer: torch.nn.Module,
    *,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w13_scale_2: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    w2_scale_2: torch.Tensor,
) -> Nvfp4W4A16DecodePack | None:
    """Hold checkpoint-layout NVFP4 tensors without cloning the pack.

    Must run *after* Marlin conversion so permute/scale temporaries do not
    coexist with an extra full-size copy. Returns None when the env flag is
    unset so the default path allocates nothing extra.
    """
    if not fused_w4a16_decode_enabled():
        if hasattr(layer, LAYER_PACK_ATTR):
            delattr(layer, LAYER_PACK_ATTR)
        return None

    pack = Nvfp4W4A16DecodePack(
        w13=_hold_checkpoint_tensor(w13),
        w13_scale=_hold_checkpoint_tensor(w13_scale),
        w13_scale_2=_expert_scale(w13_scale_2, int(w13.shape[0])),
        w2=_hold_checkpoint_tensor(w2),
        w2_scale=_hold_checkpoint_tensor(w2_scale),
        w2_scale_2=_expert_scale(w2_scale_2, int(w2.shape[0])),
    )
    setattr(layer, LAYER_PACK_ATTR, pack)
    return pack


def overlay_disable_reason(
    *,
    pack: Nvfp4W4A16DecodePack | None,
    max_m: int,
    in_dtype: torch.dtype,
    activation: MoEActivation,
    clamp_limit: float | None,
    apply_router_weight_on_input: bool,
    has_bias: bool,
    is_lora: bool,
) -> str | None:
    """Return a disable reason, or None if the overlay may attach."""
    if not fused_w4a16_decode_enabled():
        return "flag unset"
    if max_m <= 0:
        return f"{ENV_MAX_M}={max_m}"
    if pack is None:
        return "no decode-layout snapshot"
    if clamp_limit is None:
        return "swiglu_limit is unset"
    if activation != MoEActivation.SILU:
        return f"activation={activation}"
    if in_dtype != torch.bfloat16:
        return f"in_dtype={in_dtype}"
    if apply_router_weight_on_input:
        return "apply_router_weight_on_input"
    if has_bias:
        return "expert bias"
    if is_lora:
        return "LoRA"
    h = pack.hidden_size
    i = pack.intermediate_size
    if h % _NVFP4_GROUP or i % _NVFP4_GROUP:
        return f"NVFP4 group-16 mismatch H={h} I={i}"
    if tuple(pack.w13.shape) != (pack.num_experts, 2 * i, h // 2):
        return f"unexpected w13 shape {tuple(pack.w13.shape)}"
    if tuple(pack.w2.shape) != (pack.num_experts, h, i // 2):
        return f"unexpected w2 shape {tuple(pack.w2.shape)}"
    if tuple(pack.w13_scale.shape) != (pack.num_experts, 2 * i, h // _NVFP4_GROUP):
        return f"unexpected w13_scale shape {tuple(pack.w13_scale.shape)}"
    if tuple(pack.w2_scale.shape) != (pack.num_experts, h, i // _NVFP4_GROUP):
        return f"unexpected w2_scale shape {tuple(pack.w2_scale.shape)}"
    tile = decode_moe_tile_config()
    if tile["BLOCK_SIZE_K"] % 2:
        return f"BLOCK_SIZE_K={tile['BLOCK_SIZE_K']} must be even"
    return None


def occupancy_rows(
    *,
    token_counts: tuple[int, ...] = (4, 8, 16, 24, 64),
    hidden_size: int = DSV41_FLASH_TP4_H,
    intermediate_size: int = DSV41_FLASH_TP4_I,
    num_experts: int = 256,
    topk: int = 8,
    sm_count: int = _RTX_PRO_5000_SMS,
    tile: dict[str, int] | None = None,
) -> list[dict[str, float | int]]:
    """CTA / wave counts for the two Triton GEMMs (CPU-safe)."""
    tile = dict(_DEFAULT_TILE) if tile is None else tile
    bm = tile["BLOCK_SIZE_M"]
    bn = tile["BLOCK_SIZE_N"]
    rows: list[dict[str, float | int]] = []
    for m in token_counts:
        # Match moe_align_block_size + the emulation kernel's EM shrink
        # when M < BLOCK_SIZE_M.
        npairs = m * topk
        max_pad = npairs + num_experts * (bm - 1)
        if npairs < num_experts:
            max_pad = min(npairs * bm, max_pad)
        em = max_pad
        if m < bm:
            em = min(max_pad, npairs * bm)
        n_blocks_m = cdiv(em, bm)
        fc1 = n_blocks_m * cdiv(2 * intermediate_size, bn)
        fc2 = n_blocks_m * cdiv(hidden_size, bn)
        rows.append(
            {
                "M": m,
                "fc1_ctas": fc1,
                "fc1_waves": fc1 / sm_count,
                "fc2_ctas": fc2,
                "fc2_waves": fc2 / sm_count,
            }
        )
    return rows


def _is_current_stream_capturing() -> bool:
    if not torch.cuda.is_available():
        return False
    is_capturing = getattr(torch.cuda, "is_current_stream_capturing", None)
    try:
        return bool(is_capturing is not None and is_capturing())
    except RuntimeError:
        return False


def _as_uint8_scale(scale: torch.Tensor) -> torch.Tensor:
    if scale.dtype == torch.uint8:
        return scale
    return scale.view(torch.uint8)


class Nvfp4W4A16DecodeMoE:
    """Per-layer small-M runner over a checkpoint-layout NVFP4 pack."""

    def __init__(
        self,
        pack: Nvfp4W4A16DecodePack,
        *,
        topk: int,
        max_m: int,
        clamp_limit: float,
    ):
        self.pack = pack
        self.topk = int(topk)
        self.max_m = int(max_m)
        self.clamp_limit = float(clamp_limit)
        self.tile = decode_moe_tile_config()
        self.w13 = pack.w13.view(torch.uint8)
        self.w2 = pack.w2.view(torch.uint8)
        self.w13_scale = _as_uint8_scale(pack.w13_scale)
        self.w2_scale = _as_uint8_scale(pack.w2_scale)
        self.w13_scale_2 = pack.w13_scale_2
        self.w2_scale_2 = pack.w2_scale_2
        self._warmed = False

    def can_run(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        apply_router_weight_on_input: bool,
        expert_map: torch.Tensor | None,
    ) -> bool:
        m = int(hidden_states.shape[0])
        if m <= 0 or m > self.max_m:
            return False
        if activation != MoEActivation.SILU:
            return False
        if apply_router_weight_on_input or expert_map is not None:
            return False
        if hidden_states.dtype != torch.bfloat16:
            return False
        if not hidden_states.is_contiguous():
            return False
        if int(topk_ids.shape[1]) != self.topk:
            return False
        capturing_unwarmed = _is_current_stream_capturing() and not self._warmed
        return not capturing_unwarmed

    def __call__(
        self,
        *,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_map: torch.Tensor | None,
        activation_fn: Callable[..., None],
        moe_sum_fn: Callable[..., None],
    ) -> torch.Tensor:
        from vllm.model_executor.layers.fused_moe.experts.nvfp4_emulation_moe import (
            invoke_fused_moe_nvfp4_emulation_kernel,
        )
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
            moe_align_block_size,
        )
        from vllm.triton_utils import tl

        m = int(hidden_states.shape[0])
        topk = self.topk
        hidden = self.pack.hidden_size
        intermediate = self.pack.intermediate_size
        n13 = 2 * intermediate
        config = dict(self.tile)
        compute_type = tl.bfloat16

        cache1 = _resize_cache(workspace2, (m, topk, n13))
        cache2 = _resize_cache(workspace13, (m * topk, intermediate))
        cache3 = _resize_cache(workspace2, (m, topk, hidden))

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids,
            config["BLOCK_SIZE_M"],
            self.pack.num_experts,
            expert_map,
        )
        dummy_act_scale = self.w13_scale_2
        invoke_fused_moe_nvfp4_emulation_kernel(
            hidden_states,
            self.w13,
            cache1,
            self.w13_scale,
            dummy_act_scale,
            self.w13_scale_2,
            None,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            False,
            topk,
            config,
            compute_type=compute_type,
        )
        activation_fn(
            activation,
            cache2,
            cache1.view(-1, n13),
            topk_ids=topk_ids,
            expert_map=expert_map,
        )
        invoke_fused_moe_nvfp4_emulation_kernel(
            cache2,
            self.w2,
            cache3,
            self.w2_scale,
            dummy_act_scale,
            self.w2_scale_2,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            True,
            1,
            config,
            compute_type=compute_type,
        )
        moe_sum_fn(cache3, output, topk_ids, expert_map)
        self._warmed = True
        return output


_ATTACH_LOGGED = False


def try_attach_w4a16_decode(
    experts: Any,
    layer: torch.nn.Module,
) -> Nvfp4W4A16DecodeMoE | None:
    """Build the overlay runner, or None (stay on Marlin) with a log."""
    global _ATTACH_LOGGED
    pack = getattr(layer, LAYER_PACK_ATTR, None)
    max_m = fused_w4a16_decode_max_m()
    clamp_limit = experts.activation_config.clamp_limit
    apply_router = bool(getattr(layer, "apply_router_weight_on_input", False))
    reason = overlay_disable_reason(
        pack=pack,
        max_m=max_m,
        in_dtype=experts.moe_config.in_dtype,
        activation=experts.moe_config.activation,
        clamp_limit=clamp_limit,
        apply_router_weight_on_input=apply_router,
        has_bias=experts.moe_config.has_bias,
        is_lora=experts.moe_config.is_lora_enabled,
    )
    if reason is not None:
        if fused_w4a16_decode_enabled() and not _ATTACH_LOGGED:
            _ATTACH_LOGGED = True
            logger.warning("%s: disabled for this layer (%s)", ENV_FLAG, reason)
        return None

    assert pack is not None and clamp_limit is not None
    runner = Nvfp4W4A16DecodeMoE(
        pack,
        topk=int(experts.moe_config.experts_per_token),
        max_m=max_m,
        clamp_limit=clamp_limit,
    )
    if not _ATTACH_LOGGED:
        _ATTACH_LOGGED = True
        logger.info(
            "%s=1: fused NVFP4 W4A16 decode MoE for M<=%d "
            "(E=%d H=%d I=%d topk=%d swiglu_limit=%s tile=%s), Marlin above",
            ENV_FLAG,
            runner.max_m,
            runner.pack.num_experts,
            runner.pack.hidden_size,
            runner.pack.intermediate_size,
            runner.topk,
            runner.clamp_limit,
            runner.tile,
        )
    return runner
