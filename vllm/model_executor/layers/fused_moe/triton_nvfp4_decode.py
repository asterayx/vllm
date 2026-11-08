# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental Triton NVFP4 (W4A4) decode MoE for small token counts.

Hosted by B12xExperts when an opt-in flag is set and M <= MAX_M:
  * QWEN4_EXP_TRITON_MOE=1 — Qwen path, default MAX_M=96
  * DSV41_EXP_TRITON_MOE=1 — DeepSeek-V4.x NVFP4 target MoE; auto backend
    switches from Marlin to B12X so this kernel can attach. Unset restores
    auto/Marlin. Draft DSpark experts stay native MXFP4 and are not covered.

Reads b12x's *prepared* expert tensors in place (no copies):
  w1_fp4 [E, 2I, H/2] u8 in kernel order [up; gate] (after b12x's one-time w31 flip),
    w2_fp4 [E, H, I/2] u8, 128x4-swizzled e4m3 block scales, per-expert fp32 alphas / a-gscales.
Pipeline: xq (x -> FP4 + e4m3 sf per (token, expert) using a1_gscale[e]) -> fc1 (tl.dot_scaled
NVFP4 MMA, SiLU*up with optional SwiGLU clamp, bf16, re-quant with a2_gscale[e]) ->
fc2 (dot_scaled, *alpha2, [bf16], *router_w, fp32 atomic scatter) -> fin (fp32 -> bf16,
re-zero accumulator). DSv4.1 NVFP4 stores per-expert ``w13_input_scale``; a1_gscale is
``1/w13_input_scale`` with shape [E]. Shared a1_gscale is the special case where every
expert uses the same scale.
Quantization numerics mirror b12x: QMODE 1 = micro kernel (M <= 8), QMODE 2 = dynamic
kernel fast_math (M > 8). SwiGLU clamp matches Marlin/b12x SILU: gate.clamp(max=L),
up.clamp(-L, L), then silu(gate)*up. DSv4.1 checkpoints set L=10.0.
"""

import os
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

DSV41_TRITON_MOE_ENV = "DSV41_EXP_TRITON_MOE"
QWEN_TRITON_MOE_ENV = "QWEN4_EXP_TRITON_MOE"
_DSV41_MODEL_TYPES = frozenset(
    {
        "deepseek_v4",
        "deepseek_v4_text",
        "deepseek_v41",
        "deepseek_v41_text",
    }
)

# RTX PRO 5000 Blackwell (sm_120). DSv4.1-Flash TP4 expert shapes.
SM120_NUM_SMS = 110
DSV41_FLASH_HIDDEN = 5120
DSV41_FLASH_INTERMEDIATE = 2304
DSV41_FLASH_INTERMEDIATE_TP4 = DSV41_FLASH_INTERMEDIATE // 4  # 576
DSV41_FLASH_TOPK = 6
DSV41_FLASH_EXPERTS = 384
DSV41_DECODE_MS = (4, 8, 16, 24)

_DEFAULT_CFG = dict(
    BM=16,
    BN1=32,
    BK1=512,
    W1=4,
    S1=2,
    BN2=128,
    BK2=64,
    W2=4,
    S2=2,
    FB=1024,
    XB=256,
)
_BN1_CANDIDATES = (32, 64, 16)
_BN2_CANDIDATES = (128, 256, 64)
_BK1_CANDIDATES = (512, 256)
_BK2_CANDIDATES = (64, 32)


def current_hf_model_type() -> str | None:
    try:
        from vllm.config import get_current_vllm_config

        hf_config = get_current_vllm_config().model_config.hf_config
    except Exception:
        return None
    return getattr(hf_config, "model_type", None)


def dsv41_triton_nvfp4_decode_requested() -> bool:
    """True when DSV41_EXP_TRITON_MOE=1 applies to this process.

    Honors the env when model_type is unset (unit tests / early init). Other
    model types ignore the flag so it cannot change Qwen's backend selection.
    """
    if os.environ.get(DSV41_TRITON_MOE_ENV, "0") != "1":
        return False
    model_type = current_hf_model_type()
    return model_type is None or model_type in _DSV41_MODEL_TYPES


def triton_nvfp4_decode_overlay_opt_in() -> tuple[bool, int, str, str]:
    """B12xExperts overlay: (enabled, max_m, flag_name, tile_profile)."""
    dsv41 = os.environ.get(DSV41_TRITON_MOE_ENV, "0") == "1"
    qwen = os.environ.get(QWEN_TRITON_MOE_ENV, "0") == "1"
    if dsv41:
        raw = os.environ.get("DSV41_EXP_TRITON_MOE_MAX_M")
        if raw is None:
            raw = os.environ.get("QWEN4_EXP_TRITON_MOE_MAX_M", "96")
        return True, int(raw), DSV41_TRITON_MOE_ENV, "dsv41_sm120"
    if qwen:
        raw = os.environ.get("QWEN4_EXP_TRITON_MOE_MAX_M", "96")
        return True, int(raw), QWEN_TRITON_MOE_ENV, "default"
    return False, 0, "", "default"


def normalize_swiglu_clamp_limit(limit: float | None) -> float | None:
    """Positive clamp limit, else None. Marlin/b12x skip clamp when limit <= 0."""
    if limit is None:
        return None
    limit_f = float(limit)
    return limit_f if limit_f > 0.0 else None


def silu_mul_with_optional_clamp(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    limit: float | None = None,
) -> torch.Tensor:
    """CPU reference for fused FC1 SILU.

    Matches Marlin ``silu_and_mul_with_clamp`` / b12x SILU (alpha=1, beta=0):
    ``silu(gate.clamp(max=L)) * up.clamp(-L, L)``. ``limit is None`` is
    unclamped ``silu(gate) * up``.
    """
    g = gate.to(torch.float32)
    u = up.to(torch.float32)
    limit_f = normalize_swiglu_clamp_limit(limit)
    if limit_f is not None:
        g = torch.clamp(g, max=limit_f)
        u = torch.clamp(u, min=-limit_f, max=limit_f)
    return g * torch.sigmoid(g) * u


def a1_gscale_is_shared(g1: torch.Tensor) -> bool:
    """True when every expert uses the same activation global scale."""
    g = g1.reshape(-1)
    return bool(g.numel() <= 1 or (g == g[0]).all().item())


def pair_a1_gscales(
    a1_gscale: torch.Tensor, topk_ids: torch.Tensor
) -> torch.Tensor:
    """Per (token, expert) a1_gscale in row-major ``[M, topk]`` pair order.

    Matches the xq pre-pass: pair ``p = t * topk + k`` uses ``a1_gscale[ids[t, k]]``.
    """
    g = a1_gscale.float().reshape(-1)
    return g[topk_ids.reshape(-1).long()]


def cfg_divides_shapes(
    cfg: dict[str, int], hidden_size: int, intermediate_size: int
) -> bool:
    return (
        intermediate_size % cfg["BN1"] == 0
        and hidden_size % cfg["BN2"] == 0
        and hidden_size % cfg["BK1"] == 0
        and hidden_size % cfg["XB"] == 0
        and intermediate_size % cfg["BK2"] == 0
    )


def estimate_fc1_ctas(
    num_tokens: int,
    topk: int,
    intermediate_size: int,
    bn1: int,
    num_experts: int | None = None,
) -> int:
    """Upper-bound FC1 grid size: one BM-padded block per routed pair."""
    pairs = max(int(num_tokens) * int(topk), 1)
    if num_experts is not None:
        pairs = min(pairs, int(num_experts))
    n_tiles = max(int(intermediate_size) // max(int(bn1), 1), 1)
    return pairs * n_tiles


def parse_tile_override(raw: str | None, base: dict[str, int]) -> dict[str, int]:
    if not raw:
        return dict(base)
    out = dict(base)
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        key, _, value = part.partition("=")
        key = key.strip().upper()
        if key not in out:
            raise ValueError(f"unknown Triton NVFP4 decode tile key {key}")
        out[key] = int(value.strip())
    return out


def _first_divisor(size: int, candidates: tuple[int, ...]) -> int | None:
    for cand in candidates:
        if size % cand == 0:
            return cand
    return None


def choose_triton_nvfp4_decode_cfg(
    hidden_size: int,
    intermediate_size: int,
    *,
    profile: str = "default",
    num_sms: int | None = None,
    topk: int = DSV41_FLASH_TOPK,
    num_experts: int = DSV41_FLASH_EXPERTS,
) -> dict[str, int]:
    """Pick tiles that divide (H, I). ``dsv41_sm120`` targets 110 SMs.

    BN1 stays on the proven 32-wide NVFP4 MMA tile when it divides I. If M=4
    FC1 CTAs would under-fill ``num_sms``, BN1 shrinks to 16. Override with
    DSV41_EXP_TRITON_MOE_TILE=BM=16,BN1=32,...
    """
    cfg = dict(_DEFAULT_CFG)
    if profile == "dsv41_sm120":
        sms = SM120_NUM_SMS if num_sms is None else int(num_sms)
        bn1 = _first_divisor(intermediate_size, _BN1_CANDIDATES)
        if bn1 is not None:
            m4 = estimate_fc1_ctas(
                4, topk, intermediate_size, bn1, num_experts
            )
            if m4 < sms and intermediate_size % 16 == 0:
                bn1 = 16
            cfg["BN1"] = bn1
        bn2 = _first_divisor(hidden_size, _BN2_CANDIDATES)
        if bn2 is not None:
            cfg["BN2"] = bn2
        bk1 = _first_divisor(hidden_size, _BK1_CANDIDATES)
        if bk1 is not None:
            cfg["BK1"] = bk1
        bk2 = _first_divisor(intermediate_size, _BK2_CANDIDATES)
        if bk2 is not None:
            cfg["BK2"] = bk2
    env_key = (
        "DSV41_EXP_TRITON_MOE_TILE"
        if profile == "dsv41_sm120"
        else "QWEN4_EXP_TRITON_MOE_TILE"
    )
    cfg = parse_tile_override(os.environ.get(env_key), cfg)
    if not cfg_divides_shapes(cfg, hidden_size, intermediate_size):
        raise ValueError(
            f"tile does not divide shapes H={hidden_size} I={intermediate_size} "
            f"cfg={cfg}"
        )
    return cfg


def decode_occupancy_rows(
    cfg: dict[str, int],
    hidden_size: int,
    intermediate_size: int,
    *,
    topk: int = DSV41_FLASH_TOPK,
    num_sms: int = SM120_NUM_SMS,
    num_experts: int = DSV41_FLASH_EXPERTS,
    token_counts: tuple[int, ...] = DSV41_DECODE_MS,
) -> list[dict[str, float]]:
    rows = []
    fc2_n = max(hidden_size // cfg["BN2"], 1)
    for m in token_counts:
        fc1 = estimate_fc1_ctas(
            m, topk, intermediate_size, cfg["BN1"], num_experts
        )
        fc2 = min(m * topk, num_experts) * fc2_n
        rows.append(
            {
                "M": m,
                "fc1_ctas": fc1,
                "fc2_ctas": fc2,
                "fc1_waves": fc1 / num_sms,
                "fc2_waves": fc2 / num_sms,
            }
        )
    return rows


@triton.jit
def _rcp(x):  # PTX rcp.approx.ftz.f32 (what b12x fast_math uses)
    return tl.inline_asm_elementwise("rcp.approx.ftz.f32 $0, $1;", "=f,f", [x],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _ex2(x):
    return tl.inline_asm_elementwise("ex2.approx.ftz.f32 $0, $1;", "=f,f", [x],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _sf_off(r, kb, C4: tl.constexpr):  # 128x4 swizzled e4m3 scale offset; C4 = pad4(cols)/4
    return ((((r // 128) * C4 + kb // 4) * 32 + (r % 32)) * 4 + (r % 128) // 32) * 4 + kb % 4


@triton.jit
def _silu_mul(g, u, QMODE: tl.constexpr, HAS_CLAMP: tl.constexpr, LIMIT: tl.constexpr):
    # Marlin/b12x SILU: gate.clamp(max=L), up.clamp(-L, L), then silu(gate)*up.
    if HAS_CLAMP:
        g = tl.minimum(g, LIMIT)
        u = tl.minimum(tl.maximum(u, -LIMIT), LIMIT)
    if QMODE == 2:  # b12x: rcp_approx(1 + exp_fast(-g)); gate * sig * up
        sig = _rcp(1.0 + _ex2(-g * 1.4426950408889634))
    else:
        sig = tl.math.div_rn(tl.full(g.shape, 1.0, tl.float32),
                             1.0 + tl.extra.cuda.libdevice.exp(-g))
    return g * sig * u


@triton.jit
def _quant_codes(v, gs, R: tl.constexpr, C: tl.constexpr, QMODE: tl.constexpr):
    # fp32 [R, C] -> (packed e2m1 codes u8 [R, C//2] (low nibble = even elem), e4m3 sf [R, C//16])
    vb = v.reshape(R, C // 16, 16)
    mag = tl.abs(vb)
    bmax = tl.max(mag, axis=2)
    if QMODE == 2:
        raw = tl.minimum(gs * (bmax * _rcp(tl.full(bmax.shape, 6.0, tl.float32))), 448.0)
    else:
        raw = tl.minimum(tl.math.div_rn(bmax * gs, 6.0), 448.0)
    sf8 = raw.to(tl.float8e4nv)
    sf = sf8.to(tl.float32)
    if QMODE == 1:
        eff = tl.maximum(tl.math.div_rn(sf, gs), 1e-30)
        t = mag * tl.math.div_rn(tl.full(eff.shape, 1.0, tl.float32), eff)[:, :, None]
    else:
        inv = tl.where(sf == 0.0, 0.0, _rcp(sf))
        t = mag * (inv * gs)[:, :, None]
    # RNE onto e2m1 magnitude codes 0..7 = {0, .5, 1, 1.5, 2, 3, 4, 6}
    c = tl.where((t > 0.25) & (t < 0.75), 1, 0)
    c = tl.where((t >= 0.75) & (t <= 1.25), 2, c)
    c = tl.where((t > 1.25) & (t < 1.75), 3, c)
    c = tl.where((t >= 1.75) & (t <= 2.5), 4, c)
    c = tl.where((t > 2.5) & (t < 3.5), 5, c)
    c = tl.where((t >= 3.5) & (t <= 5.0), 6, c)
    c = tl.where(t > 5.0, 7, c)
    c = tl.where((vb < 0) & (c > 0), c | 8, c)
    c2 = c.reshape(R, C // 2, 2)
    a, b = tl.split(c2)
    return (a | (b << 4)).to(tl.uint8), sf8


@triton.jit
def _xq_kernel(x_ptr, g1_ptr, topk_ids_ptr, xq_ptr, xs_ptr, H: tl.constexpr,
               BK: tl.constexpr, TOPK: tl.constexpr, E: tl.constexpr,
               QMODE: tl.constexpr):
    # One CTA per (token, expert) pair so a1_gscale[e] can differ across experts.
    p = tl.program_id(0)
    k0 = tl.program_id(1) * BK
    t = p // TOPK
    e = tl.load(topk_ids_ptr + p)
    e = tl.where((e >= 0) & (e < E), e, 0).to(tl.int64)
    gs1 = tl.load(g1_ptr + e)
    xv = tl.load(x_ptr + t * H + k0 + tl.arange(0, BK)).to(tl.float32)
    q, s = _quant_codes(xv[None, :], gs1, 1, BK, QMODE)
    tl.store(xq_ptr + p * (H // 2) + k0 // 2 + tl.arange(0, BK // 2), q.reshape(BK // 2))
    tl.store(xs_ptr + p * (H // 16) + k0 // 16 + tl.arange(0, BK // 16), s.reshape(BK // 16))


@triton.jit
def _fc1_kernel(xq_ptr, xs_ptr, w1_ptr, s1_ptr, a1_ptr, g2_ptr, sorted_ptr, eid_ptr, npost_ptr,
                iq_ptr, is_ptr, num_pairs, TOPK: tl.constexpr, H: tl.constexpr, I: tl.constexpr,
                S1PE: tl.constexpr, C4: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                BK: tl.constexpr, QMODE: tl.constexpr, HAS_CLAMP: tl.constexpr,
                LIMIT: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    if pm * BM >= tl.load(npost_ptr):
        return
    e = tl.load(eid_ptr + pm).to(tl.int64)
    rows = tl.load(sorted_ptr + pm * BM + tl.arange(0, BM))
    valid = rows < num_pairs
    al1 = tl.load(a1_ptr + e)
    gs2 = tl.load(g2_ptr + e)
    n = pn * BN + tl.arange(0, BN)
    wbase = w1_ptr + e * (2 * I * (H // 2))
    sbase = s1_ptr + e * S1PE
    acc_u = tl.zeros((BM, BN), tl.float32)
    acc_g = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, H, BK):
        kp = k0 // 2 + tl.arange(0, BK // 2)
        kb = k0 // 16 + tl.arange(0, BK // 16)
        xa = tl.load(xq_ptr + rows[:, None] * (H // 2) + kp[None, :], mask=valid[:, None], other=0)
        xs = tl.load(xs_ptr + rows[:, None] * (H // 16) + kb[None, :], mask=valid[:, None],
                     other=0.0)
        wu = tl.load(wbase + n[:, None] * (H // 2) + kp[None, :], cache_modifier=".cg")
        wg = tl.load(wbase + (I + n)[:, None] * (H // 2) + kp[None, :], cache_modifier=".cg")
        su = tl.load(sbase + _sf_off(n[:, None], kb[None, :], C4))
        sg = tl.load(sbase + _sf_off((I + n)[:, None], kb[None, :], C4))
        acc_u = tl.dot_scaled(xa, xs, "e2m1", tl.trans(wu), su, "e2m1", acc_u)
        acc_g = tl.dot_scaled(xa, xs, "e2m1", tl.trans(wg), sg, "e2m1", acc_g)
    inter = _silu_mul(acc_g * al1, acc_u * al1, QMODE, HAS_CLAMP, LIMIT).to(
        tl.bfloat16).to(tl.float32)
    q, s = _quant_codes(inter, gs2, BM, BN, QMODE)
    tl.store(iq_ptr + rows[:, None] * (I // 2) + (pn * BN // 2 + tl.arange(0, BN // 2))[None, :],
             q, mask=valid[:, None])
    tl.store(is_ptr + rows[:, None] * (I // 16) + (pn * BN // 16 + tl.arange(0, BN // 16))[None, :],
             s, mask=valid[:, None])


@triton.jit
def _fc2_kernel(iq_ptr, is_ptr, w2_ptr, s2_ptr, a2_ptr, tw_ptr, sorted_ptr, eid_ptr, npost_ptr,
                acc_ptr, num_pairs, TOPK: tl.constexpr, H: tl.constexpr, I: tl.constexpr,
                S2PE: tl.constexpr, C4: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                BK: tl.constexpr, DOWNBF16: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    if pm * BM >= tl.load(npost_ptr):
        return
    e = tl.load(eid_ptr + pm).to(tl.int64)
    rows = tl.load(sorted_ptr + pm * BM + tl.arange(0, BM))
    valid = rows < num_pairs
    al2 = tl.load(a2_ptr + e)
    h = pn * BN + tl.arange(0, BN)
    wbase = w2_ptr + e * (H * (I // 2))
    sbase = s2_ptr + e * S2PE
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in tl.static_range(0, I, BK):
        kp = k0 // 2 + tl.arange(0, BK // 2)
        kb = k0 // 16 + tl.arange(0, BK // 16)
        mp = kp < I // 2
        mb = kb < I // 16
        a = tl.load(iq_ptr + rows[:, None] * (I // 2) + kp[None, :],
                    mask=valid[:, None] & mp[None, :], other=0)
        asc = tl.load(is_ptr + rows[:, None] * (I // 16) + kb[None, :],
                      mask=valid[:, None] & mb[None, :], other=0.0)
        w = tl.load(wbase + h[:, None] * (I // 2) + kp[None, :], mask=mp[None, :], other=0,
                    cache_modifier=".cg")
        ws = tl.load(sbase + _sf_off(h[:, None], kb[None, :], C4), mask=mb[None, :], other=0.0)
        acc = tl.dot_scaled(a, asc, "e2m1", tl.trans(w), ws, "e2m1", acc)
    rw = tl.load(tw_ptr + rows, mask=valid, other=0.0)
    tok = tl.where(valid, rows // TOPK, 0)
    down = acc * al2
    if DOWNBF16:
        down = down.to(tl.bfloat16).to(tl.float32)
    tl.atomic_add(acc_ptr + tok[:, None] * H + h[None, :], down * rw[:, None],
                  mask=valid[:, None], sem="relaxed")


@triton.jit
def _fin_kernel(acc_ptr, out_ptr, n, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < n
    v = tl.load(acc_ptr + o, mask=m)
    tl.store(out_ptr + o, v.to(out_ptr.dtype.element_ty), mask=m)
    tl.store(acc_ptr + o, tl.zeros_like(v), mask=m)


class _Scratch:
    """Per (device, M, shape) buffers, shared by every layer (layers run sequentially)."""

    def __init__(self, dev, M, topk, E, H, I, BM):
        npairs = M * topk
        self.max_pad = npairs + E * (BM - 1)
        self.max_blocks = triton.cdiv(self.max_pad, BM)
        self.sorted = torch.empty(self.max_pad, dtype=torch.int32, device=dev)
        self.eids = torch.empty(self.max_blocks, dtype=torch.int32, device=dev)
        self.npost = torch.empty(1, dtype=torch.int32, device=dev)
        self.xq = torch.empty(npairs, H // 2, dtype=torch.uint8, device=dev)
        self.xs = torch.empty(npairs, H // 16, dtype=torch.float8_e4m3fn, device=dev)
        self.iq = torch.empty(npairs, I // 2, dtype=torch.uint8, device=dev)
        self.isc = torch.empty(npairs, I // 16, dtype=torch.float8_e4m3fn, device=dev)
        self.acc = torch.zeros(M, H, dtype=torch.float32, device=dev)  # fin re-zeroes
        self.nbytes = sum(t.numel() * t.element_size() for t in (
            self.sorted, self.eids, self.npost, self.xq, self.xs, self.iq, self.isc, self.acc))


_SCRATCH: dict[tuple, _Scratch] = {}


def _w13_flipped(prepared: Any) -> bool:
    """True once b12x has flipped the w31 FC1 storage to kernel order [up; gate]."""
    if getattr(prepared, "w13_layout", "w31") != "w31":
        return True
    try:
        from b12x.moe.fused_moe import _impl
    except ImportError:
        return False
    key = (prepared.w1_fp4.data_ptr(), prepared.w1_blockscale.data_ptr())
    return key in _impl._W13_NORMALIZED_STORAGES


def _cuda_sm_count() -> int | None:
    if not torch.cuda.is_available():
        return None
    try:
        return int(torch.cuda.get_device_properties(0).multi_processor_count)
    except Exception:
        return None


class TritonNvfp4DecodeMoE:
    """Per-layer runner over b12x prepared experts."""

    def __init__(
        self,
        prepared: Any,
        topk: int,
        max_m: int,
        cfg: dict[str, int] | None = None,
        profile: str = "default",
        swiglu_limit: float | None = None,
    ):
        E = int(prepared.num_experts)
        H = int(prepared.hidden_size)
        I = int(prepared.intermediate_size)
        self.E, self.H, self.I, self.topk, self.max_m = E, H, I, int(topk), int(max_m)
        self.profile = profile
        self.swiglu_limit = normalize_swiglu_clamp_limit(swiglu_limit)
        if cfg is None:
            cfg = choose_triton_nvfp4_decode_cfg(
                H,
                I,
                profile=profile,
                num_sms=_cuda_sm_count(),
                topk=int(topk),
                num_experts=E,
            )
        self.c = dict(cfg)
        if not cfg_divides_shapes(self.c, H, I):
            raise ValueError(
                f"tile does not divide shapes E={E} H={H} I={I} cfg={self.c}"
            )
        self.prepared = prepared
        self.w1 = prepared.w1_fp4.view(torch.uint8)
        self.w2 = prepared.w2_fp4.view(torch.uint8)
        if tuple(self.w1.shape) != (E, 2 * I, H // 2) or tuple(self.w2.shape) != (
            E,
            H,
            I // 2,
        ):
            raise ValueError(
                f"unexpected packed weight shapes {tuple(self.w1.shape)} "
                f"{tuple(self.w2.shape)}"
            )
        self.s1 = prepared.w1_blockscale.view(torch.float8_e4m3fn).reshape(E, -1)
        self.s2 = prepared.w2_blockscale.view(torch.float8_e4m3fn).reshape(E, -1)

        def f(t):
            t = t.float().reshape(-1)
            return (t.expand(E) if t.numel() == 1 else t.reshape(E)).contiguous()

        self.a1, self.a2 = f(prepared.w1_alphas), f(prepared.w2_alphas)
        self.g1, self.g2 = f(prepared.a1_gscale), f(prepared.a2_gscale)
        self.a1_gscale_shared = a1_gscale_is_shared(self.g1)
        self.C41 = (H // 16 + 3) // 4
        self.C42 = (I // 16 + 3) // 4
        if self.s1.shape[1] != ((2 * I + 127) // 128 * 128) * self.C41 * 4 or \
                self.s2.shape[1] != ((H + 127) // 128 * 128) * self.C42 * 4:
            raise ValueError(
                f"unexpected swizzled scale sizes {tuple(self.s1.shape)} "
                f"{tuple(self.s2.shape)}"
            )
        env = os.environ.get("DSV41_EXP_TRITON_MOE_DOWNBF16")
        if env is None:
            env = os.environ.get("QWEN4_EXP_TRITON_MOE_DOWNBF16")
        self.downbf16_override = None if env is None else int(env)
        self.dev = self.w1.device
        self._flipped = False

    def flipped(self) -> bool:
        if not self._flipped:
            self._flipped = _w13_flipped(self.prepared)
        return self._flipped

    def _key(self, M):
        return (self.dev, int(M), self.topk, self.E, self.H, self.I, self.c["BM"])

    def has_scratch(self, M: int) -> bool:
        return self._key(M) in _SCRATCH

    def scratch(self, M: int) -> _Scratch:
        k = self._key(M)
        s = _SCRATCH.get(k)
        if s is None:
            s = _Scratch(self.dev, int(M), self.topk, self.E, self.H, self.I, self.c["BM"])
            _SCRATCH[k] = s
        return s

    def __call__(self, x: torch.Tensor, topk_w: torch.Tensor, topk_ids: torch.Tensor,
                 out: torch.Tensor) -> torch.Tensor:
        import vllm._custom_ops as ops
        c = self.c
        M = int(x.shape[0])
        s = self.scratch(M)
        qmode = 1 if M <= 8 else 2
        downbf16 = (1 if qmode == 2 else 0) if self.downbf16_override is None \
            else self.downbf16_override
        npairs = M * self.topk
        ops.moe_align_block_size(topk_ids, self.E, c["BM"], s.sorted, s.eids, s.npost)
        nb = min(s.max_blocks, min(npairs, self.E) + npairs // c["BM"])
        ids = topk_ids.reshape(-1)
        _xq_kernel[(npairs, self.H // c["XB"])](
            x, self.g1, ids, s.xq, s.xs, self.H, BK=c["XB"], TOPK=self.topk,
            E=self.E, QMODE=qmode, num_warps=2)
        has_clamp = 1 if self.swiglu_limit is not None else 0
        clamp_limit = 0.0 if self.swiglu_limit is None else float(self.swiglu_limit)
        _fc1_kernel[(nb, self.I // c["BN1"])](
            s.xq, s.xs, self.w1, self.s1, self.a1, self.g2, s.sorted, s.eids, s.npost,
            s.iq, s.isc, npairs, self.topk, self.H, self.I, self.s1.shape[1], self.C41,
            BM=c["BM"], BN=c["BN1"], BK=c["BK1"], QMODE=qmode, HAS_CLAMP=has_clamp,
            LIMIT=clamp_limit, num_warps=c["W1"], num_stages=c["S1"])
        _fc2_kernel[(nb, self.H // c["BN2"])](
            s.iq, s.isc, self.w2, self.s2, self.a2, topk_w, s.sorted, s.eids, s.npost, s.acc,
            npairs, self.topk, self.H, self.I, self.s2.shape[1], self.C42,
            BM=c["BM"], BN=c["BN2"], BK=c["BK2"], DOWNBF16=downbf16,
            num_warps=c["W2"], num_stages=c["S2"])
        n = M * self.H
        _fin_kernel[(triton.cdiv(n, c["FB"]),)](s.acc, out, n, BLOCK=c["FB"], num_warps=4)
        return out

    @torch.inference_mode()
    def warmup(self, token_counts, dtype) -> list[int]:
        """Allocate scratch and JIT every specialization for each M <= max_m (pre-capture)."""
        done = []
        for M in sorted({int(t) for t in token_counts if 0 < int(t) <= self.max_m}):
            x = torch.zeros((M, self.H), dtype=dtype, device=self.dev)
            out = torch.empty_like(x)
            ids = (torch.arange(self.topk, device=self.dev, dtype=torch.int32)
                   .unsqueeze(0).expand(M, -1).contiguous())
            ids.remainder_(self.E)
            w = torch.full((M, self.topk), 1.0 / self.topk, dtype=torch.float32,
                           device=self.dev)
            self(x, w, ids, out)
            done.append(M)
        torch.cuda.synchronize(self.dev)
        return done
