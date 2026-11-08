# SPDX-License-Identifier: Apache-2.0
"""DPSK_WO_A_FP8_SKINNY: SM120 skinny grouped bmm for the DSV4.1 o_proj wo_a.

On SM120 the MXFP8 wo_a ([G*N, K] e4m3 + per-row/32-K UE8M0 scale) is
dequantized to bf16 at load (emulation kernel) and run through torch.bmm, which
cuBLAS serves with cutlass_80 bf16 32x32 tiles on only ~64 CTAs: ~24us/layer for
16 MB of bf16 weight. This Triton kernel streams the (losslessly re-packed)
1-byte e4m3 weight + 1/32-byte scale, dequantizes in registers to bf16 (exact:
e4m3 * 2^e fits bf16) and does the M<=64 x N tile with fp32 accumulation.

x: [M, G, K] bf16 (any strides; the inverse-rope output view)
w: [G*N, K] e4m3, s: [G*N, K//32] uint8 (E8M0)  ->  z: [M, G, N] bf16
"""
import logging
import os

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

WO_A_FP8_MAX_M = int(os.environ.get("DPSK_WO_A_FP8_MAX_M", "64"))


@triton.jit
def _wo_a_fp8_skinny_kernel(
    x_ptr, w_ptr, s_ptr, z_ptr,
    M,
    sxm, sxg, sxk,
    szm, szg,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    g = tl.program_id(1)
    pid_k = tl.program_id(2)
    SB: tl.constexpr = BLOCK_K // 32
    KS: tl.constexpr = K // 32
    K_PER: tl.constexpr = K // SPLIT_K

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rows = (g * N + offs_n).to(tl.int64)
    offs_kk = tl.arange(0, BLOCK_K)
    offs_sb = tl.arange(0, SB)
    m_mask = offs_m[:, None] < M

    x_base = x_ptr + g * sxg + offs_m[:, None] * sxm
    w_base = w_ptr + rows[:, None] * K
    s_base = s_ptr + rows[:, None] * KS

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_start = pid_k * K_PER
    for k0 in tl.range(k_start, k_start + K_PER, BLOCK_K):
        offs_k = k0 + offs_kk
        x = tl.load(x_base + offs_k[None, :] * sxk, mask=m_mask, other=0.0)
        w = tl.load(w_base + offs_k[None, :])
        s = tl.load(s_base + (k0 // 32 + offs_sb)[None, :])
        # E8M0 -> fp32 2^(e-127) by placing the byte in the exponent field.
        sf = (s.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        wf = tl.reshape(w.to(tl.float32), (BLOCK_N, SB, 32)) * sf[:, :, None]
        wb = tl.reshape(wf, (BLOCK_N, BLOCK_K)).to(tl.bfloat16)
        acc = tl.dot(x, tl.trans(wb), acc)

    z_ptrs = z_ptr + offs_m[:, None] * szm + g * szg + offs_n[None, :]
    if SPLIT_K == 1:
        tl.store(z_ptrs, acc.to(z_ptr.dtype.element_ty), mask=m_mask)
    else:
        tl.atomic_add(z_ptrs, acc, mask=m_mask, sem="relaxed")


# Tuned on RTX PRO 5000 Blackwell (110 SMs), G=2 N=1024 K=4096, CUDA-graph
# replay with 10 L2-rotated layers: cuBLAS cutlass_80 bf16 ~23.6us (M=2..16).
#   M<=16 : BN32 BK512 st3 w4  ~6.9-7.2us (~1.24 TB/s of fp8 bytes, 3.3x)
#   M<=32 : BN32 BK128 st4 w4  ~13.4-14.2us (1.7x)
#   M<=64 : BN32 BK256 st3 w4  ~18.1-18.9us (1.5x)
def _cfg(M: int, N: int, K: int, G: int):
    bm = max(16, triton.next_power_of_2(M))
    if bm <= 16:
        cfg = dict(BLOCK_N=32, BLOCK_K=512, num_stages=3)
    elif bm <= 32:
        cfg = dict(BLOCK_N=32, BLOCK_K=128, num_stages=4)
    else:
        cfg = dict(BLOCK_N=32, BLOCK_K=256, num_stages=3)
    cfg.update(BLOCK_M=bm, SPLIT_K=1, num_warps=4)
    return cfg


def wo_a_fp8_skinny(x: torch.Tensor, w: torch.Tensor, s: torch.Tensor,
                    n_groups: int, out: torch.Tensor | None = None, **over):
    M, G, K = x.shape
    assert G == n_groups and w.shape[1] == K and K % 32 == 0
    N = w.shape[0] // G
    cfg = _cfg(M, N, K, G)
    cfg.update(over)
    sk = cfg.pop("SPLIT_K")
    if out is None:
        out = (torch.zeros if sk > 1 else torch.empty)(
            (M, G, N), device=x.device,
            dtype=torch.float32 if sk > 1 else torch.bfloat16)
    assert N % cfg["BLOCK_N"] == 0 and (K // sk) % cfg["BLOCK_K"] == 0
    grid = (N // cfg["BLOCK_N"], G, sk)
    _wo_a_fp8_skinny_kernel[grid](
        x, w, s, out, M,
        x.stride(0), x.stride(1), x.stride(2),
        out.stride(0), out.stride(1),
        N=N, K=K, SPLIT_K=sk, **cfg)
    return out


def repack_wo_a_fp8(w_bf16: torch.Tensor, scale: torch.Tensor):
    """Exact inverse of dequant_mxfp8_to_bf16: (bf16 weight, E8M0 scale) ->
    (e4m3 weight, uint8 scale). Returns None if the round trip is not exact."""
    N, K = w_bf16.shape
    s = scale[:N, : K // 32].contiguous()
    s_u8 = s.view(torch.uint8) if s.dtype != torch.uint8 else s
    inv = torch.exp2(127.0 - s_u8.to(torch.float32)).repeat_interleave(32, dim=1)
    w8 = (w_bf16.to(torch.float32) * inv).to(torch.float8_e4m3fn)
    back = (w8.to(torch.float32).view(N, K // 32, 32)
            * torch.exp2(s_u8.to(torch.float32) - 127.0).unsqueeze(-1)
            ).view(N, K).to(torch.bfloat16)
    if not torch.equal(back, w_bf16):
        return None
    return w8.contiguous(), s_u8.contiguous()


# ---------------------------------------------------------------------------
# vLLM integration (DSV4.1 FlashInfer-sparse attention, SM120 emulation wo_a)
# ---------------------------------------------------------------------------
from torch import nn  # noqa: E402

from vllm.utils.torch_utils import direct_register_custom_op  # noqa: E402


def _dpsk_wo_a_fp8_bmm(x: torch.Tensor, w_bf16: torch.Tensor, w8: torch.Tensor,
                       s8: torch.Tensor, n_groups: int) -> torch.Tensor:
    """x [M,G,K] bf16 -> z [M,G,N] bf16. Small M: fp8 skinny; else bf16 bmm."""
    M, G, K = x.shape
    N = w_bf16.shape[0] // G
    if 1 <= M <= WO_A_FP8_MAX_M:
        return wo_a_fp8_skinny(x, w8, s8, n_groups)
    z = torch.empty((M, G, N), device=x.device, dtype=torch.bfloat16)
    torch.bmm(x.transpose(0, 1), w_bf16.view(G, N, K).transpose(1, 2),
              out=z.transpose(0, 1))
    return z


def _dpsk_wo_a_fp8_bmm_fake(x, w_bf16, w8, s8, n_groups):
    M, G, _ = x.shape
    return x.new_empty((M, G, w_bf16.shape[0] // G))


direct_register_custom_op(
    op_name="dpsk_wo_a_fp8_bmm",
    op_func=_dpsk_wo_a_fp8_bmm,
    fake_impl=_dpsk_wo_a_fp8_bmm_fake,
)


def enable_dpsk_wo_a_fp8(*roots: nn.Module | None) -> int:
    """Attach (e4m3 weight, E8M0 scale) to every DSV4.1 attention whose wo_a was
    dequantized to bf16 at load; dsv41_o_proj then routes M<=MAX_M through the
    fp8 skinny kernel. Exact repack is verified; numerics checked vs torch.bmm."""
    n, seen = 0, set()
    for root in roots:
        if root is None:
            continue
        for m in root.modules():
            if id(m) in seen:
                continue
            seen.add(id(m))
            wo_a = getattr(m, "wo_a", None)
            if wo_a is None or not hasattr(m, "n_local_groups"):
                continue
            w = getattr(wo_a, "weight", None)
            sc = getattr(wo_a, "weight_scale", None)
            G = m.n_local_groups
            if (w is None or sc is None or w.dtype != torch.bfloat16 or w.dim() != 2
                    or w.shape[0] % G or w.shape[1] % 512
                    or (w.shape[0] // G) % 32
                    or sc.dim() != 2 or sc.shape[0] < w.shape[0]
                    or sc.shape[1] < w.shape[1] // 32
                    or getattr(m, "_einsum_recipe", (0, 0, 32))[2] != 32):
                logger.warning("DPSK_WO_A_FP8_SKINNY: skip %s (w=%s %s scale=%s %s)",
                               type(m).__name__, None if w is None else w.dtype,
                               None if w is None else tuple(w.shape),
                               None if sc is None else sc.dtype,
                               None if sc is None else tuple(sc.shape))
                continue
            r = repack_wo_a_fp8(w.data, sc.data)
            if r is None:
                logger.warning("DPSK_WO_A_FP8_SKINNY: inexact repack on %s; skip",
                               type(m).__name__)
                continue
            w8, s8 = r
            if n == 0:  # JIT all buckets eagerly + numeric check vs bf16 bmm
                K = w.shape[1]
                for M in (1, 4, 16, 17, 32, 33, 64):
                    x = torch.randn(G, M, K, device=w.device,
                                    dtype=torch.bfloat16).transpose(0, 1)
                    y = wo_a_fp8_skinny(x, w8, s8, G)
                    ref = torch.empty_like(y)
                    torch.bmm(x.transpose(0, 1),
                              w.view(G, -1, K).transpose(1, 2),
                              out=ref.transpose(0, 1))
                    err = (y.float() - ref.float()).abs().max().item()
                    scale = ref.float().abs().max().item() + 1e-6
                    if not err <= 1e-2 * scale:
                        logger.warning("DPSK_WO_A_FP8_SKINNY: M=%d mismatch "
                                       "err=%g scale=%g; disabled", M, err, scale)
                        return 0
            m._wo_a_fp8 = (w8, s8)
            n += 1
    logger.warning("DPSK_WO_A_FP8_SKINNY: enabled on %d attention layer(s), "
                   "max_m=%d", n, WO_A_FP8_MAX_M)
    return n
