# SPDX-License-Identifier: Apache-2.0
"""DPSK_FP8_SKINNY_LINEAR (try51): SM120 fp8 skinny GEMM for MXFP8 linears.

On SM120 the MXFP8 linears (attn fused_wqa_wkv / wq_b / wo_b / indexer wq_b,
shared experts, ...) go through EmulationMxfp8LinearKernel: dequant to bf16 at
load + F.linear, which cuBLAS serves badly for decode M<=64 (~20-30us each).
Same idea as wo_a_fp8_skinny: keep an exact (e4m3, E8M0) repack next to the
bf16 weight and stream 1-byte weights through the Triton skinny kernel (reused
with G=1) for small M, split-K (fp32 atomics) to fill the 110 SMs. Per shape a
load-time timing check vs F.linear picks the largest M bucket where fp8 wins;
larger M (prefill) keeps the bf16 path untouched.
"""

import logging
import os
import re

import torch
import triton
from torch import nn

from vllm.models.deepseek_v41.nvidia.ops.wo_a_fp8_skinny import (
    _wo_a_fp8_skinny_kernel,
    repack_wo_a_fp8,
)
from vllm.utils.torch_utils import direct_register_custom_op

logger = logging.getLogger(__name__)

MAX_M = int(os.environ.get("DPSK_FP8_SKINNY_LINEAR_MAX_M", "64"))
INCLUDE = os.environ.get("DPSK_FP8_SKINNY_LINEAR_INCLUDE", "")
EXCLUDE = os.environ.get("DPSK_FP8_SKINNY_LINEAR_EXCLUDE", r"(^|\.)wo_a$")

# (N, K) -> max M where fp8 path is used (0 = never); filled at enable time.
_DBG = os.environ.get("DPSK_FP8_SKINNY_LINEAR_DEBUG", "0") == "1"
_DBG_SEEN: set = set()
_SHAPE_MAXM: dict[tuple[int, int], int] = {}


def _cfg(M: int, N: int, K: int):
    bm = max(16, triton.next_power_of_2(M))
    bn = 64 if N % 64 == 0 else 32
    bk = 128 if bm <= 16 and K >= 4096 else 64
    sk = 1
    for cand in (1, 2, 3, 4, 6, 8, 9, 12, 16):
        if K % cand or (K // cand) % bk:
            continue
        sk = cand
        if (N // bn) * cand >= 200:
            break
    return dict(
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, SPLIT_K=sk, num_stages=3, num_warps=4
    )


def fp8_skinny_linear(
    x: torch.Tensor, w8: torch.Tensor, s8: torch.Tensor, cfg: dict | None = None
) -> torch.Tensor:
    M, K = x.shape
    N = w8.shape[0]
    cfg = dict(cfg or _cfg(M, N, K))
    sk = cfg.pop("SPLIT_K")
    if sk > 1:
        out = torch.zeros((M, N), device=x.device, dtype=torch.float32)
    else:
        out = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
    _wo_a_fp8_skinny_kernel[(N // cfg["BLOCK_N"], 1, sk)](
        x,
        w8,
        s8,
        out,
        M,
        x.stride(0),
        0,
        x.stride(1),
        out.stride(0),
        0,
        N=N,
        K=K,
        SPLIT_K=sk,
        **cfg,
    )
    return out if sk == 1 else out.to(torch.bfloat16)


def _fi_mxfp8_linear(x, w8, wsc):
    """Original FlashInferCutlassMxfp8LinearKernel path (quant + cutlass)."""
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )
    from vllm.utils import flashinfer as vllm_flashinfer

    xq, xs = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
    return vllm_flashinfer.mm_mxfp8(
        xq, w8.t(), xs, wsc, out_dtype=x.dtype, backend="cutlass"
    )


def _dpsk_fp8_skinny_linear(
    x: torch.Tensor,
    w_bf16: torch.Tensor,
    w8: torch.Tensor,
    s8: torch.Tensor,
    wsc: torch.Tensor,
    kind: int,
) -> torch.Tensor:
    """Kind 0: Emulation layer (fallback bf16 F.linear);
    kind 1: FlashInfer cutlass mxfp8 layer (fallback quant + mm_mxfp8)."""
    M = x.shape[0]
    N, K = w8.shape
    hit = 1 <= M <= _SHAPE_MAXM.get((kind, N, K), 0)
    if _DBG and (kind, N, K, M) not in _DBG_SEEN:
        _DBG_SEEN.add((kind, N, K, M))
        logger.warning(
            "DPSK_FP8_SKINNY_LINEAR dbg: call kind%d N=%d K=%d M=%d skinny=%s",
            kind,
            N,
            K,
            M,
            hit,
        )
    if hit:
        return fp8_skinny_linear(x, w8, s8)
    if kind == 1:
        return _fi_mxfp8_linear(x, w8, wsc)
    return torch.nn.functional.linear(x, w_bf16)


def _dpsk_fp8_skinny_linear_fake(x, w_bf16, w8, s8, wsc, kind):
    return x.new_empty((x.shape[0], w8.shape[0]))


direct_register_custom_op(
    op_name="dpsk_fp8_skinny_linear",
    op_func=_dpsk_fp8_skinny_linear,
    fake_impl=_dpsk_fp8_skinny_linear_fake,
)


def fp8_skinny_apply(layer: nn.Module, x: torch.Tensor):
    """Skinny fp8 path for Mxfp8 apply_weights; returns None when not handled."""
    p = getattr(layer, "_fp8_skinny", None)
    if p is None or not isinstance(x, torch.Tensor) or x.dtype != torch.bfloat16:
        if _DBG and p is not None:
            logger.warning(
                "DPSK_FP8_SKINNY_LINEAR dbg: skip non-bf16 input %s N=%d",
                type(x).__name__,
                layer.weight.shape[0],
            )
        return None
    shp = x.shape
    x2 = x.reshape(-1, shp[-1])
    w8, s8, kind = p
    if kind == 1:
        y = torch.ops.vllm.dpsk_fp8_skinny_linear(x2, w8, w8, s8, layer.weight_scale, 1)
    else:
        y = torch.ops.vllm.dpsk_fp8_skinny_linear(x2, layer.weight, w8, s8, s8, 0)
    return y.view(*shp[:-1], y.shape[-1])


def _time(fn, reps=20):
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn()
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            fn()
    torch.cuda.synchronize()
    g.replay()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(reps):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / reps


def _calibrate(kind, N, K, layers):
    """Numeric check + timing on (up to 6, L2-rotated) real layers of shape."""
    # layers: list of (ref_fn(x)->y, w8, s8)
    ws = layers[:6]
    dev = ws[0][1].device
    maxm = 0
    for M in (1, 4, 16, 17, 32, 33, 64):
        if M > MAX_M:
            break
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        for rf, w8, s8 in ws[:2]:
            y = fp8_skinny_linear(x, w8, s8).float()
            wf = (
                w8.float().view(N, K // 32, 32)
                * torch.exp2(s8.float() - 127.0).unsqueeze(-1)
            ).view(N, K)
            ref = x.float() @ wf.t()
            cub = rf(x).float()
            e = (y - ref).abs().max().item()
            ec = (cub - ref).abs().max().item()
            sc = ref.abs().max().item() + 1e-6
            if not e <= max(2 * ec, 1e-2 * sc):
                logger.warning(
                    "DPSK_FP8_SKINNY_LINEAR: (%d,%d) M=%d mismatch "
                    "err=%g base_err=%g scale=%g; off",
                    N,
                    K,
                    M,
                    e,
                    ec,
                    sc,
                )
                return 0
        if M in (1, 17, 33):
            continue
        tf = _time(lambda x=x: [fp8_skinny_linear(x, w8, s8) for _, w8, s8 in ws])
        tb = _time(lambda x=x: [rf(x) for rf, _, _ in ws])
        logger.warning(
            "DPSK_FP8_SKINNY_LINEAR: kind%d (N=%d,K=%d) M=%d base %.1fus fp8 %.1fus",
            kind,
            N,
            K,
            M,
            tb * 1e3 / len(ws),
            tf * 1e3 / len(ws),
        )
        if tf < 0.95 * tb:
            maxm = M
        else:
            break
    return maxm


def _agree_on_max_m_across_tp() -> None:
    """Make every TP rank use the same fp8 cutover per shape.

    Timing is per rank, so noise near the threshold could otherwise pick a
    different max-M on different shards of the same layer, which changes
    numerics with batch size. Take the minimum across the TP group. If the
    ranks calibrated different shape sets, disable the path everywhere.
    """
    from vllm.distributed import get_tp_group

    try:
        tp = get_tp_group()
    except AssertionError:
        return
    if tp.world_size == 1:
        return
    keys = sorted(_SHAPE_MAXM)
    sizes = torch.tensor([len(keys), -len(keys)], dtype=torch.int64)
    torch.distributed.all_reduce(
        sizes, op=torch.distributed.ReduceOp.MAX, group=tp.cpu_group
    )
    if int(sizes[0]) != -int(sizes[1]):
        logger.warning(
            "DPSK_FP8_SKINNY_LINEAR: TP ranks calibrated different shape sets; off"
        )
        for key in keys:
            _SHAPE_MAXM[key] = 0
        return
    if not keys:
        return
    values = torch.tensor([_SHAPE_MAXM[k] for k in keys], dtype=torch.int64)
    torch.distributed.all_reduce(
        values, op=torch.distributed.ReduceOp.MIN, group=tp.cpu_group
    )
    for key, value in zip(keys, values.tolist()):
        _SHAPE_MAXM[key] = int(value)


def enable_dpsk_fp8_skinny_linear(*roots: nn.Module | None) -> int:
    from vllm.model_executor.kernels.linear.mxfp8.emulation import (
        EmulationMxfp8LinearKernel,
    )
    from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
        FlashInferCutlassMxfp8LinearKernel,
    )

    inc = re.compile(INCLUDE) if INCLUDE else None
    exc = re.compile(EXCLUDE) if EXCLUDE else None
    by_shape: dict[tuple[int, int, int], list] = {}
    cands = []
    seen = set()
    for root in roots:
        if root is None:
            continue
        for name, m in root.named_modules():
            if id(m) in seen:
                continue
            seen.add(id(m))
            qm = getattr(m, "quant_method", None)
            kern = getattr(qm, "kernel", None)
            if isinstance(kern, EmulationMxfp8LinearKernel):
                kind = 0
            elif type(kern) is FlashInferCutlassMxfp8LinearKernel:
                kind = 1
            else:
                continue
            if (
                getattr(m, "is_bmm", False)
                or (exc and exc.search(name))
                or (inc and not inc.search(name))
            ):
                if kind == 1 and hasattr(m, "_dpsk_scale_2d_u8"):
                    del m._dpsk_scale_2d_u8
                continue
            w = getattr(m, "weight", None)
            if w is None or w.dim() != 2 or w.shape[0] % 32 or w.shape[1] % 64:
                continue
            N, K = w.shape
            if kind == 0:
                sc = getattr(m, "weight_scale", None)
                if w.dtype != torch.bfloat16 or sc is None or sc.dim() != 2:
                    continue
                r = repack_wo_a_fp8(w.data, sc.data)
                if r is None:
                    logger.warning(
                        "DPSK_FP8_SKINNY_LINEAR: inexact repack %s; skip", name
                    )
                    continue
                wb = w.data
                rf = lambda xx, wb=wb: torch.nn.functional.linear(xx, wb)
            else:
                s2 = getattr(m, "_dpsk_scale_2d_u8", None)
                if (
                    w.dtype != torch.float8_e4m3fn
                    or not w.is_contiguous()
                    or s2 is None
                    or tuple(s2.shape) != (N, K // 32)
                ):
                    continue
                r = (w.data, s2)
                rf = lambda xx, w8=w.data, wsc=m.weight_scale.data: _fi_mxfp8_linear(
                    xx, w8, wsc
                )
            key = (kind, N, K)
            by_shape.setdefault(key, []).append((rf, r[0], r[1]))
            cands.append((name, m, r, key))
    for key, lst in by_shape.items():
        _SHAPE_MAXM[key] = _calibrate(key[0], key[1], key[2], lst)
    _agree_on_max_m_across_tp()
    n = 0
    extra = 0
    for name, m, r, key in cands:
        if _SHAPE_MAXM.get(key, 0) > 0:
            m._fp8_skinny = (r[0], r[1], key[0])
            n += 1
            extra += (r[0].numel() if key[0] == 0 else 0) + r[1].numel()
        elif key[0] == 1:
            del m._dpsk_scale_2d_u8
    logger.warning(
        "DPSK_FP8_SKINNY_LINEAR: enabled on %d/%d linear(s), +%.0f MiB, shape max_m=%s",
        n,
        len(cands),
        extra / 2**20,
        {f"k{k[0]}:{k[1]}x{k[2]}": v for k, v in _SHAPE_MAXM.items()},
    )
    return n
