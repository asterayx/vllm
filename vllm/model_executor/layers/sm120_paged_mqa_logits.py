# SPDX-License-Identifier: Apache-2.0
"""SM120 Triton paged MQA logits for the fp8 indexer cache with block sizes
DeepGEMM's sm120 kernel does not accept (DeepSeek-V4.1 ratio-1 layers: 128-row
pages). Replaces the fp32 torch reference (dequant + einsum simt sgemm).

Page layout (segregated, same as `_paged_mqa_logits_ref_fp8`):
    page p at byte offset p * page_stride:
        [bs * D fp8 e4m3 values | bs fp32 scales]

logits[b*N + n, t] = scale[t] * sum_h w[b,n,h] * relu(q[b,n,h,:] . k[t,:])
for t < context_lens[b, n], else -inf.  Output: [B*N, max_model_len] fp32.
Gated by DPSK_INDEXER_SM120_FIX=1. No host syncs (graph/capture friendly).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _sm120_paged_mqa_logits_kernel(
    q_ptr,  # fp8 [B, N, H, D] contiguous
    kv_ptr,  # uint8 base of kv cache
    w_ptr,  # [B*N, H] (any float dtype)
    cl_ptr,  # int32 [B, CLN] context lens
    bt_ptr,  # int32 [B, BTW] block table
    out_ptr,  # fp32 [B*N, L]
    page_stride,
    nb_total,
    bt_stride,
    bt_width,
    cl_stride,
    w_stride,
    out_stride,
    L,
    N: tl.constexpr,
    CLN: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BS: tl.constexpr,
    BT: tl.constexpr,
    FP8_DOT: tl.constexpr,
):
    b = tl.program_id(0)
    tile = tl.program_id(1)
    t0 = tile * BT
    offs_t = t0 + tl.arange(0, BT)
    tmask = offs_t < L
    neg_inf = tl.full([BT], float("-inf"), tl.float32)

    # max context over the N rows of this request
    maxcl = 0
    for n in tl.static_range(N):
        if CLN == 1:
            c = tl.load(cl_ptr + b * cl_stride)
        else:
            c = tl.load(cl_ptr + b * cl_stride + n)
        maxcl = tl.maximum(maxcl, c)

    blk = t0 // BS
    if (t0 >= maxcl) or (blk >= bt_width):
        for n in tl.static_range(N):
            tl.store(out_ptr + (b * N + n).to(tl.int64) * out_stride + offs_t,
                     neg_inf, mask=tmask)
        return

    page = tl.load(bt_ptr + b * bt_stride + blk)
    page = tl.minimum(tl.maximum(page, 0), nb_total - 1).to(tl.int64)
    base = kv_ptr + page * page_stride
    row0 = t0 % BS
    offs_r = row0 + tl.arange(0, BT)
    offs_d = tl.arange(0, D)
    k = tl.load(base + offs_r[:, None] * D + offs_d[None, :])  # uint8 [BT, D]
    k = k.to(tl.float8e4nv, bitcast=True)
    sc = tl.load((base + BS * D).to(tl.pointer_type(tl.float32)) + offs_r)  # [BT]
    if not FP8_DOT:
        k = k.to(tl.bfloat16)
    offs_h = tl.arange(0, H)
    for n in tl.static_range(N):
        row = b * N + n
        q = tl.load(q_ptr + (row * H + offs_h[:, None]).to(tl.int64) * D
                    + offs_d[None, :])  # fp8 [H, D]
        if not FP8_DOT:
            q = q.to(tl.bfloat16)
        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32)  # [H, BT]
        s = tl.maximum(s, 0.0)
        w = tl.load(w_ptr + row * w_stride + offs_h).to(tl.float32)  # [H]
        lg = tl.sum(s * w[:, None], axis=0) * sc  # [BT]
        if CLN == 1:
            c = tl.load(cl_ptr + b * cl_stride)
        else:
            c = tl.load(cl_ptr + b * cl_stride + n)
        lg = tl.where(offs_t < c, lg, neg_inf)
        tl.store(out_ptr + row.to(tl.int64) * out_stride + offs_t, lg, mask=tmask)


_FP8_DOT = True


def sm120_paged_mqa_logits_fp8(
    q_fp8: torch.Tensor,  # [B, N, H, D] fp8
    kv_cache: torch.Tensor,  # [nb, bs, ...] uint8 (segregated page layout)
    weights: torch.Tensor,  # [B*N, H]
    context_lens: torch.Tensor,  # [B, N] or [B, 1] / [B]
    block_table: torch.Tensor,  # [>=B, W] int32
    max_model_len: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    B, N, H, D = q_fp8.shape
    bs = kv_cache.shape[1]
    if out is None:
        out = torch.empty((B * N, max_model_len), dtype=torch.float32,
                          device=q_fp8.device)
    if B == 0:
        return out
    q = q_fp8.contiguous()
    w = weights.reshape(B * N, H)
    if w.stride(1) != 1:
        w = w.contiguous()
    cl = context_lens.reshape(B, -1)
    if cl.dtype != torch.int32:
        cl = cl.to(torch.int32)
    if cl.stride(1) != 1 and cl.shape[1] > 1:
        cl = cl.contiguous()
    CLN = cl.shape[1]
    assert CLN in (1, N), (cl.shape, N)
    bt = block_table
    if bt.dtype != torch.int32:
        bt = bt.to(torch.int32)
    BT = min(bs, 128)
    assert bs % BT == 0
    grid = (B, triton.cdiv(max_model_len, BT))
    _sm120_paged_mqa_logits_kernel[grid](
        q, kv_cache, w, cl, bt, out,
        int(kv_cache.stride(0)), kv_cache.shape[0], bt.stride(0), bt.shape[1],
        cl.stride(0), w.stride(0), out.stride(0), max_model_len,
        N=N, CLN=CLN, H=H, D=D, BS=bs, BT=BT, FP8_DOT=_FP8_DOT,
        num_warps=4, num_stages=1,
    )
    return out
