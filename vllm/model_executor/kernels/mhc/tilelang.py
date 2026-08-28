# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

# Size-16 MHC dummies captured on GB10; size 8 then IMA'd in the same
# compiled TileLang kernel (dynamic num_tokens). Pad small batches up.
SM12X_MHC_MIN_TOKENS = 16


def sm12x_mhc_min_tokens() -> int:
    """Pad MHC token dim to this on SM12x. 0 means no pad."""
    if current_platform.is_device_capability_family(120):
        return SM12X_MHC_MIN_TOKENS
    return 0


def _pad_token_rows(t: torch.Tensor, target: int) -> torch.Tensor:
    n = t.shape[0]
    if n >= target:
        return t
    return torch.cat((t, t.new_zeros((target - n, *t.shape[1:]))), dim=0)


def _sm12x_pad_token_tensors(
    *tensors: torch.Tensor,
) -> tuple[list[torch.Tensor], int]:
    orig = tensors[0].shape[0]
    target = sm12x_mhc_min_tokens()
    if not target or orig >= target:
        return list(tensors), orig
    logger.info_once(
        "SM12x MHC: padding token dim %d -> %d to avoid small-batch IMA",
        orig,
        target,
    )
    return [_pad_token_rows(t, target) for t in tensors], orig


def mhc_pdl_enabled() -> bool:
    """TileLang MHC PDL is unsafe on SM12x (GB10 / SM121 IMA during capture)."""
    return (
        current_platform.is_arch_support_pdl()
        and current_platform.is_cuda()
        and not current_platform.is_device_capability_family(120)
    )


def use_mhc_small_fma(num_tokens: int) -> bool:
    """Hopper small-batch fused FMA (n_splits=4/8). Disabled on SM12x.

    That path first compiles ``mhc_pre_big_fuse_with_norm`` at 16 tokens
    (n_splits=4) and would recompile at 4 tokens (n_splits=8). SM121 IMA'd
    during PIECEWISE dummy capture of those sizes; stay on n_splits=1.
    """
    if current_platform.is_device_capability_family(120):
        return False
    return num_tokens <= 16


def _torch_hc_prenorm_gemm(
    x: torch.Tensor,
    fn: torch.Tensor,
    out: torch.Tensor,
    sqrsum: torch.Tensor,
) -> None:
    assert out.shape[0] == 1
    assert sqrsum.shape[0] == 1
    x_float = x.float()
    out[0].copy_(x_float @ fn.t())
    sqrsum[0].copy_(x_float.square().sum(dim=-1))


def _tilelang_hc_prenorm_gemm(
    x: torch.Tensor,
    fn: torch.Tensor,
    out: torch.Tensor,
    sqrsum: torch.Tensor,
    hidden_size: int,
    hc_mult: int,
    tile_n: int = 12,
    n_thr: int = 512,
    n_splits: int = 1,
) -> None:
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        hc_prenorm_gemm_block_m_tilelang,
        hc_prenorm_gemm_tilelang,
    )

    assert out.shape[0] == n_splits
    assert sqrsum.shape[0] == n_splits
    assert x.shape[1] == hc_mult * hidden_size
    assert x.shape[1] % n_splits == 0
    assert (x.shape[1] // n_splits) % n_thr == 0
    use_default_config = tile_n == 12 and n_thr == 512
    if n_splits == 1 and use_default_config and x.shape[0] >= 1024:
        hc_prenorm_gemm_block_m_tilelang(
            x,
            fn,
            out,
            sqrsum,
            hidden_size,
            hc_mult,
            fn.shape[0],
            n_thr,
            tile_n,
            2,
        )
        return
    if (
        n_splits == 1
        and use_default_config
        and x.shape[0] < 128
        and x.shape[1] % 1024 == 0
    ):
        hc_prenorm_gemm_tilelang(
            x,
            fn,
            out,
            sqrsum,
            hidden_size,
            hc_mult,
            fn.shape[0],
            1024,
            4,
            n_splits,
        )
        return
    hc_prenorm_gemm_tilelang(
        x,
        fn,
        out,
        sqrsum,
        hidden_size,
        hc_mult,
        fn.shape[0],
        n_thr,
        tile_n,
        n_splits,
    )


def mhc_pre_tilelang(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward pass for mHC pre block.

    Args:
        residual: shape (..., hc_mult, hidden_size), dtype torch.bfloat16
        fn: shape (hc_mult3, hc_mult * hidden_size), dtype torch.float32
        hc_scale: shape (3,), dtype torch.float32
        hc_base: shape (hc_mult3,), dtype torch.float32
        rms_eps: RMS normalization epsilon
        hc_pre_eps: pre-mix epsilon
        hc_sinkhorn_eps: sinkhorn epsilon
        hc_post_mult_value: post-mix multiplier value
        sinkhorn_repeat: number of sinkhorn iterations
        n_splits: split-k factor;
        norm_weight: optional RMSNorm weight, shape (hidden_size,), dtype
            torch.bfloat16. When provided, RMSNorm is fused into the
            layer_input write path of the big_fuse kernel.
        norm_eps: epsilon for the fused RMSNorm; only consulted when
            norm_weight is given.

    Returns:
        post_mix: shape (..., hc_mult), dtype torch.float32
        comb_mix: shape (..., hc_mult, hc_mult), dtype torch.float32
        layer_input: shape (..., hidden_size), dtype torch.bfloat16
    """
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        compute_num_split,
        mhc_pre_big_fuse_tilelang,
        mhc_pre_big_fuse_with_norm_tilelang,
    )
    from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm
    from vllm.utils.math_utils import cdiv

    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2

    hc_hidden_size = hc_mult * hidden_size
    assert fn.shape[0] == hc_mult3
    assert fn.shape[1] == hc_hidden_size
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc_mult3,)

    if norm_weight is not None:
        assert norm_weight.shape == (hidden_size,)
        if norm_weight.dtype != torch.bfloat16:
            norm_weight = norm_weight.to(torch.bfloat16)
        if not norm_weight.is_contiguous():
            norm_weight = norm_weight.contiguous()

    outer_shape = residual.shape[:-2]

    residual_flat = residual.view(-1, hc_mult, hidden_size)
    padded, orig_tokens = _sm12x_pad_token_tensors(residual_flat)
    residual_flat = padded[0]
    num_tokens = residual_flat.shape[0]

    from vllm.utils.deep_gemm import is_deep_gemm_supported

    use_deep_gemm = is_deep_gemm_supported()
    if use_deep_gemm:
        # these numbers are from deepgemm kernel impl
        block_k = 64
        block_m = 64
        n_splits = compute_num_split(block_k, hc_hidden_size, cdiv(num_tokens, block_m))
    else:
        n_splits = 1

    post_mix = torch.empty(
        num_tokens, hc_mult, dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        num_tokens, hc_mult2, dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
    )

    gemm_out_mul = torch.empty(
        n_splits, num_tokens, hc_mult3, dtype=torch.float32, device=residual.device
    )
    gemm_out_sqrsum = torch.empty(
        n_splits, num_tokens, dtype=torch.float32, device=residual.device
    )

    residual_2d = residual_flat.view(num_tokens, hc_mult * hidden_size)
    if use_deep_gemm:
        tf32_hc_prenorm_gemm(
            residual_2d,
            fn,
            gemm_out_mul,
            gemm_out_sqrsum,
            n_splits,
        )
    else:
        _tilelang_hc_prenorm_gemm(
            residual_2d,
            fn,
            gemm_out_mul,
            gemm_out_sqrsum,
            hidden_size,
            hc_mult,
        )

    if norm_weight is None:
        mhc_pre_big_fuse_tilelang(
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual_flat,
            post_mix,
            comb_mix,
            layer_input,
            hidden_size,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            hc_mult,
        )
    else:
        mhc_pre_big_fuse_with_norm_tilelang(
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual_flat,
            post_mix,
            comb_mix,
            layer_input,
            norm_weight,
            hidden_size,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            norm_eps,
            n_splits,
            hc_mult,
        )

    return (
        post_mix[:orig_tokens].view(*outer_shape, hc_mult, 1),
        comb_mix[:orig_tokens].view(*outer_shape, hc_mult, hc_mult),
        layer_input[:orig_tokens].view(*outer_shape, hidden_size),
    )


def _mhc_pre_tilelang_fake(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]

    # Create empty tensors with correct shapes for meta device / shape inference
    post_mix = torch.empty(
        *outer_shape,
        hc_mult,
        1,
        dtype=torch.float32,
        device=residual.device,
    )
    comb_mix = torch.empty(
        *outer_shape,
        hc_mult,
        hc_mult,
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input = torch.empty(
        *outer_shape,
        hidden_size,
        dtype=torch.bfloat16,
        device=residual.device,
    )

    return post_mix, comb_mix, layer_input


def mhc_pre_broadcast_tilelang(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
    fn_broadcast: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """First-layer mHC pre for a residual broadcast from ``(T, H)``."""
    assert norm_weight is not None, "broadcast mHC pre currently requires fused RMSNorm"
    assert residual.dtype == torch.bfloat16
    assert residual.dim() == 2
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    hidden_size = residual.shape[-1]
    hc_mult = fn.shape[1] // hidden_size
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2
    assert fn.shape == (hc_mult3, hc_mult * hidden_size)
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc_mult3,)
    assert fn_broadcast is not None
    assert fn_broadcast.dtype == torch.float32
    assert fn_broadcast.shape == (hc_mult3, hidden_size)

    if norm_weight.dtype != torch.bfloat16:
        norm_weight = norm_weight.to(torch.bfloat16)
    if not norm_weight.is_contiguous():
        norm_weight = norm_weight.contiguous()

    padded, orig_tokens = _sm12x_pad_token_tensors(residual)
    residual_flat = padded[0]
    num_tokens = residual_flat.shape[0]

    from vllm.utils.deep_gemm import is_deep_gemm_supported

    use_deep_gemm = is_deep_gemm_supported()
    if use_deep_gemm:
        from vllm.model_executor.kernels.mhc.tilelang_kernels import compute_num_split
        from vllm.utils.math_utils import cdiv

        n_splits = compute_num_split(64, hidden_size, cdiv(num_tokens, 64))
    else:
        # Stock SM12x / GB10 wheels often ship without a working DeepGEMM
        # prenorm kernel. Keep n_splits=1 so the torch fallback matches
        # tf32_hc_prenorm_gemm's documented math.
        n_splits = 1

    residual_out = torch.empty(
        num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16, device=residual.device
    )
    post_mix = torch.empty(
        num_tokens, hc_mult, dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        num_tokens, hc_mult2, dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
    )
    gemm_out_mul = torch.empty(
        n_splits, num_tokens, hc_mult3, dtype=torch.float32, device=residual.device
    )
    gemm_out_sqrsum = torch.empty(
        n_splits, num_tokens, dtype=torch.float32, device=residual.device
    )

    if use_deep_gemm:
        from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm

        tf32_hc_prenorm_gemm(
            residual_flat,
            fn_broadcast,
            gemm_out_mul,
            gemm_out_sqrsum,
            n_splits,
        )
    else:
        _torch_hc_prenorm_gemm(
            residual_flat,
            fn_broadcast,
            gemm_out_mul,
            gemm_out_sqrsum,
        )
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        mhc_pre_big_fuse_broadcast_with_norm_tilelang,
    )

    mhc_pre_big_fuse_broadcast_with_norm_tilelang(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual_flat,
        residual_out,
        post_mix,
        comb_mix,
        layer_input,
        norm_weight,
        hidden_size,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        norm_eps,
        n_splits,
        hc_mult,
    )
    return (
        residual_out[:orig_tokens],
        post_mix[:orig_tokens].unsqueeze(-1),
        comb_mix[:orig_tokens].view(orig_tokens, hc_mult, hc_mult),
        layer_input[:orig_tokens],
    )


def mhc_post_tilelang(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        mhc_post_tilelang as _mhc_post_kernel,
    )

    padded, orig_tokens = _sm12x_pad_token_tensors(
        x, residual, post_layer_mix, comb_res_mix
    )
    x, residual, post_layer_mix, comb_res_mix = padded
    out = torch.empty_like(residual)
    _mhc_post_kernel(
        comb_res_mix,
        residual,
        post_layer_mix.squeeze(-1),
        x,
        out,
        residual.shape[-2],
        residual.shape[-1],
    )
    return out[:orig_tokens]


def mhc_fused_post_pre_tilelang(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    tile_n: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run one MHC post block followed by the next MHC pre block.

    When ``norm_weight`` is provided, the layer_input_cur output is the
    RMSNorm'd activation (fused into the kernel); otherwise it is the
    raw pre-norm activation as before.

    Returns:
        residual_cur: post-mapped residual, shape (..., hc_mult, hidden_size)
        post_mix_cur: shape (..., hc_mult, 1)
        comb_mix_cur: shape (..., hc_mult, hc_mult)
        layer_input_cur: shape (..., hidden_size)
    """

    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        compute_num_split,
        mhc_fused_tilelang,
        mhc_post_tilelang,
        mhc_pre_big_fuse_tilelang,
        mhc_pre_big_fuse_with_norm_tilelang,
    )
    from vllm.utils.math_utils import cdiv

    assert residual.dtype == torch.bfloat16
    assert x.dtype == torch.bfloat16
    assert post_layer_mix.dtype == torch.float32
    assert comb_res_mix.dtype == torch.float32
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2
    hc_hidden_size = hc_mult * hidden_size
    outer_shape = residual.shape[:-2]

    assert x.shape == (*outer_shape, hidden_size)
    assert post_layer_mix.shape in (
        (*outer_shape, hc_mult, 1),
        (*outer_shape, hc_mult),
    )
    assert comb_res_mix.shape == (*outer_shape, hc_mult, hc_mult)
    assert fn.shape == (hc_mult3, hc_hidden_size)
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc_mult3,)

    if norm_weight is not None:
        assert norm_weight.shape == (hidden_size,)
        if norm_weight.dtype != torch.bfloat16:
            norm_weight = norm_weight.to(torch.bfloat16)
        if not norm_weight.is_contiguous():
            norm_weight = norm_weight.contiguous()

    assert n_splits in (1, 2, 4, 8)
    assert hidden_size % n_splits == 0

    residual_flat = residual.view(-1, hc_mult, hidden_size)
    orig_tokens = residual_flat.shape[0]
    x_flat = x.reshape(orig_tokens, hidden_size)
    post_layer_mix_flat = post_layer_mix.reshape(orig_tokens, hc_mult)
    comb_res_mix_flat = comb_res_mix.reshape(orig_tokens, hc_mult, hc_mult)
    padded, orig_tokens = _sm12x_pad_token_tensors(
        residual_flat, x_flat, post_layer_mix_flat, comb_res_mix_flat
    )
    residual_flat, x_flat, post_layer_mix_flat, comb_res_mix_flat = padded
    num_tokens = residual_flat.shape[0]

    from vllm.utils.deep_gemm import is_deep_gemm_supported

    use_deep_gemm = is_deep_gemm_supported()
    use_small_fma = use_mhc_small_fma(num_tokens)
    if use_small_fma:
        # TODO(gnovack): investigate autotuning these heuristics
        tile_n = 2 if num_tokens < 8 else 3
        n_splits = 8 if (num_tokens < 8 and hidden_size <= 4096) else 4
    else:
        if use_deep_gemm:
            # these number are from deepgemm kernel impl
            block_k = 64
            block_m = 64
            n_splits = compute_num_split(
                block_k, hc_hidden_size, cdiv(num_tokens, block_m)
            )
        else:
            n_splits = 1

    gemm_out_mul = torch.empty(
        n_splits,
        num_tokens,
        hc_mult3,
        dtype=torch.float32,
        device=residual.device,
    )
    gemm_out_sqrsum = torch.empty(
        n_splits,
        num_tokens,
        dtype=torch.float32,
        device=residual.device,
    )
    residual_cur = torch.empty_like(residual_flat)
    post_mix_cur = torch.empty(
        num_tokens,
        hc_mult,
        dtype=torch.float32,
        device=residual.device,
    )
    comb_mix_cur = torch.empty(
        num_tokens,
        hc_mult2,
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input_cur = torch.empty(
        num_tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device=residual.device,
    )

    if use_small_fma:
        mhc_fused_tilelang(
            comb_res_mix_flat,
            residual_flat,
            post_layer_mix_flat,
            x_flat,
            fn.view(hc_mult3, hc_mult, hidden_size),
            gemm_out_mul,
            gemm_out_sqrsum,
            residual_cur,
            hc_mult,
            hidden_size,
            hc_mult3,
            tile_n=tile_n,
            n_splits=n_splits,
        )
    else:
        mhc_post_tilelang(
            comb_res_mix_flat,
            residual_flat,
            post_layer_mix_flat,
            x_flat,
            residual_cur,
            residual.shape[-2],
            residual.shape[-1],
        )

        residual_cur_2d = residual_cur.view(num_tokens, hc_mult * hidden_size)
        if use_deep_gemm:
            from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm

            tf32_hc_prenorm_gemm(
                residual_cur_2d,
                fn,
                gemm_out_mul,
                gemm_out_sqrsum,
                n_splits,
            )
        else:
            _tilelang_hc_prenorm_gemm(
                residual_cur_2d,
                fn,
                gemm_out_mul,
                gemm_out_sqrsum,
                hidden_size,
                hc_mult,
            )

    if norm_weight is None:
        mhc_pre_big_fuse_tilelang(
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual_cur,
            post_mix_cur,
            comb_mix_cur,
            layer_input_cur,
            hidden_size,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            hc_mult,
        )
    else:
        mhc_pre_big_fuse_with_norm_tilelang(
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual_cur,
            post_mix_cur,
            comb_mix_cur,
            layer_input_cur,
            norm_weight,
            hidden_size,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            norm_eps,
            n_splits,
            hc_mult,
        )

    return (
        residual_cur[:orig_tokens].view(*outer_shape, hc_mult, hidden_size),
        post_mix_cur[:orig_tokens].view(*outer_shape, hc_mult, 1),
        comb_mix_cur[:orig_tokens].view(*outer_shape, hc_mult, hc_mult),
        layer_input_cur[:orig_tokens].view(*outer_shape, hidden_size),
    )


def _mhc_fused_post_pre_tilelang_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    tile_n: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]

    residual_cur = torch.empty_like(residual)
    post_mix_cur = torch.empty(
        *outer_shape,
        hc_mult,
        1,
        dtype=torch.float32,
        device=residual.device,
    )
    comb_mix_cur = torch.empty(
        *outer_shape,
        hc_mult,
        hc_mult,
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input_cur = torch.empty(
        *outer_shape,
        hidden_size,
        dtype=torch.bfloat16,
        device=residual.device,
    )

    return residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur


def _mhc_post_tilelang_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(residual)


def hc_head_fused_kernel_tilelang(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    """Apply the fused hc_head kernel and return the (T, H) bf16 result."""
    padded, orig_tokens = _sm12x_pad_token_tensors(hs_flat)
    hs_flat = padded[0]
    num_tokens, hc_mult, hidden_size = hs_flat.shape
    out = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=hs_flat.device
    )
    if orig_tokens == 0:
        return out[:orig_tokens]
    from vllm.model_executor.kernels.mhc.tilelang_kernels import hc_head_fuse_tilelang

    hc_head_fuse_tilelang(
        hs_flat,
        fn,
        hc_scale,
        hc_base,
        out,
        hidden_size,
        rms_eps,
        hc_eps,
        hc_mult,
    )
    return out[:orig_tokens]


def _hc_head_fused_kernel_tilelang_fake(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    num_tokens, _, hidden_size = hs_flat.shape
    return torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=hs_flat.device
    )


direct_register_custom_op(
    op_name="mhc_pre_tilelang",
    op_func=mhc_pre_tilelang,
    mutates_args=[],
    fake_impl=_mhc_pre_tilelang_fake,
)
direct_register_custom_op(
    op_name="mhc_post_tilelang",
    op_func=mhc_post_tilelang,
    mutates_args=[],
    fake_impl=_mhc_post_tilelang_fake,
)

direct_register_custom_op(
    op_name="mhc_fused_post_pre_tilelang",
    op_func=mhc_fused_post_pre_tilelang,
    mutates_args=[],
    fake_impl=_mhc_fused_post_pre_tilelang_fake,
)

direct_register_custom_op(
    op_name="hc_head_fused_kernel_tilelang",
    op_func=hc_head_fused_kernel_tilelang,
    mutates_args=[],
    fake_impl=_hc_head_fused_kernel_tilelang_fake,
)
