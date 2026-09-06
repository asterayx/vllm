# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 vision tower (ViT + aligner), replicated (no TP/DP sharding).

Ported from the official reference implementation
(deepseek-ai/DeepSeek-V4-Flash-Vision-Exp). Weight names match the HF
checkpoint so no renaming is needed at load time.
"""

from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)


@lru_cache(64)
def get_vision_cos_sin(
    n_h: int, n_w: int, dim: int, theta: float, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    if device is not None:
        cos, sin = get_vision_cos_sin(n_h, n_w, dim, theta)
        return cos.to(device=device), sin.to(device=device)
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float()
    freqs = (freqs * inv_freq).flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)


class DeepseekV4RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


class DeepseekV4PatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Linear(3 * config.vision_patch_size**2, config.vision_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.flatten(-3))


def log_vision_sdpa_backends_once(device: torch.device) -> None:
    """Report which fused SDPA kernels the vision tower can use on ``device``.

    SM121 (GB10) may lack a flash build; falling back to the math backend
    costs ``heads * N^2 * 4`` bytes per image, which shows up as encoder
    latency and peak memory. Logged once so the operator can check it.
    """
    if device.type != "cuda":
        return
    from torch.nn.attention import SDPBackend, sdpa_kernel

    q = torch.randn(1, 4, 64, 64, dtype=torch.bfloat16, device=device)
    status = []
    for name, backend in (
        ("flash", SDPBackend.FLASH_ATTENTION),
        ("mem_efficient", SDPBackend.EFFICIENT_ATTENTION),
    ):
        try:
            with sdpa_kernel(backend):
                F.scaled_dot_product_attention(q, q, q)
            status.append(f"{name}=ok")
        except Exception:  # noqa: BLE001 - probe only
            status.append(f"{name}=unavailable")
    logger.info_once(
        "DeepSeek-V4 vision SDPA backends on %s: %s", device, ", ".join(status)
    )


class DeepseekV4VisionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.vision_n_heads
        self.head_dim = config.vision_dim // config.vision_n_heads
        self.wqkv = nn.Linear(config.vision_dim, 3 * config.vision_dim)
        self.wo = nn.Linear(config.vision_dim, config.vision_dim)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        original_shape = x.shape
        if x.ndim == 2:
            x = x.unsqueeze(0)
        q, k, v = (
            t.view(*x.shape[:-1], self.n_heads, self.head_dim)
            for t in self.wqkv(x).chunk(3, dim=-1)
        )
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        o = F.scaled_dot_product_attention(
            q.transpose(-3, -2),
            k.transpose(-3, -2),
            v.transpose(-3, -2),
            attn_mask=attn_mask,
        )
        return self.wo(o.transpose(-3, -2).reshape(original_shape))


class DeepseekV4VisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.vision_dim, 2 * config.vision_inter_dim, bias=False)
        self.w2 = nn.Linear(config.vision_inter_dim, config.vision_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class DeepseekV4VisionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = DeepseekV4RMSNorm(config.vision_dim)
        self.attn = DeepseekV4VisionAttention(config)
        self.norm2 = DeepseekV4RMSNorm(config.vision_dim)
        self.mlp = DeepseekV4VisionMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin, attn_mask)
        return x + self.mlp(self.norm2(x))


class DeepseekV4ViT(nn.Module):
    """DeepSeek-V4 ViT: full bidirectional attention per image, 2D RoPE."""

    def __init__(self, config):
        super().__init__()
        self.rope_dim = config.vision_dim // config.vision_n_heads // 2
        self.rope_theta = config.vision_rope_theta
        self.patch_embed = DeepseekV4PatchEmbed(config)
        self.blocks = nn.ModuleList(
            [DeepseekV4VisionBlock(config) for _ in range(config.vision_n_layers)]
        )
        self.norm = DeepseekV4RMSNorm(config.vision_dim)
        # Plain list: compiled wrappers share the blocks' parameters and must
        # not register as extra submodules (weight names stay unchanged).
        self._compiled_blocks: list[nn.Module] | None = None
        if envs.VLLM_DSV4_VISION_COMPILE:
            self._compiled_blocks = [
                torch.compile(block, dynamic=True) for block in self.blocks
            ]
        self._sdpa_logged = False

    def _run_blocks(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self._sdpa_logged:
            self._sdpa_logged = True
            log_vision_sdpa_backends_once(x.device)
        blocks = self._compiled_blocks or self.blocks
        for block in blocks:
            x = block(x, cos, sin, attn_mask)
        return self.norm(x)

    def forward(
        self, patches: torch.Tensor, n_vit_h: int, n_vit_w: int
    ) -> torch.Tensor:
        x = self.patch_embed(patches)
        cos, sin = get_vision_cos_sin(
            n_vit_h, n_vit_w, self.rope_dim, self.rope_theta, x.device
        )
        return self._run_blocks(x, cos, sin)

    def forward_packed(
        self, patches: torch.Tensor, grids: list[tuple[int, int]]
    ) -> tuple[torch.Tensor, ...]:
        """Encode images of different grids in one pass.

        ``patches`` concatenates the images' patches; each image attends
        only to itself through a block-diagonal mask and keeps its own 2D
        RoPE table. Returns one ``(n_vit_h * n_vit_w, dim)`` tensor per
        image, equal to encoding each image alone.
        """
        x = self.patch_embed(patches)
        tables = [
            get_vision_cos_sin(h, w, self.rope_dim, self.rope_theta, x.device)
            for h, w in grids
        ]
        cos = torch.cat([t[0] for t in tables])
        sin = torch.cat([t[1] for t in tables])
        lens = [h * w for h, w in grids]
        assert sum(lens) == x.shape[0]
        segment = torch.repeat_interleave(torch.arange(len(lens)), torch.tensor(lens))
        segment = segment.to(x.device)
        attn_mask = segment.unsqueeze(0) == segment.unsqueeze(1)
        return self._run_blocks(x, cos, sin, attn_mask).split(lens, dim=0)


class DeepseekV4Aligner(nn.Module):
    """Spatial merge (downsample_ratio x downsample_ratio) + MLP projector."""

    def __init__(self, config):
        super().__init__()
        self.downsample_ratio = config.vision_downsample_ratio
        in_dim = config.vision_dim * self.downsample_ratio**2
        self.w1 = nn.Linear(in_dim, config.hidden_size)
        self.w2 = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x: torch.Tensor, n_vit_h: int, n_vit_w: int) -> torch.Tensor:
        r = self.downsample_ratio
        batch_shape = x.shape[:-2]
        dim = x.shape[-1]
        x = x.reshape(-1, n_vit_h, n_vit_w, dim)
        pad_h, pad_w = -n_vit_h % r, -n_vit_w % r
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        out_h, out_w = (n_vit_h + pad_h) // r, (n_vit_w + pad_w) // r
        # Match unfold's channel, patch-row, patch-column order.
        x = x.reshape(-1, out_h, r, out_w, r, dim).permute(0, 1, 3, 5, 2, 4)
        x = x.reshape(*batch_shape, out_h * out_w, dim * r * r)
        return self.w2(F.gelu(self.w1(x)))
