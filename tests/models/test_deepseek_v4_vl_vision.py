# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parity tests for the DeepSeek-V4 vision tower against the official
reference implementation."""

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4.common.vision import (
    DeepseekV4Aligner,
    DeepseekV4VisionAttention,
    DeepseekV4ViT,
    apply_rotary,
    get_vision_cos_sin,
)

REF_VISION_PATH = (
    Path(os.environ.get("VLLM_TEST_DSV4_REFERENCE_DIR", "/tmp/dsv4vis")) / "vision.py"
)


def _load_reference_vision():
    spec = importlib.util.spec_from_file_location("dsv4vis_ref", REF_VISION_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ref_vision = _load_reference_vision() if REF_VISION_PATH.is_file() else None
requires_reference = pytest.mark.skipif(
    ref_vision is None,
    reason="set VLLM_TEST_DSV4_REFERENCE_DIR to the reference checkout",
)


def _make_config() -> SimpleNamespace:
    return SimpleNamespace(
        vision_n_layers=2,
        vision_dim=64,
        vision_n_heads=4,
        vision_inter_dim=88,
        vision_patch_size=14,
        vision_rope_theta=10000.0,
        vision_downsample_ratio=3,
        hidden_size=96,
    )


def _ref_args(config: SimpleNamespace) -> SimpleNamespace:
    args = SimpleNamespace(**vars(config))
    args.dim = config.hidden_size
    return args


@pytest.mark.parametrize("n_vit_h,n_vit_w", [(6, 6), (7, 5), (4, 10)])
@requires_reference
def test_vit_parity(n_vit_h: int, n_vit_w: int):
    assert ref_vision is not None
    torch.manual_seed(0)
    config = _make_config()
    ours = DeepseekV4ViT(config)
    ref = ref_vision.ViT(_ref_args(config))
    ref.load_state_dict(ours.state_dict())
    ours.eval()
    ref.eval()

    n_tokens = n_vit_h * n_vit_w
    patches = torch.randn(
        n_tokens, 3, config.vision_patch_size, config.vision_patch_size
    )
    with torch.no_grad():
        out_ours = ours(patches, n_vit_h, n_vit_w)
        out_ref = ref(patches, n_vit_h, n_vit_w)
    torch.testing.assert_close(out_ours, out_ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("n_vit_h,n_vit_w", [(6, 6), (7, 5), (4, 10)])
@requires_reference
def test_aligner_parity(n_vit_h: int, n_vit_w: int):
    assert ref_vision is not None
    torch.manual_seed(0)
    config = _make_config()
    ours = DeepseekV4Aligner(config)
    ref = ref_vision.Aligner(_ref_args(config))
    ref.load_state_dict(ours.state_dict())
    ours.eval()
    ref.eval()

    n_tokens = n_vit_h * n_vit_w
    x = torch.randn(n_tokens, config.vision_dim)
    with torch.no_grad():
        out_ours = ours(x, n_vit_h, n_vit_w)
        out_ref = ref(x, n_vit_h, n_vit_w)
    assert out_ours.shape == out_ref.shape
    torch.testing.assert_close(out_ours, out_ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("n_vit_h,n_vit_w", [(7, 5), (4, 10)])
def test_aligner_output_grid(n_vit_h: int, n_vit_w: int):
    torch.manual_seed(0)
    config = _make_config()
    aligner = DeepseekV4Aligner(config)
    r = config.vision_downsample_ratio
    n_llm_h = -(-n_vit_h // r)
    n_llm_w = -(-n_vit_w // r)
    x = torch.randn(n_vit_h * n_vit_w, config.vision_dim)
    with torch.no_grad():
        out = aligner(x, n_vit_h, n_vit_w)
    assert out.shape == (n_llm_h * n_llm_w, config.hidden_size)


def test_vision_attention_matches_unbatched_sdpa():
    """Adding the batch dimension must preserve bidirectional attention."""
    torch.manual_seed(0)
    config = _make_config()
    attention = DeepseekV4VisionAttention(config)
    x = torch.randn(35, config.vision_dim)
    cos, sin = get_vision_cos_sin(7, 5, attention.head_dim // 2, 10000.0)
    q, k, v = [
        t.view(35, attention.n_heads, attention.head_dim)
        for t in attention.wqkv(x).chunk(3, dim=-1)
    ]
    expected = torch.nn.functional.scaled_dot_product_attention(
        apply_rotary(q, cos, sin).transpose(0, 1),
        apply_rotary(k, cos, sin).transpose(0, 1),
        v.transpose(0, 1),
    )
    expected = attention.wo(expected.transpose(0, 1).reshape(35, -1))
    torch.testing.assert_close(attention(x, cos, sin), expected)


@pytest.mark.parametrize("grid", [(6, 6), (7, 5)])
def test_batched_images_match_independent_encoding(grid):
    """Batching must preserve image isolation and padded spatial merge order."""
    config = _make_config()
    vision = DeepseekV4ViT(config)
    aligner = DeepseekV4Aligner(config)
    h, w = grid
    patches = torch.randn(2, h * w, 3, 14, 14)
    with torch.no_grad():
        expected = torch.stack(
            [aligner(vision(image, h, w), h, w) for image in patches]
        )
        actual = aligner(vision(patches, h, w), h, w)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_multimodal_batching_preserves_image_and_token_order(batch_size):
    """Consecutive groups split at the cap without mixing grids or permutations."""
    from vllm.models.deepseek_v4.nvidia.vl_model import (
        DeepseekV4ForConditionalGeneration,
    )

    config = _make_config()
    model = DeepseekV4ForConditionalGeneration.__new__(
        DeepseekV4ForConditionalGeneration
    )
    torch.nn.Module.__init__(model)
    model.vision = DeepseekV4ViT(config)
    model.aligner = DeepseekV4Aligner(config)
    model.vision_encoder_batch_size = batch_size
    grids = [(6, 6)] * 3 + [(7, 5), (6, 6)]
    llm_grids = [((h + 2) // 3, (w + 2) // 3) for h, w in grids]
    patches = [torch.randn(h * w, 3, 14, 14) for h, w in grids]
    perms = [torch.randperm(h * w) for h, w in llm_grids]
    with torch.no_grad():
        expected = [
            model.aligner(model.vision(image, h, w), h, w)[perm]
            for image, (h, w), perm in zip(patches, grids, perms)
        ]
        calls: list[tuple[str, int]] = []
        handle = model.vision.register_forward_pre_hook(
            lambda module, args: calls.append(("dense", args[0].shape[0]))
        )
        packed = model.vision.forward_packed

        def _packed(patches_, grids_):
            calls.append(("packed", len(grids_)))
            return packed(patches_, grids_)

        model.vision.forward_packed = _packed
        actual = model.embed_multimodal(
            patches=torch.cat(patches),
            vit_grid=torch.tensor(grids),
            llm_grid=torch.tensor(llm_grids),
            perm=torch.cat(perms),
        )
        handle.remove()
    if batch_size == 1:
        assert calls == [("dense", 1)] * 5
    else:
        # [6x6, 6x6] dense, [6x6, 7x5] packed (mixed grids), [6x6] dense.
        assert calls == [("dense", 2), ("packed", 2), ("dense", 1)]
    assert len(actual) == len(expected)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=1e-5, atol=1e-6)


def test_forward_packed_matches_independent_encoding():
    """Packing images of different grids with a block-diagonal mask must
    equal encoding each image alone (own RoPE table, no cross-attention)."""
    torch.manual_seed(0)
    config = _make_config()
    vision = DeepseekV4ViT(config)
    grids = [(6, 6), (7, 5), (4, 10)]
    patches = [torch.randn(h * w, 3, 14, 14) for h, w in grids]
    with torch.no_grad():
        expected = [vision(p, h, w) for p, (h, w) in zip(patches, grids)]
        actual = vision.forward_packed(torch.cat(patches), grids)
    assert len(actual) == 3
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=1e-5, atol=1e-5)


def test_fold_sentinel_rows_writes_local_vocab_shard():
    """After folding, embedding lookups of the five sentinel ids return the
    learned vectors and embed_input_ids no longer applies the table."""
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
        VocabParallelEmbedding,
        VocabParallelEmbeddingShardIndices,
    )
    from vllm.models.deepseek_v4.common.mm_preprocess import (
        IMAGE_SENTINEL_BASE_ID,
    )
    from vllm.models.deepseek_v4.nvidia.vl_model import (
        DeepseekV4ForConditionalGeneration,
    )

    hidden = 8
    # Shard covering [IMAGE_SENTINEL_BASE_ID - 2, IMAGE_SENTINEL_BASE_ID + 3):
    # only START/PAD/IMAGE rows live here; NEWLINE/END belong to another rank.
    start = IMAGE_SENTINEL_BASE_ID - 2
    end = IMAGE_SENTINEL_BASE_ID + 3
    embed = VocabParallelEmbedding.__new__(VocabParallelEmbedding)
    torch.nn.Module.__init__(embed)
    embed.quant_method = UnquantizedEmbeddingMethod()
    embed.weight = torch.nn.Parameter(torch.zeros(end - start, hidden))
    embed.shard_indices = VocabParallelEmbeddingShardIndices(
        padded_org_vocab_start_index=start,
        padded_org_vocab_end_index=end,
        padded_added_vocab_start_index=end,
        padded_added_vocab_end_index=end,
        org_vocab_start_index=start,
        org_vocab_end_index=end,
        added_vocab_start_index=end,
        added_vocab_end_index=end,
    )
    model = DeepseekV4ForConditionalGeneration.__new__(
        DeepseekV4ForConditionalGeneration
    )
    torch.nn.Module.__init__(model)
    model._sentinel_rows_folded = False
    for name in ("image_start", "image_pad", "image_newline", "image_end"):
        setattr(model, name, torch.nn.Parameter(torch.randn(hidden)))
    model.language_model = SimpleNamespace(model=SimpleNamespace(embed_tokens=embed))
    assert model._fold_sentinel_rows()
    assert model._sentinel_rows_folded
    torch.testing.assert_close(embed.weight[2], model.image_start.detach())
    torch.testing.assert_close(embed.weight[3], model.image_pad.detach())
    torch.testing.assert_close(embed.weight[4], model.image_pad.detach())
    assert torch.all(embed.weight[:2] == 0)

    calls = []

    def _embed(ids):
        calls.append(ids)
        return torch.ones(ids.shape[0], hidden)

    model.language_model.embed_input_ids = _embed
    ids = torch.tensor([IMAGE_SENTINEL_BASE_ID, 5])
    out = model.embed_input_ids(ids)
    assert len(calls) == 1
    assert torch.all(out == 1)


def test_fold_sentinel_rows_skips_quantized_embedding():
    from vllm.models.deepseek_v4.nvidia.vl_model import (
        DeepseekV4ForConditionalGeneration,
    )

    model = DeepseekV4ForConditionalGeneration.__new__(
        DeepseekV4ForConditionalGeneration
    )
    torch.nn.Module.__init__(model)
    model._sentinel_rows_folded = False
    model.image_start = torch.nn.Parameter(torch.zeros(4))
    model.language_model = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=torch.nn.Embedding(4, 4))
    )
    assert not model._fold_sentinel_rows()
    assert not model._sentinel_rows_folded
