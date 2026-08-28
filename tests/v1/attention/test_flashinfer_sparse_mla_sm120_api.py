# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavior checks for FlashInfer SM120 sparse MLA backend selection."""

from types import SimpleNamespace

import torch

from vllm.config import set_current_vllm_config
from vllm.models.deepseek_v4.nvidia.flashinfer_sparse import (
    DeepseekV4FlashInferSM120Attention,
    _batch_token_span,
    _required_sm120_sparse_topk,
    sm12x_q_len_spans,
    sm12x_use_per_request_decode,
    spec_decode_uniform_next_n,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils import flashinfer as fi_utils
from vllm.utils.sm12x import sm12x_align_decode_q_len, sm12x_align_prefill_q_len
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseSM120Backend,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def _fake_vllm_config(model_type: str) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type=model_type, index_topk=2048),
        ),
    )


def test_sm120_backend_uses_dedicated_backend_name() -> None:
    assert FlashInferMLASparseSM120Backend.get_name() == "FLASHINFER_MLA_SPARSE_SM120"
    assert (
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120.get_class()
        is FlashInferMLASparseSM120Backend
    )


def test_sm120_backend_uses_sparse_mqa_for_prefill() -> None:
    impl_cls = FlashInferMLASparseSM120Backend.get_impl_cls()

    assert impl_cls.is_sparse
    assert not impl_cls.supports_dense_mha_prefill


def test_v32_glm_sm120_backend_accepts_glm_block_size(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)

    with set_current_vllm_config(_fake_vllm_config("glm4_moe")):
        invalid_reasons = FlashInferMLASparseSM120Backend.validate_configuration(
            head_size=576,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8",
            block_size=256,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=DeviceCapability(12, 0),
            attn_type="decoder",
        )

    assert invalid_reasons == []


def test_sm120_dsv4_capability_checks_exact_dispatch_shape(monkeypatch) -> None:
    fake_module = SimpleNamespace(
        _DECODE_DSV4_DISPATCH=frozenset({(32, 128), (32, 192)})
    )
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)
    monkeypatch.setattr(fi_utils, "_get_submodule", lambda _name: fake_module)
    fi_utils.has_flashinfer_sparse_mla_sm120_config.cache_clear()
    fi_utils.resolve_sm120_dsv4_topk.cache_clear()

    assert fi_utils.has_flashinfer_sparse_mla_sm120_config(32, 128)
    assert fi_utils.has_flashinfer_sparse_mla_sm120_config(32, 192)
    assert not fi_utils.has_flashinfer_sparse_mla_sm120_config(32, 256)
    assert not fi_utils.has_flashinfer_sparse_mla_sm120_config(16, 192)
    assert fi_utils.resolve_sm120_dsv4_topk(192, 32) == 192

    fi_utils.has_flashinfer_sparse_mla_sm120_config.cache_clear()
    fi_utils.resolve_sm120_dsv4_topk.cache_clear()


def test_sm120_dsv4_required_topk_tracks_dspark_width() -> None:
    causal = SimpleNamespace(
        attention_config=SimpleNamespace(use_non_causal=False),
        speculative_config=SimpleNamespace(num_speculative_tokens=5),
    )
    dspark = SimpleNamespace(
        attention_config=SimpleNamespace(use_non_causal=True),
        speculative_config=SimpleNamespace(num_speculative_tokens=5),
    )

    assert _required_sm120_sparse_topk(causal, 128) == 128
    assert _required_sm120_sparse_topk(dspark, 128) == 192


def test_resolve_sm120_dsv4_topk_snaps_missing_dspark_width(monkeypatch) -> None:
    """0.6.17 ships 128/512/1024, not DSpark's aligned width 192."""
    fake_module = SimpleNamespace(
        _DECODE_DSV4_DISPATCH=frozenset({(32, 128), (32, 512), (32, 1024)})
    )
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)
    monkeypatch.setattr(fi_utils, "_get_submodule", lambda _name: fake_module)
    fi_utils.has_flashinfer_sparse_mla_sm120_config.cache_clear()
    fi_utils.resolve_sm120_dsv4_topk.cache_clear()

    assert fi_utils.resolve_sm120_dsv4_topk(128, 32) == 128
    assert fi_utils.resolve_sm120_dsv4_topk(192, 32) == 512
    assert fi_utils.resolve_sm120_dsv4_topk(192, 20) == 512
    assert fi_utils.resolve_sm120_dsv4_topk(2048, 32) is None
    assert not fi_utils.has_flashinfer_sparse_mla_sm120_config(32, 192)
    assert fi_utils.has_flashinfer_sparse_mla_sm120_config(32, 512)

    fi_utils.has_flashinfer_sparse_mla_sm120_config.cache_clear()
    fi_utils.resolve_sm120_dsv4_topk.cache_clear()


def test_spec_decode_uniform_next_n_accepts_dspark_and_skips_ragged_dummy():
    """DSpark k=5 is next_n=6. Capture size 32 over max_num_seqs=6 is ragged.

    The dummy CUDA-graph batch used to assert in _forward_decode; the helper
    must return None so the kernel falls back to per-request decode-form.
    """
    assert spec_decode_uniform_next_n(36, 6) == 6
    assert spec_decode_uniform_next_n(24, 6) == 4
    assert spec_decode_uniform_next_n(8, 4) == 2
    assert spec_decode_uniform_next_n(16, 4) == 4
    assert spec_decode_uniform_next_n(32, 4) == 8
    assert spec_decode_uniform_next_n(6, 6) is None
    assert spec_decode_uniform_next_n(32, 6) is None
    assert spec_decode_uniform_next_n(16, 6) is None
    assert spec_decode_uniform_next_n(8, 6) is None
    assert spec_decode_uniform_next_n(0, 0) is None


def test_sm12x_dummy_next_n_uses_per_request_decode(monkeypatch):
    """Size-8 dummy is 4×2 (next_n=2); batched SM120 decode IMA'd on GB10."""
    from vllm.models.deepseek_v4.nvidia import flashinfer_sparse as fi_sparse

    monkeypatch.setattr(
        fi_sparse.current_platform,
        "is_device_capability_family",
        lambda fam: fam == 120,
    )
    assert sm12x_use_per_request_decode(2, 6)
    assert sm12x_use_per_request_decode(2, 2)
    assert sm12x_use_per_request_decode(4, 6)
    assert sm12x_use_per_request_decode(4, 4)
    assert sm12x_use_per_request_decode(5, 6)
    assert sm12x_use_per_request_decode(5, 5)
    assert sm12x_use_per_request_decode(8, 6)
    assert not sm12x_use_per_request_decode(6, 6)
    assert not sm12x_use_per_request_decode(None, 6)


def test_non_sm12x_keeps_batched_spec_decode(monkeypatch):
    from vllm.models.deepseek_v4.nvidia import flashinfer_sparse as fi_sparse

    monkeypatch.setattr(
        fi_sparse.current_platform,
        "is_device_capability_family",
        lambda fam: False,
    )
    assert not sm12x_use_per_request_decode(2, 6)
    assert not sm12x_use_per_request_decode(4, 6)
    assert not sm12x_use_per_request_decode(6, 6)


def test_sm12x_keeps_q_len_2_as_one_span(monkeypatch):
    """Do not split q_len=2: a 2-token prefill's second KV is not written yet."""
    from vllm.models.deepseek_v4.nvidia import flashinfer_sparse as fi_sparse

    monkeypatch.setattr(
        fi_sparse.current_platform,
        "is_device_capability_family",
        lambda fam: fam == 120,
    )
    assert sm12x_q_len_spans(2) == [(0, 2)]
    assert sm12x_q_len_spans(1) == [(0, 1)]
    assert sm12x_q_len_spans(4) == [(0, 4)]
    assert sm12x_q_len_spans(6) == [(0, 6)]


def test_sm12x_aligns_unsafe_decode_q_len(monkeypatch):
    from vllm.utils import sm12x as sm12x_utils

    monkeypatch.setattr(
        sm12x_utils.current_platform,
        "is_device_capability_family",
        lambda fam: fam == 120,
    )
    assert sm12x_align_decode_q_len(1) == 1
    assert sm12x_align_decode_q_len(2) == 6
    assert sm12x_align_decode_q_len(3) == 6
    assert sm12x_align_decode_q_len(4) == 4
    assert sm12x_align_decode_q_len(5) == 6
    assert sm12x_align_decode_q_len(15) == 16
    assert sm12x_align_decode_q_len(16) == 16
    assert sm12x_align_decode_q_len(36) == 36
    assert sm12x_align_prefill_q_len(2) == 6
    assert sm12x_align_prefill_q_len(4) == 6
    assert sm12x_align_prefill_q_len(5) == 6
    assert sm12x_align_prefill_q_len(8) == 8
    assert sm12x_align_prefill_q_len(16) == 16


def test_non_sm12x_keeps_q_len_2(monkeypatch):
    from vllm.models.deepseek_v4.nvidia import flashinfer_sparse as fi_sparse
    from vllm.utils import sm12x as sm12x_utils

    monkeypatch.setattr(
        fi_sparse.current_platform,
        "is_device_capability_family",
        lambda fam: False,
    )
    monkeypatch.setattr(
        sm12x_utils.current_platform,
        "is_device_capability_family",
        lambda fam: False,
    )
    assert sm12x_q_len_spans(2) == [(0, 2)]
    assert sm12x_align_decode_q_len(2) == 2
    assert sm12x_align_prefill_q_len(2) == 2


def test_batch_token_span_repeats_last_real_row():
    """Padded decode-form rows reuse the last real token's indices/lens."""
    indices = torch.tensor([[10, 11], [12, 13]], dtype=torch.int32)
    lens = torch.tensor([2, 3], dtype=torch.int32)
    padded = _batch_token_span(indices, 0, 2, 4)
    padded_lens = _batch_token_span(lens, 0, 2, 4)
    assert padded.shape == (1, 4, 2)
    assert torch.equal(padded[0, :2], indices)
    assert torch.equal(padded[0, 2:], indices[-1:].expand(2, -1))
    assert padded_lens.shape == (1, 4)
    assert torch.equal(padded_lens[0, :2], lens)
    assert torch.equal(padded_lens[0, 2:], torch.tensor([3, 3], dtype=torch.int32))
    sentinels = _batch_token_span(indices, 0, 2, 4, fill=-1)
    assert torch.equal(sentinels[0, 2:], torch.full((2, 2), -1, dtype=torch.int32))


def _launch_per_request(monkeypatch, q_len: int, **kwargs) -> list[dict]:
    from vllm.models.deepseek_v4.nvidia import flashinfer_sparse as fi_sparse

    launches: list[dict] = []

    def _fake_launch(**launch_kwargs):
        launches.append(launch_kwargs)

    monkeypatch.setattr(
        fi_sparse, "flashinfer_trtllm_batch_decode_sparse_mla_dsv4", _fake_launch
    )
    dummy = SimpleNamespace(
        scale=1.0,
        attn_sink=None,
        _get_workspace=lambda device: torch.zeros(8, dtype=torch.uint8),
    )
    q = torch.arange(q_len * 8 * 512, dtype=torch.float32).view(q_len, 8, 512)
    output = torch.zeros(q_len, 8, 512)
    swa_indices = torch.arange(q_len * 16, dtype=torch.int32).view(q_len, 16)
    swa_lens = torch.arange(1, q_len + 1, dtype=torch.int32)
    DeepseekV4FlashInferSM120Attention._launch_per_request_decode(
        dummy,
        q,
        output,
        torch.zeros(1),
        None,
        swa_indices,
        swa_lens,
        None,
        None,
        0,
        q_len,
        **kwargs,
    )
    return launches


def _launch_per_request_shapes(monkeypatch, q_len: int, **kwargs) -> list[torch.Size]:
    return [
        launch["query"].shape
        for launch in _launch_per_request(monkeypatch, q_len, **kwargs)
    ]


def test_launch_per_request_decode_pads_q_len_2_once(monkeypatch):
    """A 2-token SM12x launch must be one [1, 6], not [1, 4] or two [1, 1]."""
    from vllm.utils import sm12x as sm12x_utils

    monkeypatch.setattr(
        sm12x_utils.current_platform,
        "is_device_capability_family",
        lambda fam: fam == 120,
    )
    assert _launch_per_request_shapes(monkeypatch, 2) == [torch.Size([1, 6, 8, 512])]
    assert _launch_per_request_shapes(monkeypatch, 4) == [torch.Size([1, 4, 8, 512])]
    assert _launch_per_request_shapes(monkeypatch, 5) == [torch.Size([1, 6, 8, 512])]


def test_launch_per_request_decode_never_snaps_2_to_4(monkeypatch):
    """The 15:52 Spark IMA was a leftover 2→4 decode align."""
    from vllm.utils import sm12x as sm12x_utils

    monkeypatch.setattr(
        sm12x_utils.current_platform,
        "is_device_capability_family",
        lambda fam: fam == 120,
    )
    launches = _launch_per_request(monkeypatch, 2)
    assert [launch["query"].shape for launch in launches] == [
        torch.Size([1, 6, 8, 512])
    ]
    assert launches[0]["query"].shape[1] != 4


def test_launch_per_request_prefill_pads_q_len_2_once(monkeypatch):
    """A 2-token SM12x prefill must be one [1, 6] launch, not two [1, 1]."""
    from vllm.utils import sm12x as sm12x_utils

    monkeypatch.setattr(
        sm12x_utils.current_platform,
        "is_device_capability_family",
        lambda fam: fam == 120,
    )
    launches = _launch_per_request(monkeypatch, 2, is_prefill=True)
    assert [launch["query"].shape for launch in launches] == [torch.Size([1, 6, 8, 512])]
    query = launches[0]["query"]
    indices = launches[0]["sparse_indices"]
    lens = launches[0]["swa_topk_lens"]
    assert torch.equal(query[0, 2:], query[0, 1:2].expand(4, -1, -1))
    assert torch.equal(indices[0, 2:], indices[0, 1:2].expand(4, -1))
    assert torch.equal(lens[0, 2:], lens[0, 1:2].expand(4))
    assert not torch.any(indices == -1)
    assert _launch_per_request_shapes(monkeypatch, 4, is_prefill=True) == [
        torch.Size([1, 6, 8, 512])
    ]


def test_forward_decode_q_len_2_is_one_padded_launch(monkeypatch):
    """A 2-token seed labeled decode must still be one [1, 6], not [1, 2]."""
    from vllm.models.deepseek_v4.nvidia import flashinfer_sparse as fi_sparse
    from vllm.utils import sm12x as sm12x_utils

    monkeypatch.setattr(
        fi_sparse.current_platform,
        "is_device_capability_family",
        lambda fam: fam == 120,
    )
    monkeypatch.setattr(
        sm12x_utils.current_platform,
        "is_device_capability_family",
        lambda fam: fam == 120,
    )
    shapes: list[torch.Size] = []

    def _fake_launch(**launch_kwargs):
        shapes.append(launch_kwargs["query"].shape)
        assert launch_kwargs["query"].is_contiguous()

    monkeypatch.setattr(
        fi_sparse, "flashinfer_trtllm_batch_decode_sparse_mla_dsv4", _fake_launch
    )
    dummy = SimpleNamespace(
        scale=1.0,
        attn_sink=None,
        kv_cache_torch_dtype=torch.bfloat16,
        # next_n=2 must pad even if this equals decode_query_len.
        _decode_query_len=2,
        _get_workspace=lambda device: torch.zeros(8, dtype=torch.uint8),
        _as_sparse_cache=DeepseekV4FlashInferSM120Attention._as_sparse_cache,
        swa_cache_layer=SimpleNamespace(kv_cache=torch.zeros(2, 1, 1, 512)),
    )
    dummy._prepare_query = (
        DeepseekV4FlashInferSM120Attention._prepare_query.__get__(dummy)
    )
    dummy._launch_per_request_decode = (
        DeepseekV4FlashInferSM120Attention._launch_per_request_decode.__get__(dummy)
    )
    swa = SimpleNamespace(
        num_decodes=1,
        num_decode_tokens=2,
        decode_swa_indices=torch.zeros(2, 16, dtype=torch.int32),
        decode_swa_lens=torch.ones(2, dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2]),
    )
    q = torch.zeros(2, 8, 512, dtype=torch.bfloat16)
    output = torch.zeros(2, 8, 512, dtype=torch.bfloat16)
    DeepseekV4FlashInferSM120Attention._forward_decode(
        dummy, q, None, swa, None, True, output
    )
    assert shapes == [torch.Size([1, 6, 8, 512])]


def test_forward_prefill_q_len_2_is_one_padded_launch(monkeypatch):
    """Spark 17:03: mixed-warmup seed is one [1, 6], never [1, 4] or 65."""
    from vllm.models.deepseek_v4.nvidia import flashinfer_sparse as fi_sparse
    from vllm.utils import sm12x as sm12x_utils

    monkeypatch.setattr(
        fi_sparse.current_platform,
        "is_device_capability_family",
        lambda fam: fam == 120,
    )
    monkeypatch.setattr(
        sm12x_utils.current_platform,
        "is_device_capability_family",
        lambda fam: fam == 120,
    )
    shapes: list[torch.Size] = []

    def _fake_launch(**launch_kwargs):
        shapes.append(launch_kwargs["query"].shape)

    monkeypatch.setattr(
        fi_sparse, "flashinfer_trtllm_batch_decode_sparse_mla_dsv4", _fake_launch
    )
    dummy = SimpleNamespace(
        compress_ratio=1,
        PREFILL_CHUNK_SIZE=4,
        scale=1.0,
        attn_sink=None,
        kv_cache_torch_dtype=torch.bfloat16,
        _get_workspace=lambda device: torch.zeros(8, dtype=torch.uint8),
        _as_sparse_cache=DeepseekV4FlashInferSM120Attention._as_sparse_cache,
    )
    dummy._prepare_query = (
        DeepseekV4FlashInferSM120Attention._prepare_query.__get__(dummy)
    )
    dummy._launch_per_request_decode = (
        DeepseekV4FlashInferSM120Attention._launch_per_request_decode.__get__(
            dummy
        )
    )
    swa = SimpleNamespace(
        num_prefills=1,
        num_decodes=0,
        num_decode_tokens=0,
        num_prefill_tokens=2,
        query_start_loc_cpu=torch.tensor([0, 2]),
        prefill_swa_indices=torch.zeros(2, 16, dtype=torch.int32),
        prefill_swa_lens=torch.ones(2, dtype=torch.int32),
    )
    q = torch.zeros(2, 8, 512, dtype=torch.bfloat16)
    output = torch.zeros(2, 8, 512, dtype=torch.bfloat16)
    DeepseekV4FlashInferSM120Attention._forward_prefill(
        dummy, q, None, torch.zeros(1), output, None, swa
    )
    assert shapes == [torch.Size([1, 6, 8, 512])]
    assert shapes[0][1] != 4
    assert shapes[0] != torch.Size([65, 8, 512])
