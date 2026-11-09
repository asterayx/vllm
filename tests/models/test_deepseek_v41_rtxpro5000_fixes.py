# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the DeepSeek-V4.1 RTX PRO 5000 serve-path fixes."""

import errno
import os
from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v41.nvidia import engram as nv_engram
from vllm.models.deepseek_v41.nvidia.ops import fp8_skinny_linear
from vllm.v1.worker import gpu_worker
from vllm.v1.worker.gpu.spec_decode.dspark import utils as dspark_utils


def _engram_stub(**overrides):
    fields = dict(
        dim=32,
        block_size=32,
        part_num_embeddings=8,
        row_bytes=32,
        packed_fp4=False,
        num_embeddings=16,
        vocab_start_idx=0,
        vocab_end_idx=8,
        _file_mapping=None,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.fixture
def tp_rank0(monkeypatch):
    monkeypatch.setattr(nv_engram, "get_tensor_model_parallel_rank", lambda: 0)


@pytest.mark.usefixtures("tp_rank0")
def test_engram_file_backed_tables_are_private_and_unlinked(tmp_path):
    """Two processes sharing storage_path must never map the same file."""
    allocate = nv_engram.ParallelEngramEmbedding._allocate_file_backed
    first, second = _engram_stub(), _engram_stub()

    w1, s1 = allocate(first, str(tmp_path))
    w2, s2 = allocate(second, str(tmp_path))

    assert first._file_path != second._file_path
    assert list(tmp_path.iterdir()) == []
    w1.view(torch.uint8).fill_(7)
    assert int(w2.view(torch.uint8).max()) == 0
    assert w1.shape == (8, 32) and s1.shape == (8, 1)


@pytest.mark.usefixtures("tp_rank0")
def test_engram_file_backed_full_disk_fails_cleanly(tmp_path, monkeypatch):
    """A full disk must raise at allocation, not SIGBUS during weight load."""

    def no_space(fd, offset, length):
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))

    monkeypatch.setattr(nv_engram.os, "posix_fallocate", no_space)
    allocate = nv_engram.ParallelEngramEmbedding._allocate_file_backed

    with pytest.raises(RuntimeError, match="cannot hold"):
        allocate(_engram_stub(), str(tmp_path))
    assert list(tmp_path.iterdir()) == []


def _sched_output(tokens_per_req: int, num_reqs: int, new: int, spec: int):
    return SimpleNamespace(
        num_scheduled_tokens={f"r{i}": tokens_per_req for i in range(num_reqs)},
        total_num_scheduled_tokens=tokens_per_req * num_reqs,
        scheduled_new_reqs=[object()] * new,
        scheduled_spec_decode_tokens={f"r{i}": [0] * 3 for i in range(spec)},
    )


def test_pp_meta_key_separates_prefill_tail_from_spec_decode(monkeypatch):
    """Same token counts with different padding must not share cached metadata."""
    monkeypatch.setattr(gpu_worker, "_DPSK_PP_META_CACHE", True)
    decode = _sched_output(4, 2, new=0, spec=2)
    prefill_tail = _sched_output(4, 2, new=0, spec=1)

    assert gpu_worker._dpsk_meta_key(decode) != gpu_worker._dpsk_meta_key(prefill_tail)
    assert gpu_worker._dpsk_meta_key(_sched_output(4, 2, 0, 2)) == (
        gpu_worker._dpsk_meta_key(decode)
    )


def _vllm_config(model_type: str):
    hf = SimpleNamespace(model_type=model_type)
    return SimpleNamespace(model_config=SimpleNamespace(hf_text_config=hf))


def _quant(name: str):
    return SimpleNamespace(get_name=lambda: name)


@pytest.mark.parametrize(
    ("model_type", "quant", "expected"),
    [
        ("deepseek_v41", "modelopt_mixed", True),
        ("deepseek_v4", "deepseek_v4_fp8", True),
        ("qwen3", "modelopt_mixed", False),
        ("deepseek_v41", "fp8", False),
    ],
)
def test_draft_moe_mxfp4_override_is_scoped(model_type, quant, expected):
    """Only DeepSeek-V4.x NVFP4 drafters treat mtp experts as native MXFP4."""
    assert (
        dspark_utils._draft_moe_is_native_mxfp4(_vllm_config(model_type), _quant(quant))
        is expected
    )


def test_fp8_skinny_max_m_takes_tp_minimum(monkeypatch):
    """Every TP shard of a layer must switch to fp8 at the same batch size."""
    peer = {(0, 512, 1024): 16, (1, 256, 1024): 64}
    monkeypatch.setattr(
        fp8_skinny_linear,
        "_SHAPE_MAXM",
        {(0, 512, 1024): 64, (1, 256, 1024): 32},
    )
    import vllm.distributed as dist_mod

    monkeypatch.setattr(
        dist_mod,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=2, cpu_group=None),
    )

    def fake_all_reduce(tensor, op, group):
        if op == torch.distributed.ReduceOp.MAX:
            other = torch.tensor([len(peer), -len(peer)])
            tensor.copy_(torch.maximum(tensor, other))
        else:
            assert op == torch.distributed.ReduceOp.MIN
            other = torch.tensor([peer[k] for k in sorted(peer)])
            tensor.copy_(torch.minimum(tensor, other))

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    fp8_skinny_linear._agree_on_max_m_across_tp()

    assert fp8_skinny_linear._SHAPE_MAXM == {(0, 512, 1024): 16, (1, 256, 1024): 32}


@pytest.mark.parametrize(
    ("env", "default_arch", "expected"),
    [
        (None, True, True),
        (None, False, False),
        ("0", True, False),
        ("1", False, True),
    ],
)
def test_engram_storage_path_sees_auto_enabled_breakable_graphs(
    monkeypatch, env, default_arch, expected
):
    """DSv4.1 auto-enables breakable graphs after the Engram check runs."""
    from vllm.config.vllm import VllmConfig

    if env is None:
        monkeypatch.delenv("VLLM_USE_BREAKABLE_CUDAGRAPH", raising=False)
    else:
        monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", env)
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    stub = SimpleNamespace(
        _uses_breakable_cudagraph_by_default=lambda: default_arch,
        _uses_breakable_cudagraph_for_batch_invariance=lambda: False,
    )
    assert VllmConfig._breakable_cudagraph_will_be_enabled(stub) is expected
