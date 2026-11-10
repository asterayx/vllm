# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the DeepSeek-V4.1 RTX PRO 5000 serve-path fixes."""

import errno
import os
from types import SimpleNamespace

import pytest
import torch
import triton.language as tl

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


_splitk_reduce_store = None  # set to an interpreted copy of the real helper


def _splitk_test_kernel(
    x_ptr,
    w_ptr,
    z_ptr,
    ws_ptr,
    cnt_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(2)
    k_per: tl.constexpr = K // SPLIT_K
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * k_per + tl.arange(0, k_per)
    m_mask = offs_m[:, None] < M
    x = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :], mask=m_mask, other=0.0)
    w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :])
    acc = tl.dot(x, tl.trans(w))
    _splitk_reduce_store(
        acc, ws_ptr, cnt_ptr, z_ptr, offs_m, offs_n, m_mask, M, N, N, SPLIT_K
    )


def test_fp8_skinny_splitk_reduces_in_kernel_and_rearms(monkeypatch):
    """Split-K partials sum in split order to bf16 and leave counters at zero.

    Runs under the Triton CPU interpreter, whose bf16 tl.dot is unreliable, so
    a fp32 test kernel feeds the reduction helper the real kernel uses.
    """
    monkeypatch.setenv("TRITON_INTERPRET", "1")
    import triton

    helper = triton.jit(fp8_skinny_linear._splitk_reduce_store.fn)
    monkeypatch.setitem(globals(), "_splitk_reduce_store", helper)
    kernel = triton.jit(_splitk_test_kernel)
    M, N, K, bn, sk = 3, 64, 128, 32, 4
    x = torch.randn(M, K)
    w = torch.randn(N, K)
    kp = K // sk
    ref = sum(
        x[:, i * kp : (i + 1) * kp] @ w[:, i * kp : (i + 1) * kp].T for i in range(sk)
    )
    cnt = fp8_skinny_linear.new_splitk_counters(N, torch.device("cpu"))
    for _ in range(2):
        z = torch.empty(M, N, dtype=torch.bfloat16)
        ws = torch.empty(sk, M, N)
        kernel[(N // bn, 1, sk)](
            x, w, z, ws, cnt, M, N=N, K=K, BLOCK_M=16, BLOCK_N=bn, SPLIT_K=sk
        )
        torch.testing.assert_close(z, ref.to(torch.bfloat16))
        assert int(cnt.abs().sum()) == 0


@pytest.mark.parametrize(
    ("threshold", "file_backed", "shared", "cpu_offload", "tokens", "expected"),
    [
        (0, False, False, True, 64, False),
        (16, False, False, True, 4, False),
        (16, False, False, True, 16, True),
        (16, False, True, True, 64, False),
        (16, False, False, False, 64, False),
        (0, True, False, True, 1, True),
    ],
)
def test_engram_host_gather_only_for_large_pinned_lookups(
    monkeypatch, threshold, file_backed, shared, cpu_offload, tokens, expected
):
    """Pinned tables switch to the host gather only at or above the threshold."""
    monkeypatch.setattr(nv_engram, "_DPSK_ENGRAM_HOST_GATHER_MIN_TOKENS", threshold)
    stub = SimpleNamespace(
        _file_mapping=object() if file_backed else None,
        _shared_memory=object() if shared else None,
        cpu_offload=cpu_offload,
    )
    use = nv_engram.ParallelEngramEmbedding._use_host_gather(stub, tokens)
    assert use is expected
