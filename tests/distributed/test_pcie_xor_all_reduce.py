# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock, patch

import pytest
import torch

import vllm.envs as envs
from vllm.distributed.device_communicators import cuda_communicator
from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
from vllm.distributed.device_communicators.pcie_xor_all_reduce import (
    precheck,
    try_create_pcie_xor_allreduce,
)
from vllm.distributed.device_communicators.pcie_xor_butterfly import (
    ALIGN,
    DEFAULT_MAX_BYTES,
    DEFAULT_TIMEOUT_MS,
    FLAG_BYTES,
    SUPPORTED_DTYPES,
    accepts_tensor,
    butterfly_partials,
    format_attach_line,
    format_fallback_line,
    reduce_xor_butterfly,
    slab_layout,
    xor_partner,
)
from vllm.envs import disable_envs_cache


def test_xor_partners_are_the_butterfly():
    assert [(xor_partner(rank, 0), rank) for rank in range(4)] == [
        (1, 0),
        (0, 1),
        (3, 2),
        (2, 3),
    ]
    assert xor_partner(0, 1) == 2
    assert xor_partner(1, 1) == 3
    assert xor_partner(0, 2) == 4
    assert xor_partner(7, 2) == 3


def test_balanced_tree_differs_from_a_left_fold():
    """Two-operand adds commute; the tree shape does not."""
    shards = [
        torch.tensor([2.0**100]),
        torch.tensor([1.0]),
        torch.tensor([-(2.0**100)]),
        torch.tensor([1.0]),
    ]
    balanced = (shards[0] + shards[1]) + (shards[2] + shards[3])
    left = ((shards[0] + shards[1]) + shards[2]) + shards[3]
    assert balanced.item() == 0.0
    assert left.item() == 1.0

    got = reduce_xor_butterfly(shards)
    partials = butterfly_partials(shards)
    assert all(torch.equal(partial, partials[0]) for partial in partials)
    assert got.item() == 0.0
    assert got.dtype == torch.float32


@pytest.mark.parametrize("world_size", [2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_every_rank_matches_the_schedule(world_size, dtype):
    gen = torch.Generator().manual_seed(world_size + dtype.itemsize)
    shards = [torch.randn(2, 8, generator=gen).to(dtype) for _ in range(world_size)]
    partials = butterfly_partials(shards)
    assert all(torch.equal(partial, partials[0]) for partial in partials)
    got = reduce_xor_butterfly(shards)
    assert got.dtype == dtype
    assert torch.equal(got, partials[0].to(dtype))
    reference = torch.stack([shard.float() for shard in shards]).sum(0).to(dtype)
    torch.testing.assert_close(got, reference, rtol=1e-2, atol=1e-2)


def test_decode_shaped_bf16_stays_within_bf16_tolerance():
    gen = torch.Generator().manual_seed(5120)
    shards = [torch.randn(4, 5120, generator=gen).to(torch.bfloat16) for _ in range(4)]
    got = reduce_xor_butterfly(shards)
    reference = torch.stack([shard.float() for shard in shards]).sum(0)
    # bf16 rounding of a 4-add tree, not an exact fp32 sum.
    torch.testing.assert_close(got.float(), reference, rtol=1e-2, atol=1e-2)
    assert DEFAULT_MAX_BYTES >= 4 * 5120 * 2
    assert DEFAULT_MAX_BYTES >= 64 * 5120 * 2
    assert DEFAULT_MAX_BYTES < 4096 * 5120 * 2


def test_slab_layout_separates_flags_and_parity_banks():
    layout = slab_layout(128, 4)
    assert layout.rounds == 2
    assert layout.flag_base == 0
    assert layout.payload_base % ALIGN == 0
    assert layout.slot_stride % ALIGN == 0
    assert layout.slot_stride >= 128 * 4
    flag_end = layout.flag_offset(1, layout.rounds - 1) + FLAG_BYTES
    assert flag_end <= layout.payload_base
    last = layout.payload_offset(1, layout.rounds - 1) + layout.slot_stride
    assert last == layout.nbytes
    assert layout.payload_offset(0, 0) != layout.payload_offset(1, 0)
    assert layout.payload_offset(0, 0) != layout.payload_offset(0, 1)


def test_tensor_gate_matches_dtype_and_size():
    common = {
        "contiguous": True,
        "cuda": True,
        "max_bytes": 1024,
        "max_numel": 512,
    }
    assert accepts_tensor(dtype=torch.bfloat16, nbytes=1024, numel=512, **common)
    assert accepts_tensor(dtype=torch.float32, nbytes=64, numel=16, **common)
    assert not accepts_tensor(dtype=torch.int32, nbytes=64, numel=16, **common)
    assert not accepts_tensor(dtype=torch.bfloat16, nbytes=1025, numel=512, **common)
    assert not accepts_tensor(
        dtype=torch.bfloat16,
        nbytes=16,
        numel=8,
        contiguous=False,
        cuda=True,
        max_bytes=1024,
        max_numel=512,
    )
    assert not accepts_tensor(dtype=torch.bfloat16, nbytes=16, numel=0, **common)


def test_attach_and_fallback_lines():
    attach = format_attach_line(
        world_size=4, max_bytes=DEFAULT_MAX_BYTES, max_numel=512, rounds=2
    )
    assert "world_size=4" in attach
    assert "paired-posted-write-xor-butterfly" in attach
    assert "bf16" in attach and "fp16" in attach and "fp32" in attach
    assert "NCCL" in attach
    fallback = format_fallback_line("world_size=6 is not supported")
    assert fallback.startswith("DSV41 PCIe XOR all-reduce disabled:")
    assert "Falling back to NCCL" in fallback


def test_env_defaults_match_the_schedule(monkeypatch):
    disable_envs_cache()
    monkeypatch.delenv("DSV41_EXP_PCIE_XOR_AR", raising=False)
    monkeypatch.delenv("DSV41_EXP_PCIE_XOR_AR_MAX_BYTES", raising=False)
    monkeypatch.delenv("DSV41_EXP_PCIE_XOR_AR_TIMEOUT_MS", raising=False)
    assert envs.DSV41_EXP_PCIE_XOR_AR is False
    assert envs.DSV41_EXP_PCIE_XOR_AR_MAX_BYTES == DEFAULT_MAX_BYTES
    assert envs.DSV41_EXP_PCIE_XOR_AR_TIMEOUT_MS == DEFAULT_TIMEOUT_MS
    assert set(SUPPORTED_DTYPES) == {
        torch.float16,
        torch.bfloat16,
        torch.float32,
    }


def test_unset_flag_does_not_attach(monkeypatch):
    monkeypatch.delenv("DSV41_EXP_PCIE_XOR_AR", raising=False)
    assert precheck("tp:0", torch.device("cpu")) == ("skip", None)
    with patch(
        "vllm.distributed.device_communicators.pcie_xor_all_reduce.logger"
    ) as log:
        assert try_create_pcie_xor_allreduce(None, torch.device("cpu"), "tp:0") is None
    log.warning_once.assert_not_called()


def test_non_tp_group_is_ignored(monkeypatch):
    monkeypatch.setenv("DSV41_EXP_PCIE_XOR_AR", "1")
    assert precheck("pp:0", torch.device("cuda")) == ("skip", None)


def test_invalid_flag_and_limits_fall_back(monkeypatch):
    monkeypatch.setenv("DSV41_EXP_PCIE_XOR_AR", "yes")
    action, reason = precheck("tp:0", torch.device("cuda"))
    assert action == "fallback"
    assert reason is not None and "invalid" in reason

    monkeypatch.setenv("DSV41_EXP_PCIE_XOR_AR", "1")
    monkeypatch.setattr(
        "vllm.distributed.device_communicators.pcie_xor_all_reduce.current_platform.is_cuda",
        lambda: True,
    )
    monkeypatch.setattr(
        "vllm.distributed.device_communicators.pcie_xor_all_reduce.kernel_available",
        lambda: True,
    )
    monkeypatch.setenv("DSV41_EXP_PCIE_XOR_AR_MAX_BYTES", "1")
    action, reason = precheck("tp:0", torch.device("cuda", 0))
    assert action == "fallback"
    assert reason is not None and "MAX_BYTES" in reason

    monkeypatch.setenv("DSV41_EXP_PCIE_XOR_AR_MAX_BYTES", str(DEFAULT_MAX_BYTES))
    monkeypatch.setenv("DSV41_EXP_PCIE_XOR_AR_TIMEOUT_MS", "0")
    action, reason = precheck("tp:0", torch.device("cuda", 0))
    assert action == "fallback"
    assert reason is not None and "TIMEOUT_MS" in reason

    monkeypatch.delenv("DSV41_EXP_PCIE_XOR_AR_TIMEOUT_MS", raising=False)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    action, reason = precheck("tp:0", torch.device("cuda", 0))
    assert action == "fallback"
    assert reason is not None and "BATCH_INVARIANT" in reason
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)

    monkeypatch.setattr(
        "vllm.distributed.device_communicators.pcie_xor_all_reduce.kernel_available",
        lambda: False,
    )
    action, reason = precheck("tp:0", torch.device("cuda", 0))
    assert action == "fallback"
    assert reason is not None and "pcie_xor_all_reduce" in reason


def test_requested_path_logs_cuda_fallback_without_crashing(monkeypatch):
    monkeypatch.setenv("DSV41_EXP_PCIE_XOR_AR", "1")
    with patch(
        "vllm.distributed.device_communicators.pcie_xor_all_reduce.logger"
    ) as log:
        assert try_create_pcie_xor_allreduce(None, torch.device("cpu"), "tp:0") is None
    message = log.warning_once.call_args.args[1]
    assert "CUDA" in message
    assert "Falling back to NCCL" in message
    assert "world_size" not in message or "algorithm" not in message


def test_precheck_builds_when_the_op_and_cuda_are_present(monkeypatch):
    monkeypatch.setenv("DSV41_EXP_PCIE_XOR_AR", "1")
    monkeypatch.delenv("DSV41_EXP_PCIE_XOR_AR_MAX_BYTES", raising=False)
    monkeypatch.delenv("DSV41_EXP_PCIE_XOR_AR_TIMEOUT_MS", raising=False)
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    monkeypatch.setattr(
        "vllm.distributed.device_communicators.pcie_xor_all_reduce.current_platform.is_cuda",
        lambda: True,
    )
    monkeypatch.setattr(
        "vllm.distributed.device_communicators.pcie_xor_all_reduce.kernel_available",
        lambda: True,
    )
    assert precheck("tp:0", torch.device("cuda", 0)) == ("build", None)


def _bare_communicator() -> CudaCommunicator:
    communicator = CudaCommunicator.__new__(CudaCommunicator)
    communicator.unique_name = "tp:0"
    communicator.world_size = 4
    communicator.xor_ar_comm = None
    communicator.fi_pcie_ipc_ar_comm = None
    communicator.fi_ar_comm = None
    communicator.pynccl_comm = None
    communicator.qr_comm = None
    communicator.use_aiter_allreduce = False
    communicator.aiter_ar_comm = None
    communicator.ca_comm = None
    communicator.symm_mem_comm = None
    return communicator


def test_dispatch_prefers_xor_and_can_fall_through(monkeypatch):
    monkeypatch.setattr(
        cuda_communicator, "should_nccl_symm_mem_allreduce", lambda *_: False
    )
    communicator = _bare_communicator()
    xor = Mock()
    xor.should_use.return_value = True
    expected = object()
    xor.all_reduce.return_value = expected
    communicator.xor_ar_comm = xor
    communicator.fi_ar_comm = Mock()
    communicator.pynccl_comm = Mock(world_size=4, disabled=False)

    assert communicator.all_reduce(object()) is expected
    communicator.fi_ar_comm.should_use_fi_ar.assert_not_called()

    xor.should_use.return_value = False
    fi_out = torch.empty(1)
    communicator.fi_ar_comm.disabled = False
    communicator.fi_ar_comm.should_use_fi_ar.return_value = True
    communicator.fi_ar_comm.all_reduce.return_value = fi_out
    assert communicator.all_reduce(torch.empty(1)) is fi_out


def test_backend_log_names_xor_when_attached():
    communicator = _bare_communicator()
    communicator.xor_ar_comm = Mock(disabled=False)
    with patch.object(cuda_communicator.logger, "info_once") as info:
        communicator._log_all_reduce_backend_selection()
    rendered = " ".join(str(arg) for arg in info.call_args.args)
    assert "PCIE_XOR" in rendered
    assert "PYNCCL" not in rendered or rendered.index("PCIE_XOR") < rendered.index(
        "PYNCCL"
    )
