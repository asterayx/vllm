# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in PCIe paired posted-write XOR-butterfly tensor-parallel all-reduce.

Unset ``DSV41_EXP_PCIE_XOR_AR`` leaves the process on today's NCCL path.
The flag does not turn the older FlashInfer ``PCIE_IPC`` backend back on
and does not enable the NVLink custom all-reduce. A failed attach logs one
reason and the call site keeps NCCL. Per-call size and dtype misses are
silent: prefill-sized messages are supposed to stay on NCCL.

The launch path does not synchronize and does not read a host epoch, so a
captured CUDA graph replays the same kernel with a device-side counter.
"""

import os
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.distributed.device_communicators.pcie_xor_butterfly import (
    DEFAULT_MAX_BYTES,
    DEFAULT_TIMEOUT_MS,
    MAX_CONFIG_BYTES,
    SUPPORTED_WORLD_SIZES,
    SlabLayout,
    accepts_tensor,
    format_attach_line,
    format_fallback_line,
    max_numel_for_bytes,
    parse_positive_int,
    slab_layout,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


def kernel_available() -> bool:
    """Return whether this build registered the CUDA op."""
    try:
        op = getattr(torch.ops, "_C_custom_ar", None)
        return op is not None and getattr(op, "pcie_xor_all_reduce", None) is not None
    except Exception:
        return False


def _flag_raw() -> str:
    return os.getenv("DSV41_EXP_PCIE_XOR_AR", "0").strip()


def precheck(
    unique_name: str,
    device: torch.device | int | str | None,
) -> tuple[str, str | None]:
    """Decide whether to skip, fall back, or build the communicator.

    Returns:
        ``("skip", None)`` when the flag is off or this group is not TP.
        ``("fallback", reason)`` when the flag was requested but cannot attach.
        ``("build", None)`` when the collective init may run.

    """
    raw = _flag_raw()
    if raw in ("", "0"):
        return "skip", None
    if raw != "1":
        return "fallback", (f"DSV41_EXP_PCIE_XOR_AR={raw!r} is invalid; expected 1")
    if unique_name.split(":")[0] != "tp":
        return "skip", None
    if not current_platform.is_cuda() or not _is_cuda_device(device):
        return "fallback", "PCIe XOR all-reduce requires a CUDA tensor-parallel group"
    if os.getenv("VLLM_BATCH_INVARIANT", "0").strip() == "1":
        return (
            "fallback",
            "VLLM_BATCH_INVARIANT is set and this reduction order is not NCCL's",
        )
    max_bytes = parse_positive_int("DSV41_EXP_PCIE_XOR_AR_MAX_BYTES", DEFAULT_MAX_BYTES)
    if (
        max_bytes <= 0
        or max_bytes > MAX_CONFIG_BYTES
        or max_numel_for_bytes(max_bytes) < 1
    ):
        return "fallback", (
            f"DSV41_EXP_PCIE_XOR_AR_MAX_BYTES must be in 2..{MAX_CONFIG_BYTES}"
        )
    timeout_ms = parse_positive_int(
        "DSV41_EXP_PCIE_XOR_AR_TIMEOUT_MS", DEFAULT_TIMEOUT_MS
    )
    if timeout_ms <= 0 or timeout_ms > 120_000:
        return "fallback", ("DSV41_EXP_PCIE_XOR_AR_TIMEOUT_MS must be in 1..120000")
    if not kernel_available():
        return "fallback", (
            "this build has no pcie_xor_all_reduce op; rebuild vLLM with CUDA"
        )
    return "build", None


def _is_cuda_device(device: torch.device | int | str | None) -> bool:
    if isinstance(device, int):
        return True
    if isinstance(device, str):
        return device.startswith("cuda")
    if isinstance(device, torch.device):
        return device.type == "cuda"
    return False


def _coerce_device(device: torch.device | int | str) -> torch.device:
    if isinstance(device, int):
        return torch.device("cuda", device)
    if isinstance(device, str):
        device = torch.device(device)
    if device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _should_log(group: ProcessGroup | None) -> bool:
    if group is None or not dist.is_available() or not dist.is_initialized():
        return True
    try:
        return dist.get_rank(group) == 0
    except Exception:
        return True


def _log_fallback(group: ProcessGroup | None, reason: str) -> None:
    if _should_log(group):
        logger.warning_once("%s", format_fallback_line(reason), scope="local")


class PcieXorAllReduce:
    """TP all-reduce over IPC slabs. ``disabled`` callers must use NCCL."""

    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        ranks: list[int],
    ) -> None:
        self.group = group
        self.device = device
        self.ranks = list(ranks)
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        self.disabled = True
        self.max_bytes = parse_positive_int(
            "DSV41_EXP_PCIE_XOR_AR_MAX_BYTES", DEFAULT_MAX_BYTES
        )
        self._timeout_ns = (
            parse_positive_int("DSV41_EXP_PCIE_XOR_AR_TIMEOUT_MS", DEFAULT_TIMEOUT_MS)
            * 1_000_000
        )
        self.max_numel = max_numel_for_bytes(self.max_bytes)
        self._layout: SlabLayout | None = None
        self._state: torch.Tensor | None = None
        self._partial: torch.Tensor | None = None
        self._local_ptr = 0
        self._peer_ptrs: list[int] = []
        self._closed = False
        self._released = False

        reason = self._world_reason()
        if reason:
            _log_fallback(group, reason)
            return
        reason = self._same_node_reason()
        if reason:
            _log_fallback(group, reason)
            return
        reason = self._p2p_reason()
        if reason:
            _log_fallback(group, reason)
            return
        reason = self._allocate()
        if reason:
            _log_fallback(group, reason)
            self._release_storage()
            return
        self.disabled = False
        if self.rank == 0 and self._layout is not None:
            logger.info_once(
                "%s",
                format_attach_line(
                    world_size=self.world_size,
                    max_bytes=self.max_bytes,
                    max_numel=self.max_numel,
                    rounds=self._layout.rounds,
                ),
            )

    def _world_reason(self) -> str | None:
        if self.world_size in SUPPORTED_WORLD_SIZES:
            return None
        supported = ", ".join(str(size) for size in SUPPORTED_WORLD_SIZES)
        return f"world_size={self.world_size} is not supported ({supported})"

    def _same_node_reason(self) -> str | None:
        from vllm.distributed.parallel_state import in_the_same_node_as

        if all(in_the_same_node_as(self.group, source_rank=0)):
            return None
        return "TP ranks are not all on one node"

    def _p2p_reason(self) -> str | None:
        from vllm.distributed.device_communicators.all_reduce_utils import (
            gpu_p2p_access_check,
        )

        # Single-node global ranks match the P2P cache's local device indices.
        device_ids = self.ranks or list(range(self.world_size))
        missing: list[str] = []
        try:
            for src in device_ids:
                for dst in device_ids:
                    if src != dst and not gpu_p2p_access_check(src, dst):
                        missing.append(f"{src}->{dst}")
        except Exception as exc:
            missing.append(f"probe failed: {type(exc).__name__}: {exc}")
        ok = torch.tensor([0 if missing else 1], dtype=torch.int32)
        dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=self.group)
        if int(ok.item()) == 1:
            return None
        detail = ", ".join(missing) if missing else "a peer rank rejected P2P"
        return f"GPU P2P is not available for this TP group ({detail})"

    def _allocate(self) -> str | None:
        from vllm import _custom_ops as ops

        layout = slab_layout(self.max_numel, self.world_size)
        err: str | None = None
        ptr = 0
        handle: torch.Tensor | None = None
        try:
            with torch.cuda.device(self.device):
                ptr, handle = ops.allocate_shared_buffer_and_handle(layout.nbytes)
                ptr = int(ptr)
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
        errors = [None] * self.world_size
        dist.all_gather_object(errors, err, group=self.group)
        if any(errors):
            if ptr:
                ops.free_shared_buffer(ptr)
            joined = "; ".join(
                f"rank {i}: {item}" for i, item in enumerate(errors) if item
            )
            return f"IPC slab allocation failed: {joined}"

        handles: list[Any] = [None] * self.world_size
        dist.all_gather_object(handles, handle, group=self.group)
        opened: list[int] = []
        err = None
        try:
            for index, peer_handle in enumerate(handles):
                if index == self.rank:
                    opened.append(ptr)
                else:
                    opened.append(int(ops.open_mem_handle(peer_handle)))
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
        errors = [None] * self.world_size
        dist.all_gather_object(errors, err, group=self.group)
        if any(errors):
            self._peer_ptrs = opened
            self._local_ptr = ptr
            self._release_storage()
            joined = "; ".join(
                f"rank {i}: {item}" for i, item in enumerate(errors) if item
            )
            return f"IPC peer mapping failed: {joined}"

        self._peer_ptrs = opened
        self._local_ptr = ptr
        self._layout = layout
        try:
            with torch.cuda.device(self.device):
                self._state = torch.zeros(3, dtype=torch.int64, device=self.device)
                self._state[2] = self._timeout_ns
                self._partial = torch.empty(
                    self.max_numel, dtype=torch.float32, device=self.device
                )
                torch.cuda.synchronize(self.device)
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
        errors = [None] * self.world_size
        dist.all_gather_object(errors, err, group=self.group)
        if any(errors):
            joined = "; ".join(
                f"rank {i}: {item}" for i, item in enumerate(errors) if item
            )
            return f"workspace init failed: {joined}"
        return None

    def should_use(self, inp: torch.Tensor) -> bool:
        if self.disabled or self._released or self._layout is None:
            return False
        if not isinstance(inp, torch.Tensor):
            return False
        return accepts_tensor(
            dtype=inp.dtype,
            nbytes=inp.numel() * inp.element_size(),
            numel=inp.numel(),
            contiguous=inp.is_contiguous(),
            cuda=inp.is_cuda,
            max_bytes=self.max_bytes,
            max_numel=self.max_numel,
        )

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        """Out-of-place all-reduce with no host sync, so graphs can replay it."""
        assert self._state is not None
        assert self._partial is not None
        assert self._layout is not None
        out = torch.empty_like(inp)
        torch.ops._C_custom_ar.pcie_xor_all_reduce(
            self._peer_ptrs,
            self._state,
            self._partial,
            inp,
            out,
            int(self.rank),
            int(self.world_size),
            int(self._layout.flag_base),
            int(self._layout.payload_base),
            int(self._layout.slot_stride),
        )
        return out

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if (
            self.group is not None
            and dist.is_available()
            and dist.is_initialized()
            and (self._local_ptr or self._peer_ptrs)
        ):
            try:
                dist.barrier(group=self.group)
            except Exception:
                logger.debug("PCIe XOR all-reduce close skipped its barrier")
        self._release_storage()

    def _release_storage(self) -> None:
        if self._released:
            return
        self._released = True
        self.disabled = True
        peers = list(self._peer_ptrs)
        local = self._local_ptr
        rank = self.rank
        self._peer_ptrs = []
        self._local_ptr = 0
        self._state = None
        self._partial = None
        try:
            from vllm import _custom_ops as ops
        except Exception:
            return
        for index, ptr in enumerate(peers):
            if index == rank or not ptr:
                continue
            try:
                ops.pcie_xor_close_peer(int(ptr))
            except Exception:
                logger.debug("PCIe XOR all-reduce failed to close a peer mapping")
        if local:
            try:
                ops.free_shared_buffer(int(local))
            except Exception:
                logger.debug("PCIe XOR all-reduce failed to free its slab")

    def __del__(self) -> None:
        try:
            self._release_storage()
        except Exception:
            return


def try_create_pcie_xor_allreduce(
    group: ProcessGroup | None,
    device: torch.device | int | str | None,
    unique_name: str,
    ranks: list[int] | None = None,
) -> PcieXorAllReduce | None:
    """Attach the opt-in collective, or return None and keep NCCL."""
    action, reason = precheck(unique_name, device)
    if action == "skip":
        return None
    if action == "fallback":
        assert reason is not None
        _log_fallback(group, reason)
        return None
    if group is None or device is None or ranks is None:
        _log_fallback(group, "TP group metadata was missing")
        return None
    comm = PcieXorAllReduce(group, _coerce_device(device), ranks)
    if comm.disabled:
        comm.close()
        return None
    return comm
