# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired posted-write XOR-butterfly all-reduce schedule.

This is the numeric and layout contract for
``csrc/libtorch_stable/pcie_xor_all_reduce.cu``. It is pure PyTorch so the
reduction order can be tested without a GPU.

Each round ``r`` pairs rank ``i`` with ``i XOR (1 << r)`` and both ranks
replace their fp32 partial with ``lower_rank_partial + higher_rank_partial``.
fp16 and bf16 promote exactly to fp32, so posting the original bits on round
0 and fp32 partials after that is the same sum as promoting first. The tree is
balanced, not a left fold, and every rank produces the same bits. It will not
match NCCL's ring or tree bit-for-bit; bf16/fp16 outputs should still match
within a small rounding tolerance.
"""

from dataclasses import dataclass

import torch

ALGORITHM = "paired-posted-write-xor-butterfly"
SUPPORTED_WORLD_SIZES = (2, 4, 8)
SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
# 1 MiB covers DSV4.1-Flash decode: hidden 5120 bf16, 64 tokens (C16, k=3)
# is 640 KiB. Larger prefill all-reduces stay on NCCL.
DEFAULT_MAX_BYTES = 1 << 20
DEFAULT_TIMEOUT_MS = 2000
# Cap the IPC slab. 16 MiB of input is already past the latency regime.
MAX_CONFIG_BYTES = 16 << 20
FLAG_BYTES = 128
ALIGN = 256
_DTYPE_NAMES = {
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
    torch.float32: "fp32",
}


@dataclass(frozen=True)
class SlabLayout:
    """Byte layout of one rank's IPC slab.

    Two parity banks keep epoch ``E+1`` from overwriting bank ``E`` while a
    peer may still be reading it. A rank only starts ``E+2`` after ``E+1``
    has consumed both directions of ``E+1``, which itself waited for ``E``.
    """

    world_size: int
    rounds: int
    max_numel: int
    flag_base: int
    payload_base: int
    slot_stride: int
    nbytes: int

    def flag_offset(self, parity: int, round_idx: int) -> int:
        index = parity * self.rounds + round_idx
        return self.flag_base + index * FLAG_BYTES

    def payload_offset(self, parity: int, round_idx: int) -> int:
        index = parity * self.rounds + round_idx
        return self.payload_base + index * self.slot_stride


def xor_partner(rank: int, round_idx: int) -> int:
    """Partner rank for one butterfly round. ``round_idx`` starts at 0."""
    return rank ^ (1 << round_idx)


def parse_positive_int(name: str, default: int) -> int:
    """Return ``default`` when unset, the parsed value, or ``-1`` if invalid."""
    import os

    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw.strip(), 10)
    except ValueError:
        return -1


def slab_layout(max_numel: int, world_size: int) -> SlabLayout:
    """IPC slab layout shared with the CUDA kernel's flag and payload math."""
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"world_size={world_size} is not supported")
    if max_numel <= 0:
        raise ValueError(f"max_numel={max_numel} must be positive")
    rounds = world_size.bit_length() - 1
    flag_bytes = 2 * rounds * FLAG_BYTES
    payload_base = (flag_bytes + ALIGN - 1) & ~(ALIGN - 1)
    slot_stride = (max_numel * 4 + ALIGN - 1) & ~(ALIGN - 1)
    nbytes = payload_base + (2 * rounds * slot_stride)
    return SlabLayout(
        world_size=world_size,
        rounds=rounds,
        max_numel=max_numel,
        flag_base=0,
        payload_base=payload_base,
        slot_stride=slot_stride,
        nbytes=nbytes,
    )


def max_numel_for_bytes(max_bytes: int) -> int:
    """Largest element count a 2-byte input of ``max_bytes`` can hold.

    fp32 inputs are limited by the byte cap, so they fit in this slot.
    """
    return max_bytes // 2


def butterfly_partials(shards: list[torch.Tensor]) -> list[torch.Tensor]:
    """Per-rank fp32 partials after the full XOR schedule.

    Every returned tensor is equal. ``shards`` must all have the same shape
    and a supported dtype. The upcast to fp32 happens before any add.
    """
    if not shards:
        raise ValueError("butterfly requires at least one shard")
    world = len(shards)
    if world not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"world_size={world} is not supported")
    shape = shards[0].shape
    dtype = shards[0].dtype
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"dtype={dtype} is not supported")
    partials = []
    for shard in shards:
        if shard.shape != shape or shard.dtype != dtype:
            raise ValueError("shards must share shape and dtype")
        partials.append(shard.detach().to(torch.float32))
    originals = [p.clone() for p in partials]
    rounds = world.bit_length() - 1
    for round_idx in range(rounds):
        nxt: list[torch.Tensor | None] = [None] * world
        for rank in range(world):
            partner = xor_partner(rank, round_idx)
            # Round 0 reads the original values. Later rounds read partials.
            # The upcast is exact for fp16/bf16, so this matches a kernel that
            # posts raw bits on round 0 and fp32 partials afterwards.
            if round_idx == 0:
                local = originals[rank]
                remote = originals[partner]
            else:
                local = partials[rank]
                remote = partials[partner]
            if rank < partner:
                nxt[rank] = local + remote
            else:
                nxt[rank] = remote + local
        partials = [p for p in nxt if p is not None]
        if len(partials) != world:
            raise RuntimeError("butterfly round dropped a rank")
    return partials


def reduce_xor_butterfly(shards: list[torch.Tensor]) -> torch.Tensor:
    """All-reduce ``shards`` with the XOR-butterfly numeric contract."""
    reduced = butterfly_partials(shards)[0]
    return reduced.to(shards[0].dtype)


def accepts_tensor(
    *,
    dtype: torch.dtype,
    nbytes: int,
    numel: int,
    contiguous: bool,
    cuda: bool,
    max_bytes: int,
    max_numel: int,
) -> bool:
    """Host-side gate. False means the caller keeps the NCCL path for this call."""
    return bool(
        cuda
        and contiguous
        and numel > 0
        and dtype in SUPPORTED_DTYPES
        and nbytes <= max_bytes
        and numel <= max_numel
    )


def format_attach_line(
    *,
    world_size: int,
    max_bytes: int,
    max_numel: int,
    rounds: int,
) -> str:
    dtypes = ",".join(_DTYPE_NAMES[dtype] for dtype in SUPPORTED_DTYPES)
    return (
        "DSV41 PCIe XOR all-reduce attached: "
        f"world_size={world_size} algorithm={ALGORITHM} "
        f"dtypes={dtypes} max_bytes={max_bytes} max_numel={max_numel} "
        f"rounds={rounds}. Larger inputs stay on NCCL."
    )


def format_fallback_line(reason: str) -> str:
    return f"DSV41 PCIe XOR all-reduce disabled: {reason}. Falling back to NCCL."
