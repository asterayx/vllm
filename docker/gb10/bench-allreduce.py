# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NCCL all-reduce latency for the decode message sizes on a 2-node GB10 pair.

Run inside the serve image on both nodes (see bench-allreduce.sh). Prints
the average time per all-reduce for the tensors a TP=2 decode step sends:
[real_tokens, hidden] and [padded_tokens, hidden] bf16, plus a few larger
sizes, so NCCL_* settings can be compared without restarting the server.
"""

import argparse
import os
import time

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile

INTERESTING_ENV = (
    "NCCL_PROTO",
    "NCCL_ALGO",
    "NCCL_IB_QPS_PER_CONNECTION",
    "NCCL_IB_SPLIT_DATA_ON_QPS",
    "NCCL_BUFFSIZE",
    "NCCL_NET_GDR_LEVEL",
    "NCCL_IB_GID_INDEX",
    "NCCL_IB_TC",
    "NCCL_IB_HCA",
    "NCCL_MIN_NCHANNELS",
    "NCCL_MAX_NCHANNELS",
    "NCCL_IB_ADAPTIVE_ROUTING",
)


def bench(shape: tuple[int, int], iters: int, use_graph: bool) -> float:
    x = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    for _ in range(10):
        dist.all_reduce(x)
    torch.accelerator.synchronize()
    if use_graph:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                dist.all_reduce(x)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(20):
                dist.all_reduce(x)
        g.replay()
        torch.accelerator.synchronize()
        dist.barrier()
        t = time.perf_counter()
        for _ in range(iters // 20):
            g.replay()
        torch.accelerator.synchronize()
        return (time.perf_counter() - t) / (iters // 20 * 20) * 1e6
    dist.barrier()
    t = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(x)
    torch.accelerator.synchronize()
    return (time.perf_counter() - t) / iters * 1e6


def kernel_time(shape: tuple[int, int], iters: int = 50, skip: int = 20) -> float:
    """Average NCCL kernel duration (us) as CUPTI sees it, no CPU overhead.

    An NCCL kernel spins until the peer's data arrives, so the first
    launches after the profiler starts (the ranks start it at different
    times) are dropped and the ranks are barriered inside the profile.
    """
    x = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    for _ in range(10):
        dist.all_reduce(x)
    torch.accelerator.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(skip):
            dist.all_reduce(x)
        torch.accelerator.synchronize()
        dist.barrier()
        for _ in range(iters):
            dist.all_reduce(x)
        torch.accelerator.synchronize()
    kernels = sorted(
        (
            e
            for e in prof.events()
            if e.device_type.name == "CUDA" and "nccl" in e.name.lower()
        ),
        key=lambda e: e.time_range.start,
    )
    if len(kernels) < iters:
        return float("nan")
    # dist.barrier() is itself an NCCL kernel; the last `iters` are the loop.
    tail = kernels[-iters:]
    return sum(e.time_range.elapsed_us() for e in tail) / len(tail)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--iters", type=int, default=400)
    parser.add_argument(
        "--graph",
        action="store_true",
        help="also time the all-reduce inside a CUDA graph (kernel time is "
        "the same; this only checks that capture works)",
    )
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.accelerator.set_device_index(0)
    if rank == 0:
        print(f"world {dist.get_world_size()}, nccl {torch.cuda.nccl.version()}")
        for k in INTERESTING_ENV:
            if k in os.environ:
                print(f"  {k}={os.environ[k]}")
    rows = [
        (4, "real DSpark k=3 target tokens"),
        (16, "SM12x padded step"),
        (64, "4-request batched step"),
        (256, "prefill chunk"),
        (2048, "long prefill chunk"),
    ]
    for use_graph in (False, True) if args.graph else (False,):
        if rank == 0:
            print(f"\n{'CUDA graph' if use_graph else 'eager'} all_reduce bf16:")
        for tokens, label in rows:
            shape = (tokens, args.hidden)
            us = bench(shape, args.iters, use_graph)
            kus = kernel_time(shape) if not use_graph else float("nan")
            if rank == 0:
                mb = tokens * args.hidden * 2 / 2**20
                print(
                    f"  [{tokens:5d}, {args.hidden}] {mb:7.2f} MB  wall {us:7.1f} us"
                    f"  kernel {kus:7.1f} us ({mb / kus * 1e6 / 1024:6.2f} GB/s)"
                    f"  {label}"
                )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
