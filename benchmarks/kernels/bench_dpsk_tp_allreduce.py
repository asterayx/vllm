# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NCCL all-reduce latency at DeepSeek-V4.1 decode shapes.

Times [tokens, hidden] bf16 all-reduces replayed from a CUDA graph, the way
the decode step issues them, plus eager calls for reference. Run it with the
serving process stopped, once per GPU set to compare:

    torchrun --nproc-per-node 8 benchmarks/kernels/bench_dpsk_tp_allreduce.py
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 \
        benchmarks/kernels/bench_dpsk_tp_allreduce.py
"""

import argparse
import os

import torch
import torch.distributed as dist


def _time_ms(fn, iters: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    dist.barrier()
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _max_across_ranks(value: float) -> float:
    t = torch.tensor([value], device="cuda", dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 4, 16, 64, 256])
    parser.add_argument(
        "--ops", type=int, default=88, help="all-reduces per graph (one C1 cycle)"
    )
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    world = dist.get_world_size()
    rank = dist.get_rank()

    rows = []
    for tokens in args.tokens:
        # Zeros stay zero under repeated sums, so no inf/NaN creeps in.
        x = torch.zeros(tokens, args.hidden, dtype=torch.bfloat16, device="cuda")
        for _ in range(20):
            dist.all_reduce(x)
        eager_us = _time_ms(lambda x=x: dist.all_reduce(x), args.iters * 4) * 1e3

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                dist.all_reduce(x)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(args.ops):
                dist.all_reduce(x)
        graph.replay()
        graph_us = _time_ms(graph.replay, args.iters) * 1e3 / args.ops

        rows.append(
            (
                tokens,
                tokens * args.hidden * 2 / 1024,
                _max_across_ranks(eager_us),
                _max_across_ranks(graph_us),
            )
        )
        del graph

    if rank == 0:
        devices = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
        print(f"world={world} CUDA_VISIBLE_DEVICES={devices} hidden={args.hidden}")
        print("tokens | KiB | eager us/op | graph us/op | graph ms per cycle")
        for tokens, kib, eager_us, graph_us in rows:
            print(
                f"{tokens:6d} | {kib:7.1f} | {eager_us:11.1f} | {graph_us:11.1f} | "
                f"{graph_us * args.ops / 1e3:8.3f}"
            )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
