# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two-node TP2 small-message NCCL graph benchmark; report worst-rank samples."""

import argparse
import datetime
import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--master-addr", default=os.environ.get("MASTER_ADDR", "127.0.0.1")
    )
    parser.add_argument("--port", type=int, default=29629)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://{args.master_addr}:{args.port}",
        rank=args.rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=120),
    )
    comm = PyNcclCommunicator(dist.group.WORLD, device=0)
    assert comm.available and not comm.disabled
    records = []
    sync = torch.ones(1, device="cuda")
    sync_out = torch.empty_like(sync)
    for operation in ("all_reduce", "all_gather"):
        for count in (1, 1280, 2560, 7680, 10240, 30720, 124160, 372480):
            graphs, inputs, outputs = [], [], []
            for _ in range(2):
                x = torch.full(
                    (count,), args.rank + 1, device="cuda", dtype=torch.bfloat16
                )
                y = torch.empty(
                    count * (2 if operation == "all_gather" else 1),
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                inputs.append(x)
                outputs.append(y)

                def run(x=x, y=y, operation=operation):
                    if operation == "all_reduce":
                        comm.all_reduce(x, y)
                    else:
                        comm.all_gather(y, x)

                dist.barrier()
                for _ in range(5):
                    run()
                torch.accelerator.synchronize()
                expected = torch.full_like(y, 3)
                if operation == "all_gather":
                    expected[:count] = 1
                    expected[count:] = 2
                torch.testing.assert_close(y, expected, rtol=0, atol=0)
                graph = torch.cuda.CUDAGraph()
                start = torch.cuda.Event(enable_timing=True, external=True)
                end = torch.cuda.Event(enable_timing=True, external=True)
                dist.barrier()
                with torch.cuda.graph(graph):
                    # Keep CPU submission gaps outside the measured device interval.
                    comm.all_reduce(sync, sync_out)
                    start.record()
                    run()
                    end.record()
                y.zero_()
                graph.replay()
                torch.accelerator.synchronize()
                torch.testing.assert_close(y, expected, rtol=0, atol=0)
                graphs.append((graph, start, end))
            timings = []
            for index in range(70):
                graph, start, end = graphs[index % 2]
                graph.replay()
                end.synchronize()
                if index >= 10:
                    timings.append(start.elapsed_time(end) * 1000)
            gathered = [None, None]
            dist.all_gather_object(gathered, timings)
            maxima = [max(a, b) for a, b in zip(*gathered)]
            row = dict(
                operation=operation,
                input_bytes=count * 2,
                median_worst_rank_us=statistics.median(maxima),
                p90_worst_rank_us=sorted(maxima)[53],
                rank_samples_us=gathered,
                env={
                    k: os.environ.get(k)
                    for k in [
                        "NCCL_ALGO",
                        "NCCL_PROTO",
                        "NCCL_IB_HCA",
                        "NCCL_SOCKET_IFNAME",
                    ]
                },
            )
            records.append(row)
            if args.rank == 0:
                (args.output / "results.json").write_text(json.dumps(records, indent=2))
                print(
                    json.dumps(
                        {k: v for k, v in row.items() if k != "rank_samples_us"}
                    ),
                    flush=True,
                )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
