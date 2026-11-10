# DeepSeek-V4.1-Flash on 8x RTX PRO 5000: performance probe runbook

This runbook is step 1 of the RTX PRO 5000 optimization plan: measure where the
decode step time goes before rewriting any kernel. The current code-based
estimate leaves roughly half of each step unexplained by GEMM bandwidth, so the
rest of the plan is ordered by what this probe finds.

Base serve script: `deepseek_v4_1_flash_yzss_keep_tp4pp2.sh` (TP4 x PP2,
Marlin, DSpark k=3, Engram on NVMe). Run every A/B with the same benchmark
protocol as the earlier try-series (same prompts, warm-up, temperature 0), so
results stay comparable.

## 0. Prerequisites

- Code: `rtxpro5000` + PR #20 + this PR.
- Before the first start with PR #20, stop the server and delete the old
  Engram files (new code never reads them, and both copies would need disk
  space at once):

  ```bash
  ls /data/engram/engram_*.bin | grep -v '_pid' | xargs -r rm -v
  ```

- `nsys` (Nsight Systems) on the box, matching the driver.

## 1. Static facts from the startup log (no extra run needed)

Grep the normal startup log of the baseline serve and report each line:

```bash
LOG=<serve log>
grep -E "for MXFP8 GEMM|MXFP8.*kernel" $LOG | sort | uniq -c
grep -E "Engram table|engram" $LOG | head -20
grep -E "Auto-enabling VLLM_USE_BREAKABLE_CUDAGRAPH|cudagraph_mode|Capturing CUDA graphs" $LOG | head
grep -E "max_num_batched_tokens|Chunked prefill" $LOG | head -5
grep -E "DPSK_|DSpark|moe_backend|Using .* MoE" $LOG | sort | uniq -c | head -40
grep -E "layers? .*PP|pipeline|partition" $LOG | head -10
```

Most important: which MXFP8 linear kernel is in use (FlashInfer CUTLASS = FP8
weights, Emulation = BF16 weights). That decides how much the dense-GEMM work
can gain.

## 2. Baseline benchmark plus page-fault counter

For each point record tok/s, TTFT p50/p99, TPOT p50/p99, acc_len and the major
page-fault delta:

| Point | Input | Output | Concurrency |
|---|---|---|---|
| S-C1 | 56 | 512 | 1 |
| S-C16 | 56 | 512 | 16 |
| L-C1 | 8K | 256 | 1 |
| XL-C1 | 60K | 256 | 1 |

```bash
f0=$(awk '/pgmajfault/{print $2}' /proc/vmstat)
# ... run one benchmark point ...
f1=$(awk '/pgmajfault/{print $2}' /proc/vmstat); echo "majfault delta $((f1-f0))"
```

A non-zero steady-state delta during S-C1/S-C16 means Engram lookups are
going to NVMe.

## 3. nsys trace with NVTX ranges (C1 and C16)

`DPSK_NVTX=1` turns on the ranges added by this PR. `--profiler-config
{"profiler":"cuda"}` makes `/start_profile` call `cudaProfilerStart` in every
worker, so nsys only records the window you choose.

| NVTX range | Meaning |
|---|---|
| `dpsk:worker_execute_model` / `dpsk:worker_sample_tokens` | Per-stage step time |
| `dpsk:prepare_inputs` | CPU input and attention-metadata prep |
| `dpsk:forward_piecewise` / `_full_graph` / `_eager` | Model forward |
| `dpsk:breakable_replay graphs=N eager=M` | One breakable-graph replay |
| `dpsk:attn_eager_break`, `dpsk:sparse_indexer` | Per-layer eager breaks |
| `dpsk:engram_lookup`, `dpsk:engram_wait` | Engram gather (stage 0) |
| `dpsk:pp_send`, `dpsk:pp_send_wait`, `dpsk:pp_recv`, `dpsk:pp_recv_wait` | PP hop |
| `dpsk:sample`, `dpsk:draft_propose` | Verify/sampling and DSpark draft (stage 1) |

Capture one run per concurrency (C1, then C16), each a fresh server:

```bash
export DPSK_NVTX=1
nsys profile -o /data/prof/probe_c1 \
  -t cuda,nvtx,osrt --trace-fork-before-exec=true --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  bash deepseek_v4_1_flash_yzss_keep_tp4pp2.sh \
  --profiler-config '{"profiler": "cuda", "delay_iterations": 50, "max_iterations": 200}'
```

Then, in another shell, once the server is ready:

1. Warm up with the usual protocol for that point.
2. Start the benchmark (S-C1 or S-C16) and, while it runs:

   ```bash
   curl -X POST localhost:8000/start_profile
   sleep 20
   curl -X POST localhost:8000/stop_profile
   ```

3. Stop the server so nsys finalizes the report.

Report back, per `.nsys-rep`:

```bash
nsys stats -r nvtx_pushpop_sum,cuda_gpu_kern_sum,cuda_api_sum,osrt_sum \
  --format csv -o /data/prof/probe_c1 /data/prof/probe_c1.nsys-rep
```

- The four CSVs (or the top 40 rows of each), split by process. Stage 0 is
  ranks 0-3 and stage 1 is ranks 4-7; one rank per stage is enough.
- From the timeline: for 5 consecutive steps on one stage-0 rank and one
  stage-1 rank, the start and end times of `dpsk:worker_execute_model`, the
  GPU busy time inside it, and the gap to the next step.
- Whether `dpsk:engram_lookup` waits on the GPU (look for
  `cudaStreamSynchronize` or `cudaMemcpy` inside it) and how long it takes.

## 4. Configuration A/B (one change at a time, same protocol as step 2)

For each run, do the usual correctness smoke test (hello / 391 / Blue +
grammar) and record the step-2 table.

1. **Engram pinned with exact-size registration** (removes the per-step host
   sync on stage 0):

   ```
   --engram-config '{"cpu_offload": true, "use_thp": true}'
   ```

   - Do not set `storage_path`.
   - Watch `free -g` and `dmesg | grep -i oom` during start-up.
   - Expected pinned size is about 97 GiB in total (MXFP4 tables).
   - If the log says `Engram huge-page allocation failed ... using pinned
     memory`, the fallback is the power-of-two pinned allocator. That rounds
     each shard up and can OOM. Stop and report the log line.
   - Check `/sys/kernel/mm/transparent_hugepage/enabled` is `always` or
     `madvise` first.
2. `DPSK_PP1_FULL_CG=1`: stage-1 decode runs as FULL graphs. Check the log for
   `last PP rank PIECEWISE -> FULL_AND_PIECEWISE`, or the `ignored` line.
3. `DPSK_PP_NO_AG=1`.
4. `DPSK_PREP_AHEAD=1 DPSK_PP_META_CACHE=1`. This combination needs PR #20.
   Run it for at least 30 minutes of mixed load to confirm it does not hang.
5. `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`.
6. PP layer split with `VLLM_PP_LAYER_PARTITION="a,b"`, where a + b equals
   `num_hidden_layers`.
   - Use the step-3 per-stage busy times to pick the direction: move layers
     away from the slower stage, one layer at a time.
   - Moving Engram layer 14 off stage 0 is out of scope here.

Run 2-6 on top of whichever Engram setting wins in 1.

## 5. What to send back

- Step 1 grep output.
- Step 2 table (baseline).
- Step 3 CSVs and the 5-step timing for each stage.
- Step 4 table: one row per change, with the same columns as step 2 plus the
  smoke-test result.
