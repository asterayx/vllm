# Qwen3.8-Flash-Next NVFP4 on two DGX Sparks

This branch is based on community vLLM **v0.29.0rc4**, commit
`d2906cc1958658f296aeee8b248deea684f56add`. Each node supplies one GB10 GPU;
tensor parallelism spans the two nodes.

## Checkpoint loading

`RadixArk/Qwen3.8-Flash-Next-NVFP4` stores routed experts in NVFP4, but its
excluded PLE embedding table is FP8 with a global scale. The rc4 selector
does not register that scale under a ModelOpt NVFP4 configuration, so loading
fails with a missing `ngram_embedding.weight_scale` parameter.

This branch backports the existing community work rather than proposing
a duplicate upstream fix:

- [#54722](https://github.com/vllm-project/vllm/pull/54722): FP32 scale storage
  and detection of a missing global scale.
- [#54882](https://github.com/vllm-project/vllm/pull/54882): mixed ModelOpt
  checkpoint dispatch.
- [#55334](https://github.com/vllm-project/vllm/pull/55334): select the FP8 PLE
  loader when ModelOpt excludes the table and `ple_embedding_dtype` declares FP8.

An additional regression test exercises actual PLE construction and loading on
both TP ranks, including checkpoint shards crossing a TP boundary. With the
original rc4 implementation, both ranks reproduce the missing-scale exception.

## Isolated installation

Use a separate worktree and environment on each Spark. The existing checkout,
environment, and model files do not need modification. Run from the worktree:

```bash
uv venv --python 3.12 .venv
VLLM_USE_PRECOMPILED=1 \
SETUPTOOLS_SCM_PRETEND_VERSION=0.29.0rc4+spark.tp2 \
VLLM_PRECOMPILED_WHEEL_COMMIT=d2906cc1958658f296aeee8b248deea684f56add \
VLLM_PRECOMPILED_WHEEL_VARIANT=cu130 \
uv pip install -e . --torch-backend=cu130 --index-strategy unsafe-best-match
```

Both nodes need the same model revision and complete local weights. A Hugging
Face cache snapshot can be passed directly as `MODEL_PATH`.

The interpreter's matching Python development headers are required, including
for Triton's runtime compilation. For an isolated deployment without system
package changes, place `python3.12/` and the matching architecture-specific
header directory under `.venv/include/`. The launcher uses those headers when
`.venv/include/python3.12/Python.h` exists. Both the interpreter version and
architecture must match the headers.

The launcher puts `.venv/bin` on `PATH` so FlashInfer can find `ninja` during
kernel compilation. It defaults to four compilation jobs; override `MAX_JOBS`
to adjust this limit.

## Persist the head network configuration

The tested head uses NetworkManager. For these three dedicated links, save
static addresses and automatic activation with the existing connection names:

```bash
sudo nmcli connection modify enP7s7 \
  connection.autoconnect yes connection.autoconnect-priority 100 \
  ipv4.method manual ipv4.addresses 192.168.99.10/24 \
  ipv4.gateway "" ipv4.never-default yes

sudo nmcli connection modify enP2p1s0f1np1 \
  connection.autoconnect yes connection.autoconnect-priority 100 \
  ipv4.method manual ipv4.addresses 192.168.101.10/24 \
  ipv4.gateway "" ipv4.never-default yes \
  802-3-ethernet.mtu 9000

sudo nmcli connection modify enp1s0f1np1 \
  connection.autoconnect yes connection.autoconnect-priority 100 \
  ipv4.method manual ipv4.addresses 192.168.100.10/24 \
  ipv4.gateway "" ipv4.never-default yes \
  802-3-ethernet.mtu 9000
```

`/24` is netmask `255.255.255.0`. These profiles do not install a default
route; the tested head uses Wi-Fi for its default route. NetworkManager is
already enabled at boot. `connection modify` saves persistently by default;
when the current addresses and MTUs are already correct, no immediate
reconnection is needed. See the [NetworkManager nmcli documentation](https://networkmanager.pages.freedesktop.org/NetworkManager/NetworkManager/nmcli.html).
Interface names are case-sensitive. These commands describe the head only.

## Launch

Run rank 1 on the worker, then rank 0 on the head. Set the interconnect IPs and
interface names for your machines. The following addresses match the test setup:

```bash
# Worker: aitopatom-da17
MODEL_PATH="$HOME/models/Qwen3.8-Flash-Next-NVFP4-codex-rc4" \
MASTER_ADDR=192.168.100.10 VLLM_HOST_IP=192.168.100.11 \
bash examples/online_serving/qwen38_nvfp4_spark_tp2.sh 1
```

```bash
# Head: aitopatom-d6d3
MODEL_PATH="$HOME/models/Qwen3.8-Flash-Next-NVFP4" \
MASTER_ADDR=192.168.100.10 VLLM_HOST_IP=192.168.100.10 \
bash examples/online_serving/qwen38_nvfp4_spark_tp2.sh 0
```

The script defaults to `enp1s0f1np1`, rendezvous port `29529`, and an API on
`127.0.0.1:18029`. Override `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`,
`MASTER_PORT`, `API_HOST`, or `API_PORT` as needed.

Automatic tool choice uses `--enable-auto-tool-choice` and
`--tool-call-parser qwen3_coder`, matching the checkpoint's XML function-call
format. `--reasoning-parser qwen3` separates thinking from content and tools.
For thinking-enabled requests, the checkpoint template accepts reasoning
effort `low`, `medium`, or `xhigh`; `none` disables thinking through vLLM.
Do not select `high`, which this checkpoint's template rejects.

Serving verification covered automatic Chat Completions tool selection with
thinking disabled, a tool-result follow-up, streamed tool selection with low
reasoning effort, and streamed Responses function calls. All returned the
expected function and valid JSON arguments; reasoning aliases matched in the
stream. The public Chat Completions endpoint also returned HTTP 200 with
`finish_reason="tool_calls"` for `tool_choice="auto"`.

Caches stay inside this worktree. The configuration uses decode CUDA Graphs, 512K context,
8192 batched tokens, four concurrent sequences, and a fixed 16 GiB KV cache per
GPU. The explicit cache budget takes precedence over the memory-utilization
fraction for KV allocation. Spark shares GPU and system memory; leaving the
cache budget implicit allocated about 33 GiB per GPU in the initial test and
left only about 10 GiB available on the head during serving.
Set `MAX_MODEL_LEN`, `MAX_NUM_BATCHED_TOKENS`, `MAX_NUM_SEQS`, and
`KV_CACHE_MEMORY_BYTES` identically on both nodes when tuning. Increase the cache
budget when increasing context length or concurrency, and check available system
memory during startup and serving.

The 2026-09-06 batch-size comparison retained MTP 2 and all other settings.
Each row used the same synthetic prompt, temperature 0, seed 42, thinking
disabled, and 256 generated tokens through the loopback Chat Completions API.
Cold time to first content token was:

| Batched tokens | 8,154 input tokens | 65,502 input tokens | 262,110 input tokens |
| --- | --- | --- | --- |
| 2048 | 2.92 s | 24.47 s | 115.99 s |
| 4096 | 2.92 s | 23.04 s | 109.72 s |
| 8192 | 3.33 s | 22.12 s | 106.03 s |

8192 reduced the measured 64K and 256K cold first-token latency by 9.6% and
8.6%, respectively, while decode remained around 41–43 tokens/s. The initial
8K request did not improve; its immediate repeat took 2.78 s. These are single
paired observations, not percentile estimates or concurrent-load benchmarks.
The baseline service was already warm; candidates restarted before testing.

Repeated identical prompts did not increase prefix-cache hit counters, and
their prefill times were essentially unchanged. The old cumulative hit ratio
does not establish effective reuse for these MTP requests. Those baseline
observations motivated the prefix-cache fixes described below.

The selected 8192 configuration also passed a proxy Responses request with
523,760 input tokens and 17 output tokens in 253.82 s, recovering all three
codes at approximately 10%, 50%, and 90% of the input. MTP integer generation,
tool-result follow-up, streamed Chat Completions tools and streamed Responses
tools passed. This is a long-context smoke test, not a broad accuracy eval.
Per-round request JSON, SSE responses, timing/usage JSON, metrics snapshots,
startup logs and two-second memory samples are retained in the experiment
directory `.run/perf-20260906/` on the test hosts and copied to the local tree.

512K means 524,288 total input and output tokens per request. The checkpoint
declares 262,144 positions; above that length the launcher applies a 2x YaRN
RoPE override while preserving interleaved multimodal RoPE. This is an
experimental extension, not the checkpoint's native context guarantee. The
checkpoint files are not modified. Set `MAX_MODEL_LEN=262144` or less to use
the checkpoint's original RoPE. Four concurrent sequences share the cache;
this budget does not promise four simultaneous full-length 512K requests.

The launcher defaults to MTP with two speculative tokens. Set `MTP_TOKENS=1`
for one token or `MTP_TOKENS=0` to disable it, identically on both nodes. The
checkpoint contains one MTP layer (31 BF16 tensors, 4.856 GiB in total); the
two-token setting reuses that layer for successive draft steps.

The requested trial order was two tokens, then one only on failure, then
disabled only if both failed. The two-token run started and passed actual
generation, automatic tool calls, tool-result follow-up, streaming reasoning,
and Responses function-call checks. Both draft positions recorded accepted
tokens, confirming active two-token speculation. The service stayed healthy,
so no one-token trial or performance comparison was performed.

Fused multi-step drafting is unavailable for the QSA state backend; vLLM
rebuilds attention metadata between draft steps. This fallback remains in the
Graph configuration. Two-token speculation was confirmed by accepted-token
counter increments during the serving tests.

FlashInfer autotuning is disabled in this example. On a repeated two-node
startup, rank 0 hit its cached MoE tactics while rank 1 entered profiling and
waited in `flashinfer.autotuner._profile_single_kernel`'s `all_reduce`. Disabling
this optional tuning avoids that collective mismatch; CUTLASS kernel execution
and the other kernel warmups remain enabled. Performance with heuristic tactics
can differ from a successful autotuned run.

To access the head's loopback API from the local computer, keep an SSH tunnel
open with `ssh -N -L 18029:127.0.0.1:18029 aitopatom-d6d3`, then use
`http://127.0.0.1:18029/v1` with model name `qwen38-nvfp4`.

## MTP prefix reuse and execution modes

The prefix-cache changes identify Qwen MTP attention and QSA groups explicitly,
retain the earlier target Mamba replay checkpoint required by draft lookahead,
and split prefill at Mamba state boundaries. The last point matters because QSA
also registers an 8-token ring while Mamba state blocks span 1600 tokens: using
the minimum cache block for scheduling misses the reusable Mamba checkpoint.
The checkpoint split also handles prompts ending exactly on a block boundary.

The default `EXECUTION_MODE=graph` enables `FULL_DECODE_ONLY` CUDA Graphs
without compilation. Set `EXECUTION_MODE=eager` to use eager execution,
identically on both nodes. Capture sizes are bounded to `[1, 2, 4, 8, 12]`
for this four-sequence, MTP-2 setup. Graph capture reported 0.19 GiB extra memory.

Compilation mode 3 was also tested with one Inductor compilation thread. It
exhausted the available memory before serving; the worker's 8 GiB memory guard
terminated the experiment at approximately 2.8 GiB available. The head's guard
recorded only 286 MiB available and also stopped its test process group; SSH
was temporarily unresponsive. Compilation is therefore not included in the
launcher's supported modes.

With the complete cache fix and Graph mode, immediate identical repeats
reused 6400 / 62400 / 259200 tokens for the same three benchmark inputs:

| Input tokens | Cold first-token latency | Repeated first-token latency |
| --- | --- | --- |
| 8154 | 3.20 s | 0.63 s |
| 65502 | 22.39 s | 1.28 s |
| 262110 | 105.79 s | 2.05 s |

Decode ranged from 41.59 to 46.41 tokens/s across these six requests; the
single-pair measurements do not establish a substantial Graph-only speedup.
The cache correctness checks, MTP generation and all tool/Responses checks
passed. GSM8K remained 15/16 with no invalid output (36.09 s, concurrency 2).
Minimum available memory was 22.95 GiB on the head and 27.36 GiB on the worker.
After restoring the release configuration, the proxy Responses API passed
both 512K recall requests: 523760 input tokens, 17 output tokens, with all three
codes correct. Cold total latency was 256.89 s; the identical repeat reused
521600 tokens and completed in 3.21 s (first content token at 2.92 s).
Both requests recorded 12 draft tokens and 12 accepted tokens, confirming MTP
was active at this extended context length. Final GSM8K remained 15/16 with
no invalid output; all cache and tool checks passed. The public Chat Completions
endpoint returned HTTP 200 and `OK` in 1.13 s. This remains a synthetic long-context
smoke test with experimental 2x YaRN, not a native 512K accuracy guarantee.

Raw requests, SSE, per-request cache/speculation counters, timing, configuration,
source diffs and memory records are saved under
`.run/perf-20260906-cache-graphs/` on the test hosts and local tree, including
unsuccessful experiments. These results are small regression samples.

The affected cache and scheduler suites pass 435 tests, including regression
cases that fail before the replay-retention and heterogeneous-block fixes:

```bash
.venv/bin/python -m pytest tests/v1/core/test_prefix_caching.py \
  tests/v1/core/test_mamba_align_chunk_split.py \
  tests/v1/core/test_kv_cache_utils.py \
  tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py \
  tests/v1/core/test_scheduler.py -q
```

## Verification

```bash
.venv/bin/python -m pytest tests/models/qwen4_exp/test_ple.py \
  tests/models/qwen4_exp/test_config.py -v
pre-commit run --files vllm/models/qwen4_exp/nvidia/ple_layer.py \
  tests/models/qwen4_exp/test_ple.py \
  examples/online_serving/qwen38_nvfp4_spark_tp2.sh
curl --fail http://127.0.0.1:18029/health
curl --fail http://127.0.0.1:18029/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-nvfp4","messages":[{"role":"user","content":"What is 2 + 2?"}],"max_tokens":256,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
.venv/bin/python examples/online_serving/qwen38_nvfp4_spark_eval.py
```

Local validation: 18 PLE tests passed, including both TP2 ranks; all relevant
pre-commit hooks passed. On the head GB10, the combined PLE and model config
suites passed all 25 tests. Both TP ranks loaded the complete checkpoint and
reported approximately 61.73 GiB of model memory per GPU. Both ranks completed
profiling, cache initialization, and kernel warmup. The health endpoint returned
HTTP 200; arithmetic, Chinese explanation, and synthetic red-image recognition
requests returned correct responses.

On both GB10s, the existing `test_flashinfer_fp4_moe_no_graph` reference
check passed for the checkpoint's TP2 expert shape: `n=320`, `k=2560`,
`e=512`, `topk=10`, BF16 input and SiLU activation, with both `m=1` and
`m=16`.

The first serving evaluation answered 15/16 GSM8K questions correctly (93.75%),
with no invalid responses. It used the first 16 test questions, five-shot prompts,
chat completions with thinking disabled, temperature 0, seed 42, a 1024-token
output limit, and two concurrent requests. The evaluation script reuses the
repository's GSM8K prompt builder and scorer and saves per-question outputs to
`.run/gsm8k-16.json`. This is a small regression sample, not a full accuracy or
performance benchmark.

The final 4 GiB cache configuration with autotuning disabled started successfully.
Four simultaneously submitted requests passed: recall from a 14,543-token prompt,
integer multiplication, Chinese translation, and red-image recognition. Available
system memory during these requests was approximately 41 GiB on the head and
46 GiB on the worker, compared with about 10 GiB on the head in the initial
implicit-cache run. The final GSM8K rerun also scored 15/16 (93.75%) with zero
invalid responses: 2,674 output tokens in 70.45 seconds at two-request
concurrency. These timings include request processing and are not a dedicated
throughput benchmark. The health endpoint remained HTTP 200 after evaluation.

On the two test machines, the isolated worktree is
`/home/roccen/src/vllm-qwen38-nvfp4-spark-tp2`. The original
`/home/roccen/src/vllm` checkout is preserved. Rank 0 uses
`/home/roccen/models/Qwen3.8-Flash-Next-NVFP4`; its complete checkpoint was copied
over the interconnect to rank 1 at
`/home/roccen/models/Qwen3.8-Flash-Next-NVFP4-codex-rc4`.

Both machines have the rc4 precompiled extensions and can import CUDA kernels.
The head's editable install generated the version string
`0.29.0rc5.dev0+gd2906cc19.d20260905` from the modified rc4 tree; this is build
metadata, not an rc5 source base. The source baseline and binary wheel are the
rc4 commit recorded above. The installation command pins the displayed version
for future installs.
Worker dependencies were copied from the isolated head environment, followed by
a completed editable install; matching Python headers were copied into its
`.venv/include/` for Triton. The old worker container
`qwen38-nvfp4-tp2-rank1` was stopped with user authorization.

The head's earlier SSH failure coincided with repeated system OOM events in the
kernel journal. During verification, a separate test watchdog terminates only
the test process group if system `MemAvailable` stays below 8 GiB. Compilation
caches were prepared before full loading and `MAX_JOBS=4` was used for serving.

## Compatibility proxy and public access

The Rust proxy in `docker/gb10/compat-proxy` was restored from repository commit
`0648ef21c`. It preserves native `/v1/responses` requests and adds the
`reasoning_content` alias to `reasoning` fields in JSON and SSE responses.
It does not convert Chat Completions into Responses.

Build and start it on the head from the isolated worktree:

```bash
cargo build --locked --release --manifest-path docker/gb10/compat-proxy/Cargo.toml -j4
bash examples/online_serving/qwen38_nvfp4_spark_proxy.sh
```

The deployment path is Cloudflare Tunnel → `127.0.0.1:30000` (proxy) →
`127.0.0.1:18029` (vLLM). The public API base is
`https://token.asterayx.com/v1`, and the served model is `qwen38-nvfp4`.
The proxy wrapper sets the client context limit to 524,288 tokens and the
upstream read timeout to 3,600 seconds. SSE comments keep downstream streams
active every 15 seconds while upstream prefill is silent. Use streaming for
long requests through the public tunnel; non-streaming HTTP requests remain
subject to the tunnel's response timeout.

The 512K startup reported 1,278,714 cache tokens (2.44 maximum-length requests
by the engine's estimate). A streamed Responses request through the local
proxy processed 523,760 input tokens and generated 15 output tokens in
258.52 seconds, with no cached input tokens. It correctly retrieved all three
four-digit codes placed at approximately 10%, 50%, and 90% of a repeated
archive prompt. Head available memory stayed around 28 GiB. This synthetic
retrieval smoke test does not establish accuracy on general 512K documents.
The same full-length payload also completed through the public HTTPS Responses
endpoint with HTTP 200 and all three correct codes. That repeat took 50.84
seconds including upload and reused 523,712 cached input tokens; it is not a
cold-prefill timing.
The same 16-question GSM8K evaluation with YaRN enabled scored 15/16 (93.75%)
with zero invalid responses, matching the original configuration. Its latency
included waiting behind the long prefill and is not a throughput measurement.
All 11 proxy Rust tests passed, including idle-stream heartbeats and preservation
of partial SSE lines, along with repository pre-commit checks.

`https://token.asterayx.com/configs/grok.toml` provides the Grok Build model
configuration. Merge its model block into the existing Grok configuration,
then use `grok --model qwen38-nvfp4`. It selects the `responses` API backend
and reads the API key from `VLLM_API_KEY`.

Proxy validation passed nine Rust release-mode unit tests, including split
UTF-8 SSE chunks and generated model/context settings. Public requests returned
HTTP 200 for models, JSON Responses, streamed Responses, and Chat Completions;
both Responses modes returned `OK`, and Chat Completions preserved `reasoning`
while adding `reasoning_content`. Repository pre-commit checks passed.

## Bounded decode profiling

`benchmarks/benchmark_spark_decode_profile.py` records warmup, baseline,
profiled, and post-capture requests at four context lengths. It starts the
profiler after receiving the first output token, excluding prefill. Requests,
raw SSE, metrics, and timings are saved in the specified output directory.
Run this against an otherwise idle deployment.

On each node, wrap the normal launcher with Nsight Systems, preserving the
environment variables from the launch commands above (`RANK` is 0 or 1):

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --cuda-graph-trace=node --capture-range=cudaProfilerApi \
  --capture-range-end=repeat:4 --kill=none --output="rank${RANK}" \
  bash examples/online_serving/qwen38_nvfp4_spark_tp2.sh "$RANK" \
  --profiler-config '{"profiler":"cuda","delay_iterations":4,"max_iterations":24,"detailed_trace_annotation":true}'
```

Once the API is ready, run on the head:

```bash
.venv/bin/python benchmarks/benchmark_spark_decode_profile.py \
  --model-path "$HOME/models/Qwen3.8-Flash-Next-NVFP4" \
  --output .run/decode-profile/requests
```

Optional `NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,TUNING`, and a
per-rank `NCCL_DEBUG_FILE` retain transport and algorithm-selection evidence.
The four capture ranges are ordered 8K, 64K, 256K, then approximately 512K.
Stop the instrumented deployment normally to finalize the Nsight reports,
then restore the normal launcher. Use `--phases warmup baseline post` to
measure a server without profiler support. Nsight creates a separate process
group for the application; memory guards must cover that group.

Profiled request latency includes collection and trace-flush overhead.
Baseline/post requests in the instrumented process still have injection
libraries loaded; compare against a restored process before attributing a
performance difference to the model. Kernel-time sums include overlapping
streams and must not be interpreted as wall-clock latency. CPU sampling is
disabled in this recipe, so it does not provide CPU stack attribution.

The 2026-09-07 GB10/TP2 run completed all 16 requests (256 output tokens each)
and produced eight valid traces. Each trace contained 24 decode-only steps,
62,448 kernels, and 2,664 NCCL kernels. Collection-disabled baseline/post
decode rates were 41.61/42.22, 42.14/44.15, 41.55/39.29, and 38.97/40.83
tokens/s for 8K, 64K, 256K, and approximately 512K respectively.
These are single-request observations, not a load-test distribution.

Head kernel-time sums per step were approximately 26.6–26.8 ms for BF16
WMMA GEMM, 7.0–7.1 ms for BF16 GEMV, 8.2–8.6 ms for NVFP4 grouped GEMM,
and 4.2–5.9 ms for NCCL. All captured decode collectives used Ring/LL.
At the profiling baseline, the model-specific skinny GEMM dispatch only
enabled SM103 and primarily targeted TP4. The GB10/TP2 follow-up below
measures and validates a separate SM121 dispatch table. Raw traces and request records are retained
under `.run/profile-20260907/` in the deployment experiment archive.

## GB10 TP2 BF16 dispatch

SM121 now uses a separate table of 73 measured `(N, K, M)` plans spanning 14
local projection shapes, reusing the existing CuTeDSL skinny GEMM. Unmeasured
shapes or token counts retain `F.linear`; the SM103 table is unchanged.
Set `VLLM_QWEN4_EXP_SM121_GEMM=0` on both nodes to disable this new path.

The 2026-09-07 cold-L2 CUPTI sweep initially selected 74 plans with more than
10% improvement across both timing rounds. Independent checks on both GB10s
passed against FP32 accumulation; one plan with negligible worker benefit
was removed. The installed 73 plans exactly match the retained measurements.
The first attempt without CUPTI was excluded because event fallback did not
satisfy the cold-L2 measurement contract.

With MTP2, two measured requests after warmup improved from 42.54/41.14 to
45.71/45.75 tokens/s at 8K, and from 42.71/39.75 to 47.00/44.02 at 64K.
The small sample and differing MTP acceptance rates prevent attributing all
of this gain to GEMM. Cold 64K prefill remained approximately 20.1 seconds.
GSM8K first-16/5-shot scored 15/16 with no invalid responses; cache isolation,
continuous generation, tool follow-up, and Chat/Responses smoke checks passed.

The same optimized dispatch was tested with MTP disabled, one draft token,
and two draft tokens. The values below are the mean of two decode requests
after warmup, with 256 output tokens each. All three configurations scored
15/16 on the same GSM8K subset, with no invalid responses.

| Draft tokens | 8K decode tokens/s | 64K decode tokens/s |
| --- | ---: | ---: |
| 0 | 29.67 | 29.55 |
| 1 | 41.24 | 41.16 |
| 2 | 45.73 | 45.51 |

Keep `MTP_TOKENS=2` for sustained output. Disabling MTP reduced cache-hit
first-token latency (8K approximately 0.22–0.27 seconds versus 0.63–0.64 with
MTP2), so short-answer latency can favor a different choice. These synthetic
single-request observations are not concurrency or p95 measurements.
Adding QSA metadata updates alone would not merge the first draft-prefill
and remaining decode graphs for MTP2; no such cache change is included.

Eight NCCL configurations passed collective correctness checks. A 15,360-byte
AllReduce measured 24.56 microseconds with automatic selection (24.70 on
repeat), 24.51 with Ring/LL, 67.50 with Tree/LL, 30.85 with Ring/Simple, and
32.74 with Ring/LL128. Each single-NIC configuration measured approximately
26.4–26.5 microseconds. Keep automatic selection and both RoCE links; these
small-message results do not justify globally forcing another protocol.

## Reproducing GB10 TP2 tuning

Stop the serving processes before running independent GPU microbenchmarks.
`benchmarks/kernels/benchmark_qwen_spark_gemm.py` sweeps the actual TP2 BF16
projection shapes with FP32 reference checks and cold-L2 CUDA-graph CUPTI
measurements. It requires the optional `cupti-python` package compatible with
the installed CUDA runtime; install it in a separate benchmark environment,
since its dependency constraints can change `cuda-bindings`. The benchmark
fails if CUPTI cannot import rather than accepting event-timer fallback.

```bash
.venv/bin/python benchmarks/kernels/benchmark_qwen_spark_gemm.py \
  --output .run/gemm-tuning
```

`--plans selected.json` restricts independent verification to a list of
`{"m": ..., "n": ..., "k": ..., "config": ...}` records; `config` contains
`SkinnyGemmConfig` fields. `--rows` and `--shapes` restrict the sweep. Preserve
both measurement rounds and error records rather than selecting from a
single fastest sample.

The collective benchmark runs once on each node with the same master address
and port, and the appropriate rank (0 or 1). It checks AllReduce/AllGather
correctness, retains two distinct-address CUDA graphs, and records 60 samples
per message size, reporting the maximum of both ranks for each sample.

```bash
MASTER_ADDR=192.168.100.10 \
NCCL_SOCKET_IFNAME=enp1s0f1np1 GLOO_SOCKET_IFNAME=enp1s0f1np1 \
.venv/bin/python benchmarks/kernels/benchmark_spark_collectives.py \
  --rank "$RANK" --output ".run/collectives-rank${RANK}"
```

Set identical `NCCL_ALGO`/`NCCL_PROTO` on both nodes to compare forced choices,
or leave them unset for automatic selection. Capture initialization logs with
`NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=INIT,NET,TUNING`, and a unique
`NCCL_DEBUG_FILE`. A CPU-submitted barrier followed by separately submitted
events can include host scheduling gaps; this benchmark captures the barrier
and timing events in the same graph to keep that gap outside the interval.

AI assistance was used for this branch. The backported implementation and tests
retain their upstream provenance above.
