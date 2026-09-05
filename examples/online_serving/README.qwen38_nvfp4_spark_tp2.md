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

Caches stay inside this worktree. The configuration uses eager execution, 512K context,
2048 batched tokens, four concurrent sequences, and a fixed 16 GiB KV cache per
GPU. The explicit cache budget takes precedence over the memory-utilization
fraction for KV allocation. Spark shares GPU and system memory; leaving the
cache budget implicit allocated about 33 GiB per GPU in the initial test and
left only about 10 GiB available on the head during serving.
Set `MAX_MODEL_LEN`, `MAX_NUM_BATCHED_TOKENS`, `MAX_NUM_SEQS`, and
`KV_CACHE_MEMORY_BYTES` identically on both nodes when tuning. Increase the cache
budget when increasing context length or concurrency, and check available system
memory during startup and serving.

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

In this rc4 implementation, the MTP draft retains its native 262,144-token
limit while the target remains at 524,288. Batches exceeding the drafter's
limit skip speculation. Startup also warns that draft KV cache groups cannot
be identified, disabling cross-request prefix-cache reuse. Fused multi-step
drafting is unavailable for the QSA state backend; vLLM rebuilds attention
metadata between draft steps. These fallbacks do not prevent serving, but
mean this configuration is not a demonstrated throughput improvement.
The MTP run reported 1,153,433 cache tokens and approximately 26 GiB available
system memory on the head during verification. Full-length 512K generation
was validated before MTP was enabled and was not repeated in this trial.

FlashInfer autotuning is disabled in this example. On a repeated two-node
startup, rank 0 hit its cached MoE tactics while rank 1 entered profiling and
waited in `flashinfer.autotuner._profile_single_kernel`'s `all_reduce`. Disabling
this optional tuning avoids that collective mismatch; CUTLASS kernel execution
and the other kernel warmups remain enabled. Performance with heuristic tactics
can differ from a successful autotuned run.

To access the head's loopback API from the local computer, keep an SSH tunnel
open with `ssh -N -L 18029:127.0.0.1:18029 aitopatom-d6d3`, then use
`http://127.0.0.1:18029/v1` with model name `qwen38-nvfp4`.

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

AI assistance was used for this branch. The backported implementation and tests
retain their upstream provenance above.
