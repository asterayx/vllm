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
kernel compilation. It defaults to eight compilation jobs; override `MAX_JOBS`
to adjust this limit.

## Launch

Run rank 1 on the worker, then rank 0 on the head. Set the interconnect IPs and
interface names for your machines. The following addresses match the test setup:

```bash
# Worker: aitopatom-da17
MODEL_PATH=/path/to/local/checkpoint \
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
`MASTER_PORT`, `API_HOST`, or `API_PORT` as needed. Caches stay inside this
worktree. The initial configuration uses eager execution, 16K context,
2048 batched tokens, four concurrent sequences, and 80% memory utilization.
Set `MAX_MODEL_LEN`, `MAX_NUM_BATCHED_TOKENS`, `MAX_NUM_SEQS`, and
`GPU_MEMORY_UTILIZATION` identically on both nodes when tuning.

## Verification

```bash
.venv/bin/python -m pytest tests/models/qwen4_exp/test_ple.py -v
pre-commit run --files vllm/models/qwen4_exp/nvidia/ple_layer.py \
  tests/models/qwen4_exp/test_ple.py \
  examples/online_serving/qwen38_nvfp4_spark_tp2.sh
curl --fail http://127.0.0.1:18029/health
curl --fail http://127.0.0.1:18029/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-nvfp4","messages":[{"role":"user","content":"What is 2 + 2?"}],"max_tokens":256,"temperature":0}'
```

Local validation: 18 PLE tests passed, including both TP2 ranks; all relevant
pre-commit hooks passed. On the head GB10, the combined PLE and model config
suites passed all 25 tests. Both TP ranks loaded the complete checkpoint and
reported approximately 61.73 GiB of model memory per GPU. Serving initialization,
generation, and model evaluation are still pending; weight loading alone does
not establish Spark serving support.

The broader local `test_config.py` run passed six tests but could not import the
MTP model for one test because the macOS environment lacks `torchvision`.

On the two test machines, the isolated worktree is
`/home/roccen/src/vllm-qwen38-nvfp4-spark-tp2`. The original
`/home/roccen/src/vllm` checkout is preserved. Rank 0 uses
`/home/roccen/models/Qwen3.8-Flash-Next-NVFP4`; its complete checkpoint was copied
over the interconnect to rank 1 at
`/home/roccen/models/Qwen3.8-Flash-Next-NVFP4-codex-rc4`.

Both machines have the rc4 precompiled extensions and can import CUDA kernels.
Worker dependencies were copied from the isolated head environment, followed by
a completed editable install; matching Python headers were copied into its
`.venv/include/` for Triton. The old worker container
`qwen38-nvfp4-tp2-rank1` was stopped with user authorization.

AI assistance was used for this branch. The backported implementation and tests
retain their upstream provenance above.
