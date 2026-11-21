# RTX PRO 5000 integration with vLLM v0.34.0

This branch merges the official stable release into the remote RTX PRO 5000
branch. It is an integration candidate, not a replacement deployment.

## Sources and conflict resolution

- Fork parent: asterayx/vllm `rtxpro5000`,
  `644379a54d4ae3cff303cc6ded84b71d7fc37a41`.
- Release parent: vllm-project/vllm `v0.34.0`,
  `62d4bc3c7e33b1b6c906e233e6af2d3506c482ca`.
- The fork's pre-customization base is `db9527a46`. Its 26 backported upstream
  commits are all ancestors of the release. Those backports use the release
  implementation; the custom delta after that base was merged and ported.
  Both original parents are retained in the merge commit.

The release removes Model Runner V1. The integration keeps Model Runner V2
and the release KV-cache planner. The latter already iterates only each worker's
local group layers, covering the custom PP allocation fix.

File-backed Engram storage and packed MXFP4 tables now live in
`vllm/models/deepseek_v41/common/engram.py`, following upstream's unification
of the NVIDIA and common implementations. Regular FP8 tables retain upstream
sorted lookup, peer tables and packed staging. File-backed/MXFP4 tables use
compatible BF16 staging. File mappings remain private, unlinked and unpinned.

SM120 GEMM plans, NVFP4 decode MoE, local-vocabulary argmax, PP metadata caching,
NVTX ranges, PCIe IPC token limits and deployment scripts are retained.
Upstream's capability map now also includes the fork's SM120 plans.

General FlashInfer warmup uses upstream's per-rank cache handling.
Legacy sparse MLA warmup tunes all TP peers together, separates PP stages,
clears inconsistent in-memory state and saves each rank's file.
The managed autotune-v2 path remains upstream's implementation.

Draft snapshots and per-request persistence are retained. Grammar trimming
uses the release contract: retain the first invalid draft for rejection,
rather than copying its grammar mask onto unreachable later rows.
Regression expectations follow that contract and the current bitmask capacity.
The GLM parser still ignores thinking switches absent from the actual template.

## Validation on xzss

The source and environment are isolated in
`/home/roccen/vllm-integration-v0.34.0`.
The existing `/home/roccen/vllm` checkout and serving environment were not updated.

Python 3.12, the official v0.34.0 precompiled extensions, PyTorch 2.14.0 and
FlashInfer 0.7.3 were installed in the integration checkout's own `.venv`.
No C/CUDA source differs from the release, so its precompiled extensions apply.

Relevant test groups, including focused reruns after fixes:

| Coverage | Result |
| --- | --- |
| RTX fixes, draft snapshots, NVTX, autotune, NVFP4 padding, selected Engram tests | 159 passed, 5 existing skips |
| Scheduler and async scheduler: draft/persistence/grammar cases | 31 passed |
| Structured output, GLM parsing, tool serialization, PCIe IPC dispatch | 117 passed, 1 existing xfail |
| SM120 NVFP4 MoE, intermediate size 160, token counts 1/8/32/96 | 4 passed |

The new Engram regression covers GPU, pinned host and file-backed storage with
both FP8 and MXFP4, checking exact output equality. Existing graph-break and
prefetch-order tests pass. The SM120 MoE cases run with
`QWEN4_EXP_TRITON_MOE=1`.

For model-level smoke validation, Qwen/Qwen3-0.6B ran on one free GPU with greedy
decoding, seed 0, eager execution, identical prompts and identical settings.
The official release checkout and integration checkout generated exactly the
same 72 token IDs across three prompts. This is a framework correctness smoke,
not Qwen3.8/DeepSeek full-model accuracy or performance acceptance.

Static validation uses the repository pre-commit hooks on the integration
delta relative to v0.34.0, including Ruff, mypy, shell scripts, imports and CI
test wiring. No new skips, xfails or relaxed numerical tolerances were added.

## Reproduction

Use the integration checkout's `.venv/bin/python` and a free GPU. The tested
environment sets `CUDA_HOME=/usr/local/cuda-13.5` and
`FLASHINFER_NVCC=/usr/local/cuda-13.5/bin/nvcc`.

The exact commands, logs and the small-model comparison script/results are
retained on xzss under `/home/roccen/vllm-v034-validation`.
Large-model serving, multi-GPU throughput, and all opt-in optimization
combinations still require dedicated acceptance testing before deployment.

AI assistance: Codex performed the merge, conflict migration and validation.
