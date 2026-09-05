# Spark Vision fixes: validation and remaining work

Base: `cursor/spark-obs-codex-df88` at `a9e1dda29`.
Upstream base: v0.28.0 (`2cf0a6915`).
AI assistance: Codex.

## Changes

- Connect image placeholder expansion to v0.28.0's `_apply_prompt_updates`
  hook and pass the tokenizer to planning and embedding-mask callbacks.
- Honor the existing sentinel-span and sliding-window configuration in
  model runner V2, matching the behavior already implemented in V1.
  Image spans include IMAGE_START through IMAGE_END and exclude leading
  compressor padding. Newlines do not split an image into separate spans.
- Keep DSpark graph descriptors at the logical query width. FlashInfer's
  internal padding does not change the graph or sampling layout. Vision
  k=3 can match the existing safe total-token sizes. Text k=5 has no
  compatible FULL graph in the existing SM12x safe-size set and remains
  eager; do not remove the safety filter to force a match.
- Use four-dimensional SDPA inputs for the vision tower and cache RoPE
  tensors per device. Preserve CPU-generated RoPE values.
- Stream language-model weights while buffering only tower weights;
  keep the language-model load/finalization contiguous and single-pass.
- Buffer SSE bytes until a complete line arrives before decoding UTF-8.
  Chinese and emoji characters may cross arbitrary HTTP chunk boundaries.
- Allow an explicit reference checkout for Vision parity tests. Missing
  references skip only dependent tests rather than failing collection or
  suppressing independent tests.

## CPU and proxy verification

Python 3.12 environment managed by uv; tests use `.venv/bin/python`.
The CUDA precompiled editable installation is unavailable on this macOS
arm64 host. CPU tests use the source checkout and common dependencies.

The reference files are `inference/vision.py` and
`inference/image_processor.py` from
[DeepSeek-V4-Flash-Vision-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp/tree/6821d6ad3681a4b137b066b76094fa82ebd0a380/inference),
revision `6821d6ad3681a4b137b066b76094fa82ebd0a380`.
Place those files in the directory passed below; tests do not download or
execute an unpinned remote reference automatically.

```bash
HF_HUB_OFFLINE=1 VLLM_TEST_DSV4_REFERENCE_DIR=/path/to/reference \
  .venv/bin/python -m pytest -q \
  tests/models/test_deepseek_v4_vl_vision.py \
  tests/models/test_deepseek_v4_vl_preprocess.py \
  tests/models/test_deepseek_v4_vl_weights.py \
  tests/v1/cudagraph/test_cudagraph_manager.py \
  tests/v1/attention/test_deepseek_v4_swa_visible.py \
  tests/utils/test_sm12x.py

cargo test --locked --manifest-path docker/gb10/compat-proxy/Cargo.toml
```

Results on 2026-09-05:

- Main Python suite above: **749 passed, 17 skipped** (CUDA-only tests).
- `tests/models/test_deepseek_v4_vl_input_ids.py` and
  `tests/config/test_speculative_draft_hf_overrides.py`: **10 passed**.
- `tests/config/test_model_arch_config.py -k deepseek_v4`: **2 passed**.
- Proxy tests: **8 passed**, compiled with the repository root's Rust 1.95
  toolchain. The proxy's standalone Rust 1.88 toolchain was not verified.
- All applicable pre-commit hooks passed on the committed code, including
  formatting, type checks, and SPDX checks.
- An additional unfiltered architecture-config sweep had 35 passes and
  23 failures because unrelated Hugging Face model configs were unavailable
  in offline mode. These failures are not counted as successful validation.

## Required Spark validation

No GPU performance or full-model quality results are claimed by these
commits. Before using them as a serving release:

1. Run the same tests on SM121, including the CUDA visibility tests skipped
   on the development host. Verify loading the real checkpoint on both ranks.
2. Compare eager and graph execution with DSpark k=3 at concurrency 1–6.
   Check actual graph replay, request padding, draft sampling, and output
   equivalence; successful capture alone does not establish correctness.
3. Compare image preprocessing, image spans, and full-model logits with
   the pinned reference. Include multiple images, non-square images,
   prefix-cache hits, chunk boundaries, and mixed text/image batches.
4. Run model evaluations and record the model revision, quantization,
   dependencies, seed, and commands. CPU tower parity is not a substitute.
5. Measure TTFT, TPOT, accepted tokens per draft round, draft/target/NCCL
   time, peak memory during loading, and throughput. Sweep concurrency
   1/2/4/6, prefill budgets 2048/4096/8192, and short/long contexts.

## Remaining implementation work

- **Compressed attention correctness:** existing SM12x paths still drop
  compressed KV for image-wide prefill and padded short prefill. A correct
  dual-cache kernel or numerically correct fallback is required. Do not
  call this quality-preserving without full-model comparisons, especially
  for long-context cached suffixes and mixed batches.
- **Batched FlashInfer:** validate batched target q=4 and draft padded q=6
  before replacing the per-request launch path.
- **Small-M kernels:** reduce mHC/MoE padding and index-repair overhead only
  after reproducing the historical illegal-memory-access cases on SM121.
- **Tuning:** profile the FP8 output projection, Humming/Triton selection,
  multi-image encoding, and communication before changing kernel defaults.
- **Deployment reproducibility:** bake and verify the selected Humming
  package and record resolved per-layer backends before changing startup
  fallback behavior.
