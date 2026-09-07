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

## Second optimization pass (2026-09-05)

- Consecutive images with equal ViT and LLM grids now share encoder calls,
  capped at two images by default. `--hf-overrides
  '{"vision_encoder_batch_size":1}'` restores single-image calls; merge this
  field into any existing HF overrides. `VLLM_BATCH_INVARIANT=1` forces one.
  Batching increases temporary activation memory; validate peak memory and
  BF16 output tolerance on Spark before choosing a larger cap.
- Spatial merging uses reshape/permutation instead of `unfold`, retaining
  the reference channel and patch ordering, including padded image edges.
- Padded sparse attention allocates output storage without copying old
  output, avoids a redundant input-padding copy, and removes one reduction
  from negative-index repair. Existing sparse kernel selection is unchanged.
- The proxy scans only newly received bytes and drains complete lines once
  per chunk. JSON without missing reasoning aliases avoids serialization.
- Docker build failure handling now limits the ignored failure to the
  optional FlashInfer cache uninstall. `MAX_NUM_BATCHED_TOKENS` exposes the
  prefill budget in `run.sh`, with its existing default of 8192.

Validation: the main Python command above plus
`tests/models/test_deepseek_v4_vl_input_ids.py`,
`tests/v1/attention/test_flashinfer_sparse_mla_sm120_api.py`, and
`tests/config/test_speculative_draft_hf_overrides.py` completed with
**784 passed, 17 CUDA-only skips**. A subsequent vision-suite run after
strengthening the batch-cap assertions passed all **13 tests**. Proxy tests
passed **10 tests**. Shell syntax was checked with `bash -n`.
The Docker image was not rebuilt, and no full-model evaluation or Spark
performance measurement was possible on this host.

## Third pass: review fixes (2026-09-06)

Branch `claude/spark-vision-fixes-r2` on top of `codex/spark-vision-fixes`
(`04bd89aae`). AI assistance: Claude Code. CPU-only host, no GPU; every
kernel-path change below is gated or covered by unit tests, and the
knobs table in `README.md` lists what still needs GPU confirmation.

Correctness:

- **Image prefill keeps compressed attention for text rows.** A >64-token
  prefill chunk with an image now launches the 128-wide dual-cache (C4A)
  cubin for every row and re-launches only the in-image rows SWA-only on
  the 512-wide single-cache cubin. Before, the whole chunk (every request
  in it) was SWA-only. In-image rows still lose C4A (no dual-cache cubin
  accepts the widened window); short (`<=64`, unaligned) prefill spans are
  unchanged. `VLLM_SM12X_SPLIT_IMAGE_PREFILL=0` restores the old path.
- **Image spans never read past the written KV.** The SWA index kernel
  clamps the in-image right window to `seq_len`, and `run-vision.sh`
  passes `--disable-chunked-mm-input` so an image is prefilled in one
  chunk (prefix-cache hits inside a span are covered by the clamp).
- **Chat images stay where the client put them.** The renderer parses
  messages with `content_format="openai"`; the tokenizer inlines
  `<｜deepseek_image｜>` per part (no leading placeholders, no `\n` joins).
  Before, every image was moved to the front of the user turn.
- **Hash-layer image routing** (image sentinel rows use
  `topk(sqrtsoftplus + bias_vl)` instead of the hash table) is now locked
  by a CPU test of the fallback path; the reference `inference/model.py`
  was not reachable from this host, so the semantics remain a structural
  inference (the checkpoint ships `bias_vl` on hash layers).
  `check-extensions.py` now fails the image build when `_moe_C` lacks the
  `bias_vl` argument (otherwise every image row is re-routed in torch on
  every layer).
- **Draft padded q=3->6 with C4A**, batched `[B,4]` decode and q=3->4 are
  still unvalidated. `docker/gb10/validate-knobs.py` compares greedy
  outputs and speed between a baseline server and one started with a knob.

Performance:

- FlashInfer sentinel repair runs once per launch (the wrapper repeated it).
- `input_ids` and the padding mask are padded once per forward; MoE layers
  slice. `b12x` MoE plans above 256 tokens are bucketed.
- Vision: sentinel vectors are folded into the embedding rows after load
  (no per-step table/where); images with different grids in one encoder
  call are packed with a block-diagonal mask; RoPE cache holds 64 grids;
  the SDPA backends available on the device are logged once;
  `VLLM_DSV4_VISION_COMPILE=1` compiles the tower blocks.
- `wo_a` UE8M0 scales are upcast once at load for Humming/Marlin.
- Aux streams (attention, shared experts), batched decode, q=3->4, extra
  DSpark capture tokens are exposed as env knobs (defaults unchanged).
- Startup runs one 128-token prefill on SM12x so the >64-token sparse
  prefill path is compiled before the first real prompt.
- DSpark logs a warning when it ends up with no graphs (text k=5).

CPU validation on this host (uv venv, PyPI torch 2.14 without a driver):
the SM12x, FlashInfer API, vision, weights, tokenizer, b12x, warmup and
router suites pass except three pre-existing CUDA-only failures
(`test_cudagraph_manager::test_full_capture_sets_graph_pool_id_before_cuda_graph`,
`test_b12x::test_b12x_moe_config_support[mxfp4-w4a8-relu2]`,
`test_b12x::test_b12x_moe_reload_reprepares_current_parameters`), which
fail identically on the parent commit.

Spark procedure for each knob:

```bash
NODE_RANK=0 ./docker/gb10/run-vision-image.sh          # baseline
./docker/gb10/validate-knobs.py run --out base.json --images ~/val-images
docker rm -f dspark-vision-tp2-rank0 dspark-vision-tp2-rank1
# add -e VLLM_SM12X_BATCHED_DECODE_NEXT_N=4 (etc.) to the docker run
./docker/gb10/validate-knobs.py run --out knob.json --images ~/val-images
./docker/gb10/validate-knobs.py compare base.json knob.json
```

### Spark results (2026-09-06, Vision-Exp, 2x GB10, k=3)

- Default configuration boots with the third-pass changes: 128-token
  long-prefill warmup, `image prefill keeps C4A` split, sentinel fold and
  SDPA probe (`flash=ok, mem_efficient=ok`) all ran. `_moe_C` has the
  12-argument `topk_softplus_sqrt` (bias_vl kernel path).
- `VLLM_SM12X_BATCHED_DECODE_NEXT_N=4`: 3 of 4 prompts token-identical to
  the per-request baseline (145- and 2652-token prompts included). The
  fourth prompt was a flat-distribution prompt whose first token changed on
  every restart in both configurations (restart-level noise; replaced in
  the script). Single stream 0.95-1.0x, concurrency 4: 88.3 -> 98.1 tok/s
  aggregate (1.11x). Now the `run-vision.sh` default.
- `docker/gb10/profile-decode.sh` + `summarize-profile.py` give the GPU
  time split of a decode step (needs `VLLM_PROFILE_DIR` at start). Use it
  before changing kernel defaults further.
- **Not worth doing: a split-K Triton GEMM for the M<=32 bf16 projections**
  (gate, compressor `fused_wkv_wgate`, indexer `weights_proj`). Measured on
  GB10 with cold L2 (256 MB of rotating weight copies, CUDA graph replay,
  K=4096): cuBLAS reads at 154-214 GB/s (N=256: 12.6 us, N=1024: 45 us,
  73-78% of the 273 GB/s peak); the Triton kernel was 15% faster only at
  N=1024 and 30% slower at N=256. The hot-L2 numbers (6-20 us) are not
  representative of the model, where each weight is read once per step.
  The 50 us cuBLAS kernel in the decode profile is the 8 MB compressor
  projection at the bandwidth floor; the 114 us one reads ~21 MB, so it is
  a bf16 matrix that is not the gate or the compressor. Cutting that share
  means reading fewer bytes (FP8 weights), not a different kernel. The
  commit was reverted; see the git history for the kernel.
- **MoE all-reduce on real rows** (`VLLM_SM12X_REDUCE_REAL_ROWS`, default
  on). `bench-allreduce.sh` on the two Sparks: a 32 KB all-reduce
  (`[4, 4096]` bf16, the real rows of a single-stream DSpark k=3 step) is
  22 us wall, the 128 KB one (`[16, 4096]`, the SM12x padded MoE block) is
  214 us (512 KB is 101 us, so 128 KB also sits in a bad NCCL protocol
  band). The fused MoE now runs with `reduce_results=False` and
  `DeepseekV4MoE._forward_fused_moe` all-reduces `[:orig_tokens]` after
  the padded block; the attention `wo_b` all-reduce was already on real
  rows. Expected: most of the 43 MoE all-reduces per step drop from
  ~100 us to the 32 KB cost. Batches of >=16 real tokens are unchanged.
  Only taken when FusedMoE really skipped its reduce
  (`moe_config.skip_final_all_reduce`), so EP/all2all configurations keep
  the old path. **Validated on Spark 2026-09-07** (`validate-knobs.py`,
  `NCCL_PROTO=` on both runs): prompts identical or diverging only at
  0.000-nat ties and the restart-level thinking prefix of prompt 0; single
  stream +5% / +10% / +16% on the three comparable prompts; concurrency 4
  unchanged (88.1 tok/s, 16 real rows have no padding).
- **`NCCL_PROTO=Simple`** (now the `run.sh` default). `bench-allreduce.sh`
  kernel times on the two Sparks: NCCL's own choice is 98 us at 128 KB,
  112 us at 512 KB and 321 us at 2 MB; Simple is 38 / 74 / 117 us. Only
  32 KB prefers the default (23 vs 31 us). Per decode step (86 all-reduces)
  that is +0.7 ms single stream after real-rows, -5 ms at concurrency 4,
  and -17 ms per 256-token prefill chunk. **Validated on Spark
  2026-09-07** on top of real-rows: single stream +2% / -2% on the two
  comparable prompts, concurrency 4 88.1 -> 100.8 tok/s aggregate (1.14x);
  divergences at 0.125-0.25 nats are the same restart-level noise seen
  between two runs of one configuration. `NCCL_IB_QPS_PER_CONNECTION`,
  `NCCL_IB_SPLIT_DATA_ON_QPS`, `NCCL_NET_GDR_LEVEL` and
  `NCCL_MIN_NCHANNELS` were within noise on this point-to-point link.
- **Stay off** (Spark 2026-09-07, `validate-knobs.py --concurrency 4`
  against the new defaults; no crashes, outputs identical up to restart
  noise): `VLLM_SM12X_DECODE_Q_ALIGN_ALLOW_4=1` single stream 0.99x,
  concurrency 1.01x (decode attention is latency-bound at [1,4] and [1,6]
  alike); `VLLM_SM12X_SHARED_EXPERTS_STREAM=1` single stream 1.00x,
  concurrency 0.97x (both expert paths are bandwidth-bound, the second
  stream only adds event synchronization); `VLLM_SM12X_ATTN_AUX_STREAMS=1`
  single stream 1.00x, concurrency 0.86x (the three side projections are
  too small to hide anything and the fan-out/join events stall the main
  stream on batched steps).
- **Draft acceptance is unchanged by the new defaults.** A real coding
  session on the new defaults showed drafted throughput 41-42.6 tok/s
  (37-39 before: the step got ~10% faster) but only 37-50% draft
  acceptance against 88% on an earlier session. `validate-knobs.py` now
  records per-prompt acceptance from the `vllm:spec_decode_*` counters:
  the two content-identical prompts accept 78.6% / 89.5% on the new
  defaults, matching the 76.6% aggregate with real-rows and batched decode
  off. Low acceptance in a session is the content (and any client-side
  temperature), not the serving path; the restart-noisy prompt 0 is
  excluded from that comparison for the same reason.
- **Server-side sampling defaults** (`SAMPLING_DEFAULTS`, `run-vision.sh`
  sets temperature 0.2 / top_p 0.95). The checkpoint's
  `generation_config.json` is `_from_model_config: true` with temperature
  1.0 and top_p 1.0, so a client that sends no sampling parameters (Grok
  Build via the compat proxy does not) samples the full distribution; that
  is the most likely cause of the 37-50% acceptance in the coding session
  above. To validate: rerun the same session and compare `Draft
  acceptance rate` and `Avg generation throughput`; `SAMPLING_DEFAULTS=`
  (empty) restores the checkpoint values.
- Measurement note: single-stream tok/s of the 2652-token prompt includes
  its prefill, so a second `validate-knobs.py run` against the same server
  hits the prefix cache and reads ~70% faster. Compare single-stream
  numbers only between first runs after a restart; the concurrency
  aggregate is warmed by the single-stream phase in every run and stays
  comparable.
- Still to validate the same way: an image prompt set (`--images`) for
  the split-prefill path.

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
