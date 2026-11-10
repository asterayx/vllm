#!/bin/bash
# PRODUCTION (Ricardo approved 2026-11-11): Engram file-backed on NVMe by default.
# Identical to deepseek_v4_1_flash_yzss_tp4pp2_e1_prod.sh except this header and
# --engram-config, which now comes from DPSK_ENGRAM_CONFIG:
#   default (file-backed, NVMe): '{"cpu_offload": true, "storage_path": "/data/engram"}'
#   pinned (E1, ~97 GiB locked host memory): DPSK_ENGRAM_CONFIG='{"cpu_offload": true, "use_thp": true}'
# #22 A/B: file-backed +7..15% at C4-C8, 0..-8% at C1 vs pinned. /data/engram needs
# ~97 GiB free; after a restart, warm up before taking traffic (cold page cache
# raises TTFT).
DPSK_ENGRAM_CONFIG=${DPSK_ENGRAM_CONFIG:-'{"cpu_offload": true, "storage_path": "/data/engram"}'}
unset FLASHINFER_AUTOTUNE_DIR DPSK_NVTX DPSK_FP8_SKINNY_LINEAR_MAX_M VLLM_USE_BREAKABLE_CUDAGRAPH NCCL_ALGO NCCL_PROTO
export DPSK_FP8_SKINNY_LINEAR=0
echo 1000 > /proc/self/oom_score_adj
unset DSV41_EXP_INDEXER_LOGITS_SHRINK
unset DSV41_EXP_CED_PREFILL DSV41_EXP_CED_PREFILL_ROWS
# DeepSeek-V4.1-Flash-1017-NVFP4 on yzss, TP=4 x PP=2 (PP stage0=GPU0-3/NUMA0, stage1=GPU4-7/NUMA1)
# Based on ~/vllm/examples/deployment/deepseek_v4_1_flash_rtx_pro_5000.sh (FP8 KV required on sm_120; Engram cpu_offload default)
# no --numa-bind: both Engram layers (1,14) live on PP stage0 => ~200GB host tables > NUMA0 128GB
cd /home/roccen/vllm
export PYTHONPATH=/home/roccen/vllm${PYTHONPATH:+:$PYTHONPATH}
export PATH=/home/roccen/vllm/.venv/bin:$PATH
# RoCE ports (ens2f0np0/ens2f1np1) are link-down; single node -> keep NCCL off the IB/RoCE net plugin
# iommu=pt set 2026-11-03: PCIe P2P verified (cudaMemcpyPeer PASS) -> P2P re-enabled
# 23:52 DeepGEMM JIT needs nvcc; system /usr/local/cuda has no nvcc; venv nvcc 13.4 mismatches runtime headers 13.2 -> merged toolkit ~/cuda132_tk (venv cu13 symlinks + nvcc/crt/nvvm 13.2.86)
export CUDA_HOME=/home/roccen/cuda132_tk PATH=/home/roccen/cuda132_tk/bin:$PATH
export LIBRARY_PATH=/home/roccen/cuda132_tk/lib64:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/home/roccen/cuda132_tk/lib64:${LD_LIBRARY_PATH:-}
export NCCL_IB_DISABLE=1 PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT
# 12:52 try41: DSpark draft FULL_DECODE_ONLY CUDA graph (+21% C1, +18% C16 vs eager draft) - permanent
export DPSK_DRAFT_FULL_CG=1
# 12:52 try42: FlashInfer PCIe IPC custom all-reduce for TP4 (vLLM CustomAllreduce refuses >2 PCIe-only GPUs)
# 13:22 try44: SM120 Triton fp8 paged MQA logits for 128-row ratio-1 indexer pages (replaces fp32 torch ref) - opt-in
export DPSK_INDEXER_SM120_FIX=${DPSK_INDEXER_SM120_FIX:-1}
# 14:40 try49: vocab-shard local greedy argmax (target verify + DSpark draft, tiny (max,idx) all-gather) - rollback: DPSK_VOCAB_LOCAL_ARGMAX=0
export DPSK_VOCAB_LOCAL_ARGMAX=${DPSK_VOCAB_LOCAL_ARGMAX:-1}
# 14:40 try49: lm_head (32320x5120 bf16) cuteDSL skinny GEMM for M in {3,4,6,8,12} instead of cuBLAS cutlass_80 - rollback: DPSK_LMHEAD_SKINNY=0
export DPSK_LMHEAD_SKINNY=${DPSK_LMHEAD_SKINNY:-1}
# 1449 try50 (exp): SM120 Triton fp8 skinny grouped bmm for o_proj wo_a (M<=64; bf16 bmm fallback) - opt-in, rollback: DPSK_WO_A_FP8_SKINNY=0
export DPSK_WO_A_FP8_SKINNY=${DPSK_WO_A_FP8_SKINNY:-1}
export DPSK_FP8_SKINNY_LINEAR=${DPSK_FP8_SKINNY_LINEAR:-0}
# try53: MoE NVFP4 backend marlin (W4A16) KEEP: C1 step ~13.0 vs 14.0ms flashinfer_cutlass, acc_len same
export DPSK_MOE_BACKEND=${DPSK_MOE_BACKEND:-marlin}
# try55: DPSK_PP_MICROBATCH=1 PROMOTED default 1 (parent decision 17:59): C16 warm +9.6% (935 vs 853), consistent, C1 flat. Rollback: DPSK_PP_MICROBATCH=0
# try55 sweep 18:2x (warm, same protocol): C16 ON 941 vs OFF 960, C32 935 vs 928, C8 656 vs 587 (OFF noisy 534-683), C1 flat -> ~neutral; earlier +9.6% was warm-up artifact. Kept 1 (no regression).
export DPSK_PP_MICROBATCH=${DPSK_PP_MICROBATCH:-1}
# try56: max_num_seqs 16 -> 64 (C32 was queueing). Rollback: DPSK_MAX_NUM_SEQS=16
export DPSK_MAX_NUM_SEQS=${DPSK_MAX_NUM_SEQS:-64}
export VLLM_ALLREDUCE_FLASHINFER_PCIE_IPC_MAX_TOKENS=${VLLM_ALLREDUCE_FLASHINFER_PCIE_IPC_MAX_TOKENS:-128}
# 2026-11-03 23:33: host Engram cpu_offload (2 tables x 94.4 GiB fp8 = ~189 GiB pinned on PP0) OOM-killed on this 247 GiB host ; GPU-resident tables OOM on TP4xPP2 (23:34) -> file-backed Engram tables on NVMe /data/engram (patched vLLM storage_path, no pin)
exec /home/roccen/vllm/.venv/bin/python -m vllm.entrypoints.cli.main serve /data/models/nvidia/DeepSeek-V4.1-Flash-1017-NVFP4 \
  --served-model-name dpsk-v4.1-flash \
  --tensor-parallel-size 4 --pipeline-parallel-size 2 \
  --max-model-len 65536 --max-num-seqs ${DPSK_MAX_NUM_SEQS} --gpu-memory-utilization 0.91 \
  --kv-cache-dtype fp8 \
  --tokenizer-mode deepseek_v41 --reasoning-parser deepseek_v41 \
  --enable-auto-tool-choice --tool-call-parser deepseek_v41 \
  --model-loader-extra-config "{\"enable_multithread_load\": false, \"num_threads\": 16}" \
  --compilation-config "{\"cudagraph_mode\": \"PIECEWISE\"}" \
  --engram-config "$DPSK_ENGRAM_CONFIG" \
  --speculative-config "{\"method\": \"dspark\", \"num_speculative_tokens\": ${DPSK_NUM_SPEC:-3}}" \
  --per-request-spec-decode-metrics detailed \
  --host 0.0.0.0 --port 8000 --kernel-config "{\"moe_backend\": \"${DPSK_MOE_BACKEND}\"}" --max-model-len 1048576 "$@"
# 01:36 try19: PP-safe flashinfer autotune group (TP group per PP stage) — try18 deadlocked (PP0 world all_reduce vs PP1 recv)
# 01:41 try20: flashinfer prefill dual PBSX=128 instantiation (ratio-1 layers at block 128)
# 02:01 try21 (debug): try20 ready but NaN logits on prefill -> eager + DPSK_NAN_DBG layer probe
# try22 (debug): intra-layer NaN probes (layer0: NaN appears in layer 0 on PP0)
# try23 (debug): attention-internal probes for layer 0
# try24 (debug): layer0 SWA sparse-MLA in/out dump (DPSK_DUMP_DIR)
# try25 (debug): null-block row-0 scrub before SM120 sparse-MLA kernels (fix layer0 NaN: padding idx -1 clamped to slot 0 holding NaN bytes)
# try26 (prod): scrub fix, no --enforce-eager, no DPSK_NAN_DBG/DPSK_DUMP_DIR
# try27 (prod): MTP dspark num_speculative_tokens=3 + torch profiler (/start_profile,/stop_profile -> profiles/try27_mtp) + per-request spec metrics summary
# try28 (prod): try27 failed (DeepseekV41 dspark w/ PP unsupported) -> model.py: supports_aux_hidden_states_over_pp + PP aux relay + embed_tokens on last PP rank
# try29 (debug): DPSK_SPEC_DBG=1 model_runner SPECDBG log (try28 up but dspark acceptance 0%)
# 12:05 try37 (prof): reprofile no with_stack, record_shapes, warmup5/active10 -> profiles/opt_reprofile_20261104
# 12:16 try38 (prof): try37 worker heap crash (free(): invalid pointer) after B trace export -> record_shapes=false, dump_cuda_time_total=false
# 12:30 try39 (prof, same cfg): try38 died 12:26 on 3rd profiler session - libkineto ~MemoryTraceLogger heap corruption (free(): invalid pointer) when old ProfilerResult GCed during new session; rule: 1 profile session per serve lifetime
# 12:40 try40 (prod, profiler OFF; prof line saved in serve_tp4pp2.sh.bak_try39_prof): try39 died 12:35 during E 58K prefill, 2nd profiler session of lifetime - same kineto free(): invalid pointer in VllmWorker-1 (PP0_TP1), not OOM. Base for step1 draft-CG / custom AR experiments
# 12:45 try41 (exp, profiler OFF): DPSK_DRAFT_FULL_CG=1 env (draft FULL_DECODE_ONLY, main PIECEWISE); try40 baseline C1 94 tok/s, C16 489 tok/s agg
# 12:52 try42 (exp, profiler OFF): DPSK_DRAFT_FULL_CG=1 persisted + VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1 MAX_TOKENS=128 (step2 custom AR)
# 13:05 try43 (ctrl): same as try42 but VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=0 via env override - warm A/B control
# 13:12 try43 warm ctrl == try42 AR (C1~140, C16~700): PCIe IPC AR neutral -> default 0, live serve=try43
# 13:22 try44 (prod): DPSK_INDEXER_SM120_FIX=1 (Triton paged logits, sparse_attn_indexer.py + layers/sm120_paged_mqa_logits.py); try43 base 60K C8 365-378 tok/s TPOT ~20ms
# 13:37 try44 result: 60K C8 365-378 -> 577-642 tok/s (TPOT 20 -> 11-12 ms), 60K C1 157 -> 171-178, short C16 711 -> 765-838; acc_len unchanged; unit test vs torch ref relmax 3e-7, top512 identical -> script default DPSK_INDEXER_SM120_FIX=1 (code default 0)
# 13:58 try45 (exp): DPSK_VOCAB_LOCAL_ARGMAX=1 (+CHECK=20: 32 checks, mismatches=0); short C1 step ~14.75 ms vs ctrl 15.1; C16 806-884 vs ctrl 753-856; 3K C1 ~neutral
# 14:08 try46 (exp): + DPSK_LMHEAD_SKINNY=1 (1 lm_head on PP1, shared target/draft); microbench L2-rotated 282->256us M<=8; in-situ 285->270us
# 14:15 try47 (ctrl): try44 config, same protocol as try45/46 (warm A/B control)
# 14:29 try48 (prof, 1 session): A C1 trace profiles/try48_vocabargmax: full-vocab AllGather (183us x1.5/step) -> 4x ~8us (max,idx); lm_head skinny 270us; fp32 simt_sgemm gone on PP1
# 14:40 try49 (prod): defaults DPSK_VOCAB_LOCAL_ARGMAX=1 DPSK_LMHEAD_SKINNY=1 on top of try44; rollback to try44 = DPSK_VOCAB_LOCAL_ARGMAX=0 DPSK_LMHEAD_SKINNY=0
# 1449 try50 (exp): DPSK_WO_A_FP8_SKINNY=1 on top of try49 (wo_a fp8 e4m3+e8m0 repack, Triton skinny bmm; microbench M<=16 23.6->7.0us, M32 ~14us, M64 ~18.5us vs cuBLAS cutlass_80 bf16)
# 1503 try50 result: KEEP (script default DPSK_WO_A_FP8_SKINNY=1, code default 0). short C1 step med 14.4->13.8ms (warm 8 runs 13.4-14.6 vs ctrl 14.3-14.6), C16 agg 797-889 vs ctrl 807-862 (neutral+), 17*23=391, 15K C1 195-203 tok/s short/bad=0, KV cache unchanged 3.55M tok
# 16:00 try51 REVERTED: DPSK_FP8_SKINNY_LINEAR default 0 (micro 3-6x but E2E flat; see bench_try51_fp8lin.txt). live = try50 config (try51_revert log)
# 16:25 try52: 1x C1 profile (profiles/try52_c1prof, summary.txt) then clean restart -> try52_prod_try50 (try50 config, profiler off)
