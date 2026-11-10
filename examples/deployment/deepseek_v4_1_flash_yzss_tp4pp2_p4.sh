#!/bin/bash
# DeepSeek-V4.1-Flash-1017-NVFP4 on yzss, TP=4 x PP=2: production config
# (deepseek_v4_1_flash_yzss_keep_tp4pp2.sh on cursor/yzss-keep-serve-tp4pp2-d6e3)
# plus the three changes measured in asterayx/vllm#21 and #22 ("P4"):
#   1. Engram tables pinned in host memory with exact-size THP registration
#      instead of NVMe storage_path (#21 E1). Needs ~97 GiB locked host memory
#      and ~2 min longer start-up; /sys/kernel/mm/transparent_hugepage/enabled
#      must be always or madvise.
#   2. DPSK_FP8_SKINNY_LINEAR=1: Triton FP8 skinny GEMM for small-M MXFP8
#      linears, split-K reduced in-kernel (needs #22 code).
#   3. FLASHINFER_AUTOTUNE_DIR -> decode-first sparse-MLA calibration, so target
#      decode uses sparse_mla_decode_dsv4 instead of prefill_mg_dual (#22 P3).
#      Generated once from FlashInfer's own cache by fi_sparse_mla_decode_first.py.
# Rollback per item: DPSK_ENGRAM_CONFIG='{"cpu_offload": true, "storage_path":
# "/data/engram"}', DPSK_FP8_SKINNY_LINEAR=0, DPSK_FI_AUTOTUNE_DIR="" (unset).
set -euo pipefail
VLLM_DIR=${VLLM_DIR:-/home/roccen/vllm}
cd "$VLLM_DIR"
export PYTHONPATH=$VLLM_DIR${PYTHONPATH:+:$PYTHONPATH}
export PATH=$VLLM_DIR/.venv/bin:$PATH
export CUDA_HOME=/home/roccen/cuda132_tk PATH=/home/roccen/cuda132_tk/bin:$PATH
export LIBRARY_PATH=/home/roccen/cuda132_tk/lib64:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/home/roccen/cuda132_tk/lib64:${LD_LIBRARY_PATH:-}
export NCCL_IB_DISABLE=1 PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT

# Unchanged production flags.
export DPSK_DRAFT_FULL_CG=1
export VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=${VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC:-0}
export VLLM_ALLREDUCE_FLASHINFER_PCIE_IPC_MAX_TOKENS=${VLLM_ALLREDUCE_FLASHINFER_PCIE_IPC_MAX_TOKENS:-128}
export DPSK_INDEXER_SM120_FIX=${DPSK_INDEXER_SM120_FIX:-1}
export DPSK_VOCAB_LOCAL_ARGMAX=${DPSK_VOCAB_LOCAL_ARGMAX:-1}
export DPSK_LMHEAD_SKINNY=${DPSK_LMHEAD_SKINNY:-1}
export DPSK_WO_A_FP8_SKINNY=${DPSK_WO_A_FP8_SKINNY:-1}
export DPSK_MOE_BACKEND=${DPSK_MOE_BACKEND:-marlin}
export DPSK_PP_MICROBATCH=${DPSK_PP_MICROBATCH:-1}
export DPSK_MAX_NUM_SEQS=${DPSK_MAX_NUM_SEQS:-64}

# P4 changes.
export DPSK_FP8_SKINNY_LINEAR=${DPSK_FP8_SKINNY_LINEAR:-1}
ENGRAM_CONFIG=${DPSK_ENGRAM_CONFIG:-'{"cpu_offload": true, "use_thp": true}'}
FI_DIR=${DPSK_FI_AUTOTUNE_DIR-$HOME/dpsk_deploy/fi_decode_first}
if [ -n "$FI_DIR" ]; then
  if [ ! -f "$FI_DIR/sparse_mla_sm120_cpb.json" ]; then
    python examples/deployment/fi_sparse_mla_decode_first.py --out-dir "$FI_DIR"
  fi
  export FLASHINFER_AUTOTUNE_DIR=$FI_DIR
fi

exec python -m vllm.entrypoints.cli.main serve /data/models/nvidia/DeepSeek-V4.1-Flash-1017-NVFP4 \
  --served-model-name dpsk-v4.1-flash \
  --tensor-parallel-size 4 --pipeline-parallel-size 2 \
  --max-model-len 65536 --max-num-seqs "${DPSK_MAX_NUM_SEQS}" --gpu-memory-utilization 0.93 \
  --kv-cache-dtype fp8 \
  --tokenizer-mode deepseek_v41 --reasoning-parser deepseek_v41 \
  --enable-auto-tool-choice --tool-call-parser deepseek_v41 \
  --language-model-only \
  --model-loader-extra-config '{"enable_multithread_load": false, "num_threads": 16}' \
  --compilation-config '{"cudagraph_mode": "PIECEWISE"}' \
  --engram-config "$ENGRAM_CONFIG" \
  --speculative-config "{\"method\": \"dspark\", \"num_speculative_tokens\": ${DPSK_NUM_SPEC:-3}}" \
  --per-request-spec-decode-metrics summary \
  --host 0.0.0.0 --port 8000 --kernel-config "{\"moe_backend\": \"${DPSK_MOE_BACKEND}\"}" "$@"
