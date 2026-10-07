#!/usr/bin/env bash
# Serve Qwen3.8-Flash-Next on RTX PRO 5000 Blackwell (sm_120, 48/72 GB) GPUs.
#
# NVFP4 checkpoint footprint (512 experts x 48 layers):
#   GPU:  ~68 GB NVFP4 routed experts + ~10 GB bf16 dense/embeddings
#         + ~2.5 GB FP8 MTP experts + ~1 GB vision  => ~80 GB total
#   Host: ~51 GB FP8 n-gram (PLE) table in pinned memory, sharded across TP
# 48 GB cards: TP=2 is tight, prefer TP=4. 72 GB cards: TP=2 fits easily;
# TP=4 leaves the most KV cache.
#
# Usage:
#   MODEL=~/models/nvidia/Qwen3.8-Flash-Next-NVFP4 TP=2 ./qwen3_8_flash_next_rtx_pro_5000.sh
#   CHECK=1 ...   # run the sm_120-relevant kernel tests first
#   TEXT_ONLY=1   # skip the vision tower (--language-model-only)
#   EP=1          # shard experts instead of their intermediate dim
#   SPEC=2        # MTP speculative decoding with 2 draft tokens
set -euo pipefail

MODEL=${MODEL:-nvidia/Qwen3.8-Flash-Next-NVFP4}
TP=${TP:-2}
MAX_LEN=${MAX_LEN:-32768}
MAX_SEQS=${MAX_SEQS:-16}
GPU_UTIL=${GPU_UTIL:-0.95}
PORT=${PORT:-8000}
PYTHON=${PYTHON:-.venv/bin/python}

nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv

if [[ "${CHECK:-0}" == "1" ]]; then
    "$PYTHON" -m pytest -q \
        tests/models/qwen4_exp/test_hc_ops.py \
        tests/models/qwen4_exp/test_qsa_reference.py
fi

extra=()
if [[ "${TEXT_ONLY:-0}" == "1" ]]; then
    extra+=(--language-model-only)
fi
if [[ "${EP:-0}" == "1" ]]; then
    extra+=(--enable-expert-parallel)
fi
if [[ "${SPEC:-0}" != "0" ]]; then
    extra+=(--speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": $SPEC}")
fi

# Without NVLink, P2P over PCIe can hang on some boards; set NCCL_P2P_DISABLE=1
# if the first all-reduce never completes.
exec "$PYTHON" -m vllm.entrypoints.cli.main serve "$MODEL" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_LEN" \
    --max-num-seqs "$MAX_SEQS" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --reasoning-parser qwen3 \
    --port "$PORT" \
    "${extra[@]}" \
    "$@"
