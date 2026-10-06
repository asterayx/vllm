#!/usr/bin/env bash
# Serve Qwen3.8-Flash-Next on RTX PRO 5000 Blackwell (sm_120, 48 GB) GPUs.
#
# The FP8 checkpoint does not fit on a single 48 GB card, so TP defaults to 2;
# raise it if `nvidia-smi` shows more cards and you need longer contexts.
#
# Usage:
#   CHECK=1 ./qwen3_8_flash_next_rtx_pro_5000.sh   # run sm_120 kernel tests first
#   TP=4 MAX_LEN=65536 ./qwen3_8_flash_next_rtx_pro_5000.sh
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen3.8-Flash-Next-FP8}
TP=${TP:-2}
MAX_LEN=${MAX_LEN:-32768}
GPU_UTIL=${GPU_UTIL:-0.92}
PORT=${PORT:-8000}
PYTHON=${PYTHON:-.venv/bin/python}

nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv

if [[ "${CHECK:-0}" == "1" ]]; then
    "$PYTHON" -m pytest -q \
        tests/models/qwen4_exp/test_hc_ops.py \
        tests/models/qwen4_exp/test_qsa_reference.py
fi

# Without NVLink, P2P over PCIe can hang on some boards; set NCCL_P2P_DISABLE=1
# if the first all-reduce never completes.
exec "$PYTHON" -m vllm.entrypoints.cli.main serve "$MODEL" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --reasoning-parser qwen3 \
    --port "$PORT" \
    "$@"
