#!/usr/bin/env bash
# Serve official DeepSeek-V4-Flash-Vision-Exp on 2× DGX Spark.
# Same stack as docker/gb10/run.sh (b12x, SM12x graphs) but DSpark k=3:
# Vision-Exp ships num_nextn_predict_layers=3; k=5 held ~68% accept on
# the first 10s then fell to ~27% on long text CoT (pos 4/5 ~0.11/0.03).
# k=3 is next_n=4, a native SM12x decode width (no 5→6 pad).
# 6 seqs * 4 = 24.
#
# Download the checkpoint on both nodes first, e.g.:
#   huggingface-cli download deepseek-ai/DeepSeek-V4-Flash-Vision-Exp \
#     --local-dir ~/models/DeepSeek-V4-Flash-Vision-Exp
#
# Stop the text-only 0731 containers first (same default port 30001):
#   docker rm -f dspark-tp2-rank0 dspark-tp2-rank1
#
# Head (bind-mount host source):
#   NODE_RANK=0 ./docker/gb10/run-vision.sh
# Worker:
#   NODE_RANK=1 VLLM_HOST_IP=192.168.100.11 HEADLESS=--headless \
#     ./docker/gb10/run-vision.sh
# Release (baked /opt/vllm): ./docker/gb10/run-vision-image.sh
#
# If load OOMs on the vision tower, retry with e.g.
#   GPU_MEMORY_UTILIZATION=0.82 MAX_NUM_SEQS=4 ./docker/gb10/run-vision.sh
set -euo pipefail

export MODEL_HOST="${MODEL_HOST:-${HOME}/models/DeepSeek-V4-Flash-Vision-Exp}"
export MODEL_PATH="${MODEL_PATH:-/models/DeepSeek-V4-Flash-Vision-Exp}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-v4-flash-vision-exp}"
export NAME="${NAME:-dspark-vision-tp2-rank${NODE_RANK:-0}}"
# Do not put JSON inside ${VAR:-...}: bash ends the expansion at the
# first `}` and produced `...probabilistic"}}`.
if [ -z "${EXTRA_VLLM_ARGS+x}" ]; then
  export EXTRA_VLLM_ARGS='--limit-mm-per-prompt {"image":4}'
fi
if [ -z "${SPECULATIVE_CONFIG+x}" ]; then
  export SPECULATIVE_CONFIG='{"method":"dspark","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}'
fi
export MAX_CUGRAPH="${MAX_CUGRAPH:-24}"
if [ -z "${CUGRAPH_CFG+x}" ]; then
  export CUGRAPH_CFG='{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"],"cudagraph_capture_sizes":[1,2,4,8,12,16,24]}'
fi

exec "$(cd "$(dirname "$0")" && pwd)/run.sh"
