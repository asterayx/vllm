#!/usr/bin/env bash
# Serve GLM-5.3-Flash (Glm5Next: KDA linear attention + sparse MLA with a
# Kpool indexer) on RTX PRO 5000 Blackwell (sm_120) GPUs.
#
# SM120 sparse MLA runs on FlashInfer (FLASHINFER_MLA_SPARSE_SM120); the
# indexer needs DeepGEMM (vendored). FP8 KV cache halves MLA cache size.
#
# Usage:
#   MODEL=~/models/zai-org/GLM-5.3-Flash TP=4 ./glm5_3_flash_rtx_pro_5000.sh
#   NUMA=1        # bind each GPU worker to its NUMA node
#   NCCL_LL=1     # P2P across sockets + LL for decode-sized TP all-reduce
#   SPEC=N        # MTP speculative decoding with N draft tokens
#   TOOLS=1       # enable tool calling (glm47 parser)
set -euo pipefail

MODEL=${MODEL:-zai-org/GLM-5.3-Flash}
TP=${TP:-4}
MAX_LEN=${MAX_LEN:-65536}
MAX_SEQS=${MAX_SEQS:-16}
GPU_UTIL=${GPU_UTIL:-0.93}
PORT=${PORT:-8000}
PYTHON=${PYTHON:-.venv/bin/python}
# Keep colored logs when piped through tee; view saved logs with `less -R`.
export VLLM_LOGGING_COLOR=${VLLM_LOGGING_COLOR:-1}

nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv

extra=()
if [[ "${NUMA:-0}" == "1" ]]; then
    extra+=(--numa-bind)
fi
if [[ "${NCCL_LL:-0}" == "1" ]]; then
    src="$(dirname "$0")/nccl_ll_tuner.c"
    tuner="${XDG_CACHE_HOME:-$HOME/.cache}/vllm/libnccl_ll_tuner.so"
    if [[ ! -f "$tuner" || "$src" -nt "$tuner" ]]; then
        mkdir -p "$(dirname "$tuner")"
        gcc -fPIC -shared -O2 -o "$tuner" "$src"
    fi
    export NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-SYS}
    export NCCL_TUNER_PLUGIN=$tuner
fi
if [[ "${SPEC:-0}" != "0" ]]; then
    extra+=(--speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": $SPEC, \"moe_backend\": \"${SPEC_MOE:-auto}\"}")
fi
if [[ "${TOOLS:-0}" == "1" ]]; then
    extra+=(--enable-auto-tool-choice --tool-call-parser glm47)
fi

exec "$PYTHON" -m vllm.entrypoints.cli.main serve "$MODEL" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_LEN" \
    --max-num-seqs "$MAX_SEQS" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --kv-cache-dtype fp8 \
    --reasoning-parser glm47 \
    --port "$PORT" \
    "${extra[@]}" \
    "$@"
