#!/usr/bin/env bash
# Serve DeepSeek-V4.1-Flash on 4x RTX PRO 5000 Blackwell (sm_120) with the
# Engram tables offloaded to host DDR.
#
# Engram: two layers of ~384M FP8 rows, ~200 GB of tables. They stay in pinned
# host memory (cpu_offload, the default) and are read over PCIe through UVA,
# prefetched on a side stream. Stop other servers that pin host memory (e.g.
# the Qwen3.8 PLE table, ~51 GB) before starting; check `free -g` first.
# SM120 attention is FlashInfer sparse MLA and requires an FP8 KV cache.
#
# Usage:
#   MODEL=~/models/deepseek-ai/DeepSeek-V4.1-Flash ./deepseek_v4_1_flash_rtx_pro_5000.sh
#   NUMA=1        # bind each GPU worker (and its Engram shard) to its NUMA node
#   NCCL_LL=1     # P2P across sockets + LL for decode-sized TP all-reduce
#   ENGRAM_THP=1  # back the host Engram tables with transparent huge pages
#   SPEC=N        # speculative decoding with the checkpoint's draft head
#   SPEC_METHOD=dspark  # speculative method name (default dspark)
set -euo pipefail

MODEL=${MODEL:-deepseek-ai/DeepSeek-V4.1-Flash}
TP=${TP:-4}
MAX_LEN=${MAX_LEN:-65536}
MAX_SEQS=${MAX_SEQS:-16}
GPU_UTIL=${GPU_UTIL:-0.93}
PORT=${PORT:-8000}
PYTHON=${PYTHON:-.venv/bin/python}

nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv
free -g | head -2

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
if [[ "${ENGRAM_THP:-0}" == "1" ]]; then
    extra+=(--engram-config '{"cpu_offload": true, "use_thp": true}')
fi
if [[ "${SPEC:-0}" != "0" ]]; then
    extra+=(--speculative-config "{\"method\": \"${SPEC_METHOD:-dspark}\", \"num_speculative_tokens\": $SPEC}")
fi

exec "$PYTHON" -m vllm.entrypoints.cli.main serve "$MODEL" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_LEN" \
    --max-num-seqs "$MAX_SEQS" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --kv-cache-dtype fp8 \
    --reasoning-parser deepseek_v41 \
    --port "$PORT" \
    "${extra[@]}" \
    "$@"
