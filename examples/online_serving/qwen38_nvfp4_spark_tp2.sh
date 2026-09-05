#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

rank=${1:?Usage: bash qwen38_nvfp4_spark_tp2.sh NODE_RANK [extra vllm arguments]}
shift
if [[ "$rank" != 0 && "$rank" != 1 ]]; then
    echo "NODE_RANK must be 0 or 1" >&2
    exit 2
fi

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export PATH="$repo_root/.venv/bin:$PATH"
model=${MODEL_PATH:-$HOME/models/Qwen3.8-Flash-Next-NVFP4}
if [[ ! -f "$model/config.json" ]]; then
    echo "Set MODEL_PATH to the local checkpoint directory (config.json missing)." >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export VLLM_HOST_IP=${VLLM_HOST_IP:?Set VLLM_HOST_IP to the local interconnect IP}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp1s0f1np1}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-$NCCL_SOCKET_IFNAME}
export VLLM_USE_DEEP_GEMM=0
export FLASHINFER_CUDA_ARCH_LIST=${FLASHINFER_CUDA_ARCH_LIST:-12.1a}
export MAX_JOBS=${MAX_JOBS:-4}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$repo_root/.cache/vllm}
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-$repo_root/.cache/flashinfer}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$repo_root/.cache/triton}
if [[ -f "$repo_root/.venv/include/python3.12/Python.h" ]]; then
    export CPATH="$repo_root/.venv/include/python3.12:$repo_root/.venv/include${CPATH:+:$CPATH}"
fi

args=(
    "$model"
    --served-model-name qwen38-nvfp4
    --tensor-parallel-size 2
    --distributed-executor-backend mp
    --nnodes 2
    --node-rank "$rank"
    --master-addr "${MASTER_ADDR:?Set MASTER_ADDR to the rank 0 interconnect IP}"
    --master-port "${MASTER_PORT:-29529}"
    --quantization modelopt
    --moe-backend "${MOE_BACKEND:-flashinfer_cutlass}"
    --kernel-config '{"enable_flashinfer_autotune": false}'
    --dtype bfloat16
    --max-model-len "${MAX_MODEL_LEN:-524288}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-2048}"
    --max-num-seqs "${MAX_NUM_SEQS:-4}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.80}"
    --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES:-17179869184}"
    --disable-custom-all-reduce
    --enforce-eager
)
if (( ${MAX_MODEL_LEN:-524288} > 262144 )); then
    args+=(--hf-overrides '{"text_config":{"rope_parameters":{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":262144,"rope_theta":10000000,"partial_rotary_factor":0.25,"mrope_section":[11,11,10],"mrope_interleaved":true}}}')
fi
if [[ "$rank" == 1 ]]; then
    args+=(--headless)
else
    args+=(--host "${API_HOST:-127.0.0.1}" --port "${API_PORT:-18029}")
fi

exec "$repo_root/.venv/bin/vllm" serve "${args[@]}" "$@"
