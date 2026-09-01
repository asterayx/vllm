#!/usr/bin/env bash
# Head (this node is 192.168.100.10):
#   NODE_RANK=0 ./docker/gb10/run.sh
# Worker (set VLLM_HOST_IP to THAT Spark's ConnectX IP):
#   NODE_RANK=1 VLLM_HOST_IP=192.168.100.11 HEADLESS=--headless ./docker/gb10/run.sh
set -euo pipefail

IMAGE="${VLLM_GB10_IMAGE:-vllm-gb10:dspark}"
MODEL_HOST="${MODEL_HOST:-${HOME}/models/DeepSeek-V4-Flash-0731}"
MODEL_PATH="${MODEL_PATH:-/models/DeepSeek-V4-Flash-0731}"
VLLM_SRC="${VLLM_SRC:-$(cd "$(dirname "$0")/../.." && pwd)}"
VLLM_CACHE="${VLLM_CACHE:-${HOME}/vllm-cache}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
B12X_CACHE="${B12X_CACHE:-${HOME}/.cache/b12x}"
MASTER_ADDR="${MASTER_ADDR:-192.168.100.10}"
VLLM_HOST_IP="${VLLM_HOST_IP:-${MASTER_ADDR}}"
NODE_RANK="${NODE_RANK:-0}"
HEADLESS="${HEADLESS:-}"
NAME="${NAME:-dspark-tp2-rank${NODE_RANK}}"
LINEAR_BACKEND="${LINEAR_BACKEND:-b12x}"
MOE_BACKEND="${MOE_BACKEND:-b12x}"

mkdir -p "${VLLM_CACHE}" "${HF_CACHE}/flashinfer" "${B12X_CACHE}"
docker rm -f "${NAME}" 2>/dev/null || true

# 6 seqs * (1 + DSpark k=5) = 36; include 36 so capture is not truncated to 32.
CUGRAPH_CFG='{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"],"cudagraph_capture_sizes":[1,2,4,8,16,24,32,36]}'

docker run -d --name "${NAME}" \
  --gpus all --ipc=host --network host --privileged \
  --shm-size=64g --ulimit memlock=-1 --ulimit stack=67108864 \
  --device /dev/infiniband \
  -v /dev/infiniband:/dev/infiniband \
  -v /sys/class/infiniband:/sys/class/infiniband \
  -v "${MODEL_HOST}:${MODEL_PATH}:ro" \
  -v "${VLLM_SRC}:/opt/vllm" \
  -v "${VLLM_CACHE}:/root/.cache/vllm" \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -v "${HF_CACHE}/flashinfer:/root/.cache/flashinfer" \
  -v "${B12X_CACHE}:/root/.cache/b12x" \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e PYTHONPATH=/opt/vllm \
  -e VLLM_HOST_IP="${VLLM_HOST_IP}" \
  -e VLLM_LOGGING_COLOR=1 \
  -e VLLM_USE_DEEP_GEMM=0 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e B12X_CUTE_COMPILE_CACHE_DIR=/root/.cache/b12x/cute_compile \
  -e B12X_PRINT_COMPILE_PROGRESS="${B12X_PRINT_COMPILE_PROGRESS:-1}" \
  -e NCCL_NET=IB \
  -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_MERGE_NICS=1 \
  -e NCCL_CROSS_NIC=1 \
  -e NCCL_CUMEM_ENABLE=0 \
  -e NCCL_NVLS_ENABLE=0 \
  -e NCCL_IB_ROCE_VERSION_NUM=2 \
  -e NCCL_IB_ADDR_FAMILY=AF_INET \
  -e NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp1s0f1np1}" \
  -e GLOO_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp1s0f1np1}" \
  -e TP_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp1s0f1np1}" \
  -e NCCL_IB_HCA="${NCCL_IB_HCA:-rocep1s0f1,roceP2p1s0f1}" \
  -e NCCL_DEBUG="${NCCL_DEBUG:-WARN}" \
  --entrypoint bash \
  "${IMAGE}" \
  -c 'export PATH=/opt/venv/bin:${PATH}
       for site in /opt/venv/lib/python*/site-packages; do
         [ -d "$site" ] || continue
         rm -f "$site"/__editable__.vllm* "$site"/__editable___vllm*
         echo /opt/vllm > "$site"/_vllm_relocated.pth
       done
       export PYTHONPATH=/opt/vllm${PYTHONPATH:+:$PYTHONPATH}
       exec vllm "$@"' \
  vllm \
  serve "${MODEL_PATH}" \
  --served-model-name deepseek-v4-flash-0731 \
  --host 0.0.0.0 --port "${VLLM_PORT:-30001}" \
  --trust-remote-code \
  --tensor-parallel-size 2 --pipeline-parallel-size 1 \
  --distributed-executor-backend mp \
  --nnodes 2 --node-rank "${NODE_RANK}" \
  --master-addr "${MASTER_ADDR}" --master-port "${MASTER_PORT:-29500}" \
  --kv-cache-dtype fp8_ds_mla --block-size 256 \
  --max-model-len "${MAX_MODEL_LEN:-524288}" \
  --max-num-seqs "${MAX_NUM_SEQS:-6}" --max-num-batched-tokens 8192 \
  --max-cudagraph-capture-size "${MAX_CUGRAPH:-36}" \
  --gpu-memory-utilization 0.87 \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}' \
  --enable-prefix-caching --async-scheduling --enable-chunked-prefill \
  --enable-flashinfer-autotune \
  --linear-backend "${LINEAR_BACKEND}" \
  --moe-backend "${MOE_BACKEND}" \
  --compilation-config "${CUGRAPH_CFG}" \
  ${HEADLESS}

echo "started ${NAME}; logs: docker logs -f ${NAME}"
