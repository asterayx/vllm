#!/usr/bin/env bash
# Serve RadixArk/Qwen3.8-Flash-Next-NVFP4 on 2× DGX Spark (TP=2).
#
# This model is Qwen4Exp (GDN + QSA + PLE + MoE). Official v0.28.0 does not
# include it; this branch backports #53896 and the NVFP4 PLE load fix (#54765).
#
# Download the checkpoint on both nodes first:
#   huggingface-cli download RadixArk/Qwen3.8-Flash-Next-NVFP4 \
#     --local-dir ~/models/Qwen3.8-Flash-Next-NVFP4
#
# Stop any DeepSeek containers that share port 30001:
#   docker rm -f dspark-tp2-rank0 dspark-tp2-rank1
#
# Head:
#   NODE_RANK=0 ./docker/gb10/run-qwen38.sh
# Worker:
#   NODE_RANK=1 VLLM_HOST_IP=192.168.100.11 HEADLESS=--headless \
#     ./docker/gb10/run-qwen38.sh
#
# Do not set VLLM_PLE_CPU_OFFLOAD=1: PLE offload rejects nnodes=2, so the
# FP8 n-gram table (~25 GiB/rank) stays in GPU memory.
# Do not set --kv-cache-dtype fp8: QSA requires a BF16 main KV cache.
set -euo pipefail

IMAGE="${VLLM_GB10_IMAGE:-vllm-gb10:v0.28.0-dsv4-spark}"
MODEL_HOST="${MODEL_HOST:-${HOME}/models/Qwen3.8-Flash-Next-NVFP4}"
MODEL_PATH="${MODEL_PATH:-/models/Qwen3.8-Flash-Next-NVFP4}"
MOUNT_VLLM_SRC="${MOUNT_VLLM_SRC:-1}"
VLLM_CACHE="${VLLM_CACHE:-${HOME}/vllm-cache}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
MASTER_ADDR="${MASTER_ADDR:-192.168.100.10}"
VLLM_HOST_IP="${VLLM_HOST_IP:-${MASTER_ADDR}}"
NODE_RANK="${NODE_RANK:-0}"
HEADLESS="${HEADLESS:-}"
NAME="${NAME:-qwen38-nvfp4-tp2-rank${NODE_RANK}}"

src_mount=()
pythonpath_args=()
if [ "${MOUNT_VLLM_SRC}" != "0" ]; then
  VLLM_SRC="${VLLM_SRC:-$(cd "$(dirname "$0")/../.." && pwd)}"
  src_mount=(-v "${VLLM_SRC}:/opt/vllm")
  pythonpath_args=(-e PYTHONPATH=/opt/vllm)
  echo "dev: mounting ${VLLM_SRC} -> /opt/vllm"
else
  echo "image: using baked /opt/vllm (no host source mount)"
fi

mkdir -p "${VLLM_CACHE}" "${HF_CACHE}/flashinfer"
docker rm -f "${NAME}" 2>/dev/null || true

# JSON defaults cannot live in ${VAR:-...} (bash cuts at the first `}`).
if [ -z "${CUGRAPH_CFG+x}" ]; then
  CUGRAPH_CFG='{"mode":"none","cudagraph_mode":"FULL_DECODE_ONLY"}'
fi

spec_args=()
if [ -n "${SPECULATIVE_CONFIG:-}" ]; then
  spec_args=(--speculative-config "${SPECULATIVE_CONFIG}")
fi

docker run -d --name "${NAME}" \
  --gpus all --ipc=host --network host --privileged \
  --shm-size=64g --ulimit memlock=-1 --ulimit stack=67108864 \
  --device /dev/infiniband \
  -v /dev/infiniband:/dev/infiniband \
  -v /sys/class/infiniband:/sys/class/infiniband \
  -v "${MODEL_HOST}:${MODEL_PATH}:ro" \
  "${src_mount[@]}" \
  -v "${VLLM_CACHE}:/root/.cache/vllm" \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -v "${HF_CACHE}/flashinfer:/root/.cache/flashinfer" \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  "${pythonpath_args[@]}" \
  -e MOUNT_VLLM_SRC="${MOUNT_VLLM_SRC}" \
  -e VLLM_HOST_IP="${VLLM_HOST_IP}" \
  -e VLLM_LOGGING_COLOR=1 \
  -e VLLM_USE_DEEP_GEMM=0 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
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
  -w / \
  --entrypoint bash \
  "${IMAGE}" \
  -c 'export PATH=/opt/venv/bin:${PATH}
       if [ "${MOUNT_VLLM_SRC:-1}" != "0" ]; then
         export PYTHONPATH=/opt/vllm${PYTHONPATH:+:$PYTHONPATH}
         for site in /opt/venv/lib/python*/site-packages; do
           [ -d "$site" ] || continue
           rm -f "$site"/__editable__.vllm* "$site"/__editable___vllm*
           echo /opt/vllm > "$site"/_vllm_relocated.pth
         done
       fi
       exec /opt/venv/bin/python -m vllm.entrypoints.cli.main "$@"' \
  vllm \
  serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME:-qwen3.8-flash-next-nvfp4}" \
  --host 0.0.0.0 --port "${VLLM_PORT:-30001}" \
  --trust-remote-code \
  --tensor-parallel-size 2 --pipeline-parallel-size 1 \
  --distributed-executor-backend mp \
  --nnodes 2 --node-rank "${NODE_RANK}" \
  --master-addr "${MASTER_ADDR}" --master-port "${MASTER_PORT:-29500}" \
  --max-model-len "${MAX_MODEL_LEN:-262144}" \
  --max-num-seqs "${MAX_NUM_SEQS:-8}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-8192}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}" \
  --enable-prefix-caching \
  --no-enable-flashinfer-autotune \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --compilation-config "${CUGRAPH_CFG}" \
  "${spec_args[@]}" \
  ${EXTRA_VLLM_ARGS:-} \
  ${HEADLESS}

echo "started ${NAME}; logs: docker logs -f ${NAME}"
