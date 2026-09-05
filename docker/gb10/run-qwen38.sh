#!/usr/bin/env bash
# Serve RadixArk/Qwen3.8-Flash-Next-NVFP4 on 2× DGX Spark (TP=2).
#
# This model is Qwen4Exp (GDN + QSA + PLE + MoE). Official v0.28.0 does not
# include it; this branch backports #53896 and the NVFP4 PLE load fix (#54765).
#
# Weights: either a local dir with config.json, or the HF hub/cache.
#   huggingface-cli download RadixArk/Qwen3.8-Flash-Next-NVFP4 \
#     --local-dir ~/models/Qwen3.8-Flash-Next-NVFP4
#   MODEL_HOST=~/models/Qwen3.8-Flash-Next-NVFP4 ./docker/gb10/run-qwen38.sh
# If MODEL_HOST has no config.json, the script serves the HF repo id and
# does not bind-mount an empty directory (Docker would hide the hub cache).
#
# Head:
#   NODE_RANK=0 ./docker/gb10/run-qwen38.sh
# Worker:
#   NODE_RANK=1 VLLM_HOST_IP=192.168.100.11 HEADLESS=--headless \
#     ./docker/gb10/run-qwen38.sh
#
# By default this script stops leftover Spark serve containers on this
# node (dspark-tp2-rank*, dspark-vision-tp2-rank*). Those hold the 128 GB
# unified GPU and make NCCL fail at ncclCommInitRank with CUDA OOM.
#   STOP_OTHER_SPARK_SERVERS=0 ./docker/gb10/run-qwen38.sh
#
# 262144 context does not fit with the FP8 PLE table on-GPU. Override
# after a successful load if you have headroom:
#   MAX_MODEL_LEN=32768 MAX_NUM_SEQS=4 ./docker/gb10/run-qwen38.sh
#
# Do not set VLLM_PLE_CPU_OFFLOAD=1: PLE offload rejects nnodes=2, so the
# FP8 n-gram table (~25 GiB/rank) stays in GPU memory.
# Do not set --kv-cache-dtype fp8: QSA requires a BF16 main KV cache.
set -euo pipefail

IMAGE="${VLLM_GB10_IMAGE:-vllm-gb10:v0.28.0-dsv4-spark}"
HF_MODEL_ID="${HF_MODEL_ID:-RadixArk/Qwen3.8-Flash-Next-NVFP4}"
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

model_mount=()
hf_offline=0
if [ -f "${MODEL_HOST}/config.json" ]; then
  SERVE_MODEL="${MODEL_PATH}"
  model_mount=(-v "${MODEL_HOST}:${MODEL_PATH}:ro")
  hf_offline=1
  echo "weights: ${MODEL_HOST} -> ${MODEL_PATH}"
else
  if [ -e "${MODEL_HOST}" ]; then
    echo "weights: ${MODEL_HOST} exists but has no config.json; using ${HF_MODEL_ID}" >&2
    if [ -d "${MODEL_HOST}" ]; then
      echo "weights: contents: $(ls -A "${MODEL_HOST}" | head -n 8 | tr '\n' ' ')" >&2
    fi
  else
    echo "weights: ${MODEL_HOST} missing; using ${HF_MODEL_ID}" >&2
    echo "weights: download with --local-dir ${MODEL_HOST} to skip hub lookup" >&2
  fi
  SERVE_MODEL="${HF_MODEL_ID}"
  # Do not bind-mount a missing/empty MODEL_HOST. Docker creates an empty
  # directory and vLLM then rejects /models/... as an invalid local path.
fi
if [ "${HF_HUB_OFFLINE:-}" = "1" ] || [ "${HF_HUB_OFFLINE:-}" = "0" ]; then
  hf_offline="${HF_HUB_OFFLINE}"
fi

MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"

if [ "${QWEN38_RESOLVE_ONLY:-0}" = "1" ]; then
  printf 'serve=%s\noffline=%s\nmount=%s\nmax_model_len=%s\n' \
    "${SERVE_MODEL}" "${hf_offline}" "${model_mount[*]-}" "${MAX_MODEL_LEN}"
  exit 0
fi

src_mount=()
pythonpath_args=()
hf_token_args=()
if [ -n "${HF_TOKEN:-}" ]; then
  hf_token_args=(-e "HF_TOKEN=${HF_TOKEN}")
fi
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

if [ "${STOP_OTHER_SPARK_SERVERS:-1}" != "0" ]; then
  leftover=(
    dspark-tp2-rank0
    dspark-tp2-rank1
    dspark-vision-tp2-rank0
    dspark-vision-tp2-rank1
  )
  for c in "${leftover[@]}"; do
    if docker inspect "${c}" >/dev/null 2>&1; then
      docker rm -f "${c}" >/dev/null
      echo "stopped leftover GPU server: ${c}"
    fi
  done
fi

echo "docker: $(docker ps --format '{{.Names}}' | tr '\n' ' ')"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "gpu: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | tr -d ' ') MiB used/total"
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
    --format=csv,noheader 2>/dev/null | sed 's/^/gpu-proc: /' || true
fi

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
  "${model_mount[@]}" \
  "${src_mount[@]}" \
  -v "${VLLM_CACHE}:/root/.cache/vllm" \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -v "${HF_CACHE}/flashinfer:/root/.cache/flashinfer" \
  -e HF_HUB_OFFLINE="${hf_offline}" \
  -e TRANSFORMERS_OFFLINE="${hf_offline}" \
  "${hf_token_args[@]}" \
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
  serve "${SERVE_MODEL}" \
  --served-model-name "${SERVED_MODEL_NAME:-qwen3.8-flash-next-nvfp4}" \
  --host 0.0.0.0 --port "${VLLM_PORT:-30001}" \
  --trust-remote-code \
  --tensor-parallel-size 2 --pipeline-parallel-size 1 \
  --distributed-executor-backend mp \
  --nnodes 2 --node-rank "${NODE_RANK}" \
  --master-addr "${MASTER_ADDR}" --master-port "${MASTER_PORT:-29500}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
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
