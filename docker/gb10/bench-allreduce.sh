#!/usr/bin/env bash
# NCCL all-reduce latency between the two Sparks, in a throwaway container
# with the same NCCL environment run.sh gives the server. Stop the server
# first (it holds the GPU memory), then on each node:
#   MASTER_ADDR=192.168.101.12 NODE_RANK=0 ./docker/gb10/bench-allreduce.sh
#   MASTER_ADDR=192.168.101.12 NODE_RANK=1 ./docker/gb10/bench-allreduce.sh
# Any NCCL_* variable set on the host is forwarded, so settings can be
# compared without restarting the server, e.g.
#   NCCL_PROTO=Simple MASTER_ADDR=... NODE_RANK=0 ./docker/gb10/bench-allreduce.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=version.sh
source "${HERE}/version.sh"
IMAGE="${VLLM_GB10_IMAGE}"
MASTER_ADDR="${MASTER_ADDR:-192.168.100.10}"
MASTER_PORT="${MASTER_PORT:-29577}"
NODE_RANK="${NODE_RANK:-0}"
VLLM_SRC="${VLLM_SRC:-$(cd "${HERE}/../.." && pwd)}"

nccl_args=()
while IFS='=' read -r name _; do
  nccl_args+=(-e "${name}")
done < <(env | grep -E '^NCCL_' || true)

docker run --rm -i --gpus all --ipc=host --network host --privileged \
  --ulimit memlock=-1 \
  --device /dev/infiniband \
  -v /dev/infiniband:/dev/infiniband \
  -v /sys/class/infiniband:/sys/class/infiniband \
  -v "${VLLM_SRC}:/opt/vllm" \
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
  -e NCCL_IB_HCA="${NCCL_IB_HCA:-rocep1s0f1,roceP2p1s0f1}" \
  -e NCCL_DEBUG="${NCCL_DEBUG:-WARN}" \
  "${nccl_args[@]}" \
  -e MASTER_ADDR="${MASTER_ADDR}" \
  -e MASTER_PORT="${MASTER_PORT}" \
  -e RANK="${NODE_RANK}" \
  -e WORLD_SIZE=2 \
  -e LOCAL_RANK=0 \
  -w /opt/vllm \
  "${IMAGE}" \
  python docker/gb10/bench-allreduce.py "$@"
