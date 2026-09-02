#!/usr/bin/env bash
# Copy a built image to the other Spark over SSH (use the ConnectX IP).
#
#   ./docker/gb10/sync-image.sh roccen@192.168.100.11
#   ./docker/gb10/sync-image.sh roccen@192.168.100.11 vllm-gb10:v0.28.0-dsv4-spark
#   VLLM_GB10_IMAGE=vllm-gb10:v0.28.0-dsv4-spark \
#     ./docker/gb10/sync-image.sh roccen@192.168.100.11
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 user@<other-spark-connectx-ip> [image]" >&2
  exit 1
fi

REMOTE="$1"
IMAGE="${2:-${VLLM_GB10_IMAGE:-vllm-gb10:v0.28.0-dsv4-spark}}"

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "No local image ${IMAGE}." >&2
  echo "List: docker image ls | grep vllm-gb10" >&2
  echo "Pass the tag you built, e.g.:" >&2
  echo "  $0 ${REMOTE} vllm-gb10:v0.28.0-dsv4-spark" >&2
  exit 1
fi

echo "Saving ${IMAGE} and loading on ${REMOTE}..."
docker save "${IMAGE}" | ssh -C "${REMOTE}" docker load
echo "Loaded ${IMAGE} on ${REMOTE}."
echo "Start the worker with the same tag:"
echo "  VLLM_GB10_IMAGE=${IMAGE} NODE_RANK=1 VLLM_HOST_IP=192.168.100.11 HEADLESS=--headless ./docker/gb10/run-image.sh"
echo "Rsync HF weights separately if the other node does not have them:"
echo "  rsync -aHAX --info=progress2 ~/.cache/huggingface/ ${REMOTE}:.cache/huggingface/"
