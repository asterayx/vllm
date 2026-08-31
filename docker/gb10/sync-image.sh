#!/usr/bin/env bash
# Copy the built image to the other Spark over SSH (use the ConnectX IP).
#   docker/gb10/sync-image.sh user@192.168.x.x
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 user@<other-spark-connectx-ip>" >&2
  exit 1
fi

REMOTE="$1"
IMAGE="${VLLM_GB10_IMAGE:-vllm-gb10:dspark}"

echo "Saving ${IMAGE} and loading on ${REMOTE}..."
docker save "${IMAGE}" | ssh -C "${REMOTE}" docker load
echo "Loaded ${IMAGE} on ${REMOTE}."
echo "Rsync HF weights separately if the other node does not have them:"
echo "  rsync -aHAX --info=progress2 ~/.cache/huggingface/ ${REMOTE}:.cache/huggingface/"
