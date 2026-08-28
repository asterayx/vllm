#!/usr/bin/env bash
# Build the GB10 image on this Spark. Run from the repo root.
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "Build this image on aarch64 (the Spark), not x86." >&2
  exit 1
fi

IMAGE="${VLLM_GB10_IMAGE:-vllm-gb10:dspark}"
MAX_JOBS="${MAX_JOBS:-8}"

exec docker build \
  --platform linux/arm64 \
  -f docker/Dockerfile.gb10 \
  --build-arg MAX_JOBS="${MAX_JOBS}" \
  -t "${IMAGE}" \
  .
