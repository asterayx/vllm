#!/usr/bin/env bash
# From-scratch image (re-downloads torch, recompiles vLLM).
# Do not --no-cache unless a cached RUN layer is known-bad.
# Do not pack-venv an old later-main ~/.venvs/vllm028 onto this tag.
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "Build this image on aarch64 (the Spark), not x86." >&2
  exit 1
fi

IMAGE="${VLLM_GB10_IMAGE:-vllm-gb10:v0.28.0-dsv4-spark}"
MAX_JOBS="${MAX_JOBS:-8}"

exec docker build \
  --platform linux/arm64 \
  -f docker/Dockerfile.gb10 \
  --build-arg MAX_JOBS="${MAX_JOBS}" \
  -t "${IMAGE}" \
  .
