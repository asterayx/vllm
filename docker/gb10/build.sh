#!/usr/bin/env bash
# From-scratch image (downloads wheels, compiles vLLM). Rebuilds reuse the
# BuildKit uv cache. Host compile + pack instead:
#   ./docker/gb10/build-venv.sh
#   VENV=~/.venvs/vllm028 ./docker/gb10/pack-venv.sh
# Do not --no-cache unless a cached RUN layer is known-bad.
# Do not pack-venv an old later-main ~/.venvs/vllm028 onto this tag.
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "Build this image on aarch64 (the Spark), not x86." >&2
  exit 1
fi

# shellcheck source=version.sh
source "$(dirname "$0")/version.sh"
IMAGE="${VLLM_GB10_IMAGE}"
MAX_JOBS="${MAX_JOBS:-8}"

export DOCKER_BUILDKIT=1
docker build \
  --platform linux/arm64 \
  -f docker/Dockerfile.gb10 \
  --build-arg MAX_JOBS="${MAX_JOBS}" \
  --build-arg VLLM_SPARK_VERSION="${VLLM_SPARK_VERSION}" \
  -t "${IMAGE}" \
  -t "${VLLM_GB10_IMAGE_RELEASE}" \
  .
echo "tagged ${IMAGE} and ${VLLM_GB10_IMAGE_RELEASE} (${VLLM_SPARK_VERSION})"
