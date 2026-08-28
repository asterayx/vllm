#!/usr/bin/env bash
# Pack the already-built host venv into vllm-gb10:dspark. Does not compile vLLM.
# Run from the repo root on the Spark that already has ~/.venv/vllm028.
#
#   ./docker/gb10/pack-venv.sh
#   INSTALL_FLASHINFER_NIGHTLY=1 ./docker/gb10/pack-venv.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "Pack this image on aarch64 (the Spark), not x86." >&2
  exit 1
fi

VENV="${VENV:-${HOME}/.venv/vllm028}"
IMAGE="${VLLM_GB10_IMAGE:-vllm-gb10:dspark}"
INSTALL_FLASHINFER_NIGHTLY="${INSTALL_FLASHINFER_NIGHTLY:-0}"

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "No venv at ${VENV}. Set VENV=..." >&2
  exit 1
fi

STAGE="$(mktemp -d)"
trap 'rm -rf "${STAGE}"' EXIT

mkdir -p "${STAGE}/venv" "${STAGE}/src"
rsync -a --delete "${VENV}/" "${STAGE}/venv/"
rsync -a --delete \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '.mypy_cache/' \
  --exclude '.pytest_cache/' \
  ./ "${STAGE}/src/"
cp docker/gb10/relocate-venv.sh "${STAGE}/relocate-venv.sh"
cp docker/Dockerfile.gb10-venv "${STAGE}/Dockerfile"

echo "Packing ${VENV} + $(pwd) -> ${IMAGE} (no vLLM compile)"
docker build \
  --platform linux/arm64 \
  -f "${STAGE}/Dockerfile" \
  --build-arg OLD_VENV="${VENV}" \
  --build-arg OLD_SRC="$(pwd)" \
  --build-arg INSTALL_FLASHINFER_NIGHTLY="${INSTALL_FLASHINFER_NIGHTLY}" \
  -t "${IMAGE}" \
  "${STAGE}"
