#!/usr/bin/env bash
# Pack the already-built host venv into vllm-gb10:dspark. Does not compile vLLM.
# Run from the repo root on the Spark that already has the compiled venv.
#
#   VENV=/home/roccen/.venvs/vllm028 ./docker/gb10/pack-venv.sh
#   INSTALL_FLASHINFER=1 VENV=/home/roccen/.venvs/vllm028 ./docker/gb10/pack-venv.sh
#   INSTALL_B12X=1 VENV=/home/roccen/.venvs/vllm028 ./docker/gb10/pack-venv.sh
# To inject b12x into an already-packed image without recopying the venv:
#   ./docker/gb10/pack-b12x.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "Pack this image on aarch64 (the Spark), not x86." >&2
  exit 1
fi

if [[ -z "${VENV:-}" ]]; then
  for candidate in "${HOME}/.venvs/vllm028" "${HOME}/.venv/vllm028"; do
    if [[ -x "${candidate}/bin/python" ]]; then
      VENV="${candidate}"
      break
    fi
  done
fi
VENV="${VENV:-${HOME}/.venvs/vllm028}"
IMAGE="${VLLM_GB10_IMAGE:-vllm-gb10:dspark}"
INSTALL_FLASHINFER="${INSTALL_FLASHINFER:-${INSTALL_FLASHINFER_NIGHTLY:-0}}"
INSTALL_B12X="${INSTALL_B12X:-0}"

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "No venv at ${VENV}. Set VENV=/home/roccen/.venvs/vllm028" >&2
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
cp docker/gb10/install-ibverbs.sh "${STAGE}/install-ibverbs.sh"
cp docker/gb10/install-b12x.sh "${STAGE}/install-b12x.sh"
cp docker/Dockerfile.gb10-venv "${STAGE}/Dockerfile"
./docker/gb10/collect-ibverbs.sh "${STAGE}/ibverbs"

echo "Packing ${VENV} + host ibverbs + $(pwd) -> ${IMAGE} (no vLLM compile)"
docker build \
  --platform linux/arm64 \
  -f "${STAGE}/Dockerfile" \
  --build-arg OLD_VENV="${VENV}" \
  --build-arg OLD_SRC="$(pwd)" \
  --build-arg INSTALL_FLASHINFER="${INSTALL_FLASHINFER}" \
  --build-arg INSTALL_FLASHINFER_NIGHTLY="${INSTALL_FLASHINFER_NIGHTLY:-0}" \
  --build-arg INSTALL_B12X="${INSTALL_B12X}" \
  -t "${IMAGE}" \
  "${STAGE}"
