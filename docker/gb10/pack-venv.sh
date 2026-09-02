#!/usr/bin/env bash
# Pack an already-compiled host venv into the GB10 image. Does not compile.
# Build that venv first:
#   ./docker/gb10/build-venv.sh
# The venv must be from THIS v0.28.0 tree (vllm._C_stable_libtorch present).
# Do not pack later-main ~/.venvs/vllm028 onto vllm-gb10:v0.28.0-dsv4-spark.
#
#   VENV=/home/roccen/.venvs/vllm028 ./docker/gb10/pack-venv.sh
#   INSTALL_FLASHINFER=1 VENV=... ./docker/gb10/pack-venv.sh
#   INSTALL_B12X=1 VENV=... ./docker/gb10/pack-venv.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "Pack this image on aarch64 (the Spark), not x86." >&2
  exit 1
fi

if [[ -z "${VENV:-}" ]]; then
  for candidate in \
      "${HOME}/.venvs/vllm-gb10-v0280" \
      "${HOME}/.venvs/vllm028" \
      "${HOME}/.venv/vllm028"; do
    if [[ -x "${candidate}/bin/python" ]]; then
      VENV="${candidate}"
      break
    fi
  done
fi
VENV="${VENV:-${HOME}/.venvs/vllm-gb10-v0280}"
IMAGE="${VLLM_GB10_IMAGE:-vllm-gb10:v0.28.0-dsv4-spark}"
INSTALL_FLASHINFER="${INSTALL_FLASHINFER:-0}"
INSTALL_B12X="${INSTALL_B12X:-0}"

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "No venv at ${VENV}. Build one on this tree first, or use build.sh." >&2
  exit 1
fi

if ! PYTHONPATH="$(pwd)" "${VENV}/bin/python" -c \
    "import importlib; importlib.import_module('vllm._C_stable_libtorch')"; then
  echo "${VENV} has no vllm._C_stable_libtorch." >&2
  echo "That is a later-main or incomplete venv. Use:" >&2
  echo "  VLLM_GB10_IMAGE=${IMAGE} ./docker/gb10/build.sh" >&2
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
cp docker/gb10/check-extensions.py "${STAGE}/check-extensions.py"
cp docker/Dockerfile.gb10-venv "${STAGE}/Dockerfile"
./docker/gb10/collect-ibverbs.sh "${STAGE}/ibverbs"

echo "Packing ${VENV} + host ibverbs + $(pwd) -> ${IMAGE} (no vLLM compile)"
docker build \
  --platform linux/arm64 \
  -f "${STAGE}/Dockerfile" \
  --build-arg OLD_VENV="${VENV}" \
  --build-arg OLD_SRC="$(pwd)" \
  --build-arg INSTALL_FLASHINFER="${INSTALL_FLASHINFER}" \
  --build-arg INSTALL_B12X="${INSTALL_B12X}" \
  -t "${IMAGE}" \
  "${STAGE}"
