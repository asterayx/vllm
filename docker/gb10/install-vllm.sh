#!/usr/bin/env bash
# Compile vLLM CUDA extensions inplace, then install editable metadata.
# Used by docker/gb10/build.sh (in-image) and docker/gb10/build-venv.sh (host).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"

PYTHON="${1:-${VIRTUAL_ENV:-/opt/venv}/bin/python}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}"
export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-cuda}"
export MAX_JOBS="${MAX_JOBS:-8}"
export SETUPTOOLS_SCM_PRETEND_VERSION="${SETUPTOOLS_SCM_PRETEND_VERSION:-0.28.0+dsv4.spark}"
export VLLM_ROOT="${VLLM_ROOT:-${ROOT}}"

if ! command -v cmake >/dev/null; then
  echo "cmake not found. apt-get install -y cmake ninja-build build-essential" >&2
  exit 1
fi
if ! command -v nvcc >/dev/null; then
  echo "nvcc not found. Use the CUDA devel toolkit (nvcc on PATH)." >&2
  exit 1
fi
cmake --version
nvcc --version
"${PYTHON}" -c "import torch; print('torch', torch.__version__, torch.version.cuda, flush=True)"

echo "building vLLM extensions inplace (MAX_JOBS=${MAX_JOBS})"
"${PYTHON}" setup.py build_ext --inplace

echo "installing editable vllm metadata"
uv pip install --python "${PYTHON}" --no-build-isolation --reinstall-package vllm -e .

"${PYTHON}" "${ROOT}/docker/gb10/check-extensions.py"
