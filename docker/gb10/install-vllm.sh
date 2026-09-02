#!/usr/bin/env bash
# Compile vLLM CUDA extensions into /opt/vllm, then install the package.
# Fail if cmake/nvcc are missing or _C_stable_libtorch is not produced.
set -euo pipefail

PYTHON="${1:-/opt/venv/bin/python}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}"
export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-cuda}"
export MAX_JOBS="${MAX_JOBS:-8}"
export SETUPTOOLS_SCM_PRETEND_VERSION="${SETUPTOOLS_SCM_PRETEND_VERSION:-0.28.0+dsv4.spark}"

command -v cmake >/dev/null
command -v nvcc >/dev/null
cmake --version
nvcc --version
"${PYTHON}" -c "import torch; print('torch', torch.__version__, torch.version.cuda, flush=True)"

echo "building vLLM extensions inplace (MAX_JOBS=${MAX_JOBS})"
"${PYTHON}" setup.py build_ext --inplace

echo "installing editable vllm metadata"
uv pip install --python "${PYTHON}" --no-build-isolation --reinstall-package vllm -e .

"${PYTHON}" /opt/vllm/docker/gb10/check-extensions.py
