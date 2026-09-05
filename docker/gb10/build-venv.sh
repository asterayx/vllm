#!/usr/bin/env bash
# Host compile for DGX Spark: fill a uv venv and build vLLM inplace.
# Does not run use_existing_torch.py (that dirties the git tree).
# Needs rustc/cargo (rust-toolchain.toml, usually rustup 1.95) for vllm-rs.
#
#   source ~/.cargo/env
#   ./docker/gb10/build-venv.sh
#   VENV=~/.venvs/vllm028 RECREATE=1 MAX_JOBS=8 ./docker/gb10/build-venv.sh
# After CUDA build_ext finished, resume rust + editable install:
#   source ~/.cargo/env
#   SKIP_BUILD_EXT=1 ./docker/gb10/install-vllm.sh ~/.venvs/vllm028/bin/python
# Then:
#   VENV=~/.venvs/vllm028 INSTALL_B12X=1 INSTALL_HUMMING=1 ./docker/gb10/pack-venv.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "Build this venv on aarch64 (the Spark), not x86." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "uv not on PATH. export PATH=\"\$HOME/.local/bin:\$PATH\"" >&2
  exit 1
fi

VENV="${VENV:-${HOME}/.venvs/vllm028}"
RECREATE="${RECREATE:-0}"
INSTALL_B12X="${INSTALL_B12X:-1}"
INSTALL_HUMMING="${INSTALL_HUMMING:-1}"
export MAX_JOBS="${MAX_JOBS:-8}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}"
export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-12.1a}"
export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-cuda}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export UV_INDEX_STRATEGY="${UV_INDEX_STRATEGY:-unsafe-best-match}"
# shellcheck source=version.sh
source "$(cd "$(dirname "$0")" && pwd)/version.sh"
export SETUPTOOLS_SCM_PRETEND_VERSION="${SETUPTOOLS_SCM_PRETEND_VERSION:-${VLLM_SPARK_VERSION}}"
export VLLM_VERSION_OVERRIDE="${VLLM_VERSION_OVERRIDE:-${VLLM_SPARK_VERSION}}"
export VLLM_ROOT="${ROOT}"
export VIRTUAL_ENV="${VENV}"
export PATH="${VENV}/bin:${HOME}/.local/bin:${PATH}"

if [[ "${RECREATE}" == "1" && -d "${VENV}" ]]; then
  echo "removing ${VENV}"
  rm -rf "${VENV}"
fi

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "creating ${VENV}"
  uv venv --python 3.12 "${VENV}"
fi
PYTHON="${VENV}/bin/python"

echo "installing requirements/cuda.txt into ${VENV}"
uv pip install --python "${PYTHON}" \
  --torch-backend cu130 \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  -r requirements/cuda.txt

"${PYTHON}" -c "import torch; assert torch.version.cuda, torch.__version__; print('torch', torch.__version__, 'cuda', torch.version.cuda, flush=True)"

echo "installing requirements/build/cuda.txt"
uv pip install --python "${PYTHON}" \
  --torch-backend cu130 \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  -r requirements/build/cuda.txt

"${ROOT}/docker/gb10/install-vllm.sh" "${PYTHON}"

if [[ "${INSTALL_B12X}" == "1" ]]; then
  "${ROOT}/docker/gb10/install-b12x.sh" "${PYTHON}"
fi
if [[ "${INSTALL_HUMMING}" == "1" ]]; then
  "${ROOT}/docker/gb10/install-humming.sh" "${PYTHON}"
fi

echo "venv ready: ${VENV}"
echo "pack image: VENV=${VENV} INSTALL_B12X=${INSTALL_B12X} INSTALL_HUMMING=${INSTALL_HUMMING} ./docker/gb10/pack-venv.sh"
