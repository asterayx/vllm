#!/bin/bash
set -euo pipefail

export PATH="/opt/venv/bin:/usr/local/cuda/bin:${PATH:-}"
export VIRTUAL_ENV="${VIRTUAL_ENV:-/opt/venv}"
export PYTHONPATH="/opt/vllm${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}"

# Packed host venvs are editable. A leftover PEP 660 finder pointing at
# /home/... wins over PYTHONPATH and raises ModuleNotFoundError: vllm.
for site in /opt/venv/lib/python*/site-packages; do
  [[ -d "${site}" ]] || continue
  rm -f "${site}"/__editable__.vllm* "${site}"/__editable___vllm*
  printf '%s\n' /opt/vllm > "${site}/_vllm_relocated.pth"
done

# WORKDIR is /opt/vllm; a bare `vllm` hits the package directory.
if [[ "${1:-}" == "vllm" ]]; then
  shift
  exec /opt/venv/bin/python -m vllm.entrypoints.cli.main "$@"
fi
exec "$@"
