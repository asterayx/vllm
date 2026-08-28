#!/bin/bash
set -euo pipefail

export PATH="/opt/venv/bin:/usr/local/cuda/bin:${PATH:-}"
export VIRTUAL_ENV="${VIRTUAL_ENV:-/opt/venv}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}"

exec "$@"
