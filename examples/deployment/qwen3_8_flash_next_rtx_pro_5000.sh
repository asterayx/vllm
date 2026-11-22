#!/usr/bin/env bash
# Serve Qwen3.8-Flash-Next on RTX PRO 5000 Blackwell (sm_120, 48/72 GB) GPUs.
#
# NVFP4 checkpoint footprint (512 experts x 48 layers):
#   GPU:  ~68 GB NVFP4 routed experts + ~10 GB bf16 dense/embeddings
#         + ~2.5 GB FP8 MTP experts + ~1 GB vision  => ~80 GB total
#   Host: ~51 GB FP8 n-gram (PLE) table in pinned memory, sharded across TP
# 48 GB cards: TP=2 is tight, prefer TP=4. 72 GB cards: TP=2 fits easily;
# TP=4 leaves the most KV cache.
#
# Full runbook (install, tunnel, clients, telemetry): qwen3_8_flash_next_rtx_pro_5000.md
#
# Recommended on 4x RTX PRO 5000 72GB, dual-socket, PCIe only (b12x>=1.5.0);
# random 1024/512 at concurrency 1/8/16: ~193/770/1110 output tok/s:
#   NUMA=1 NCCL_LL=1 TP=4 SPEC=2 ./qwen3_8_flash_next_rtx_pro_5000.sh \
#       --moe-backend b12x --kv-cache-dtype fp8
#
# Usage:
#   MODEL=~/models/nvidia/Qwen3.8-Flash-Next-NVFP4 TP=2 ./qwen3_8_flash_next_rtx_pro_5000.sh
#   CHECK=1 ...   # run the sm_120-relevant kernel tests first
#   TEXT_ONLY=1   # skip the vision tower (--language-model-only)
#   EP=1          # shard experts instead of their intermediate dim
#   SPEC=2        # MTP speculative decoding with 2 draft tokens
#   --moe-backend b12x  # needs b12x>=1.5.0 at TP4 (160 per rank, padded to 192)
#   SPEC_MOE=auto # MoE backend for the FP8 MTP layer (not inherited from
#                 # --moe-backend, which may be NVFP4-only, e.g. b12x)
#   NUMA=1        # bind each GPU worker (and its PLE table shard) to its NUMA node
#   NCCL_LL=1     # PCIe-only, multi-socket hosts: P2P across sockets plus the LL
#                 # protocol for decode-sized TP all-reduce (nccl_ll_tuner.c,
#                 # built with gcc on first use; LL_TUNER_MAX_BYTES sets the cutoff)
#   PCIE_IPC=1    # FlashInfer PCIe IPC all-reduce for small decode batches;
#                 # VLLM_ALLREDUCE_FLASHINFER_PCIE_IPC_MAX_TOKENS caps the tokens
set -euo pipefail

MODEL=${MODEL:-nvidia/Qwen3.8-Flash-Next-NVFP4}
TP=${TP:-2}
MAX_LEN=${MAX_LEN:-32768}
MAX_SEQS=${MAX_SEQS:-16}
GPU_UTIL=${GPU_UTIL:-0.95}
PORT=${PORT:-8000}
PYTHON=${PYTHON:-.venv/bin/python}
# Keep colored logs when piped through tee; view saved logs with `less -R`.
export VLLM_LOGGING_COLOR=${VLLM_LOGGING_COLOR:-1}
# FlashInfer JIT needs ninja (installed into the venv) on PATH.
PY_BIN_DIR=$(dirname "$PYTHON")
if [[ -x "$PY_BIN_DIR/ninja" ]]; then
    export PATH="$(cd "$PY_BIN_DIR" && pwd):$PATH"
fi
# vLLM treats FlashInfer as unavailable without flashinfer-cubin or nvcc on
# PATH, silently dropping FlashInfer backends (e.g. SM120 sparse MLA).
if ! command -v nvcc >/dev/null && [[ -x "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" ]]; then
    export PATH="${CUDA_HOME:-/usr/local/cuda}/bin:$PATH"
fi

nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv

if [[ "${CHECK:-0}" == "1" ]]; then
    "$PYTHON" -m pytest -q \
        tests/models/qwen4_exp/test_hc_ops.py \
        tests/models/qwen4_exp/test_qsa_reference.py
fi

extra=()
if [[ "${TEXT_ONLY:-0}" == "1" ]]; then
    extra+=(--language-model-only)
fi
if [[ "${EP:-0}" == "1" ]]; then
    extra+=(--enable-expert-parallel)
fi
if [[ "${NUMA:-0}" == "1" ]]; then
    extra+=(--numa-bind)
fi
if [[ "${NCCL_LL:-0}" == "1" ]]; then
    src="$(dirname "$0")/nccl_ll_tuner.c"
    tuner="${XDG_CACHE_HOME:-$HOME/.cache}/vllm/libnccl_ll_tuner.so"
    if [[ ! -f "$tuner" || "$src" -nt "$tuner" ]]; then
        mkdir -p "$(dirname "$tuner")"
        gcc -fPIC -shared -O2 -o "$tuner" "$src"
    fi
    export NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-SYS}
    export NCCL_TUNER_PLUGIN=$tuner
fi
if [[ "${PCIE_IPC:-0}" == "1" ]]; then
    export VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1
fi
if [[ "${SPEC:-0}" != "0" ]]; then
    extra+=(--speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": $SPEC, \"moe_backend\": \"${SPEC_MOE:-auto}\"}")
fi

# Without NVLink, P2P over PCIe can hang on some boards; set NCCL_P2P_DISABLE=1
# if the first all-reduce never completes.
exec "$PYTHON" -m vllm.entrypoints.cli.main serve "$MODEL" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_LEN" \
    --max-num-seqs "$MAX_SEQS" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --reasoning-parser qwen3 \
    --port "$PORT" \
    "${extra[@]}" \
    "$@"
