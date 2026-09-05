#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
binary="$repo_root/docker/gb10/compat-proxy/target/release/spark-compat-proxy"
if [[ ! -x "$binary" ]]; then
    echo "Build docker/gb10/compat-proxy with cargo build --locked --release first." >&2
    exit 2
fi

exec "$binary" \
    --listen "${PROXY_LISTEN:-127.0.0.1:30000}" \
    --upstream "${VLLM_UPSTREAM:-http://127.0.0.1:${API_PORT:-18029}}" \
    --public-base "${PUBLIC_BASE:-https://token.asterayx.com}" \
    --model qwen38-nvfp4 \
    --display-name Qwen3.8-Flash-Next-NVFP4 \
    --context-window "${MAX_MODEL_LEN:-524288}" \
    --upstream-read-timeout-secs "${UPSTREAM_READ_TIMEOUT_SECS:-3600}" \
    "$@"
