#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PREFIX="${PREFIX:-/usr/local}"
BIN="${PREFIX}/bin/spark-compat-proxy"

if command -v rustup >/dev/null 2>&1; then
  # Prefer an installed toolchain that can parse edition 2024 crates.
  # Some images have a broken rustup "1.95" default (missing rust-std).
  TOOLCHAIN="${RUSTUP_TOOLCHAIN:-}"
  if [[ -z "${TOOLCHAIN}" ]]; then
    for cand in 1.88.0 1.85.0 1.83.0 stable; do
      if rustup run "${cand}" rustc --version >/dev/null 2>&1; then
        TOOLCHAIN="${cand}"
        break
      fi
    done
  fi
  if [[ -n "${TOOLCHAIN}" ]]; then
    CARGO=(cargo "+${TOOLCHAIN}")
  else
    CARGO=(cargo)
  fi
else
  CARGO=(cargo)
fi

"${CARGO[@]}" build --release --manifest-path "${ROOT}/Cargo.toml"
install -m755 "${ROOT}/target/release/spark-compat-proxy" "${BIN}"
echo "installed ${BIN}"
echo "  spark-compat-proxy --help"
echo "  spark-compat-proxy write-configs --out /tmp/spark-client-configs"
