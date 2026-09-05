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
SRC="${ROOT}/target/release/spark-compat-proxy"
if [[ ! -x "${SRC}" ]]; then
  echo "build did not produce ${SRC}" >&2
  exit 1
fi

# /usr/local/bin is root-owned on a new Spark. systemd 203/EXEC is
# almost always "ExecStart path missing" after a non-root install.
BIN_DIR="$(dirname "${BIN}")"
if [[ -w "${BIN_DIR}" ]]; then
  install -m755 "${SRC}" "${BIN}"
else
  echo "installing ${BIN} with sudo (${BIN_DIR} not writable)"
  sudo install -m755 "${SRC}" "${BIN}"
fi
if [[ ! -x "${BIN}" ]]; then
  echo "not executable: ${BIN}" >&2
  ls -l "${BIN}" >&2 || true
  exit 1
fi
"${BIN}" --help >/dev/null
echo "installed ${BIN}"

if [[ "${INSTALL_SYSTEMD:-0}" == "1" ]]; then
  UNIT_USER="${UNIT_USER:-$(id -un)}"
  PUBLIC_BASE="${PUBLIC_BASE:-http://127.0.0.1:30000}"
  tmp="$(mktemp)"
  sed -e "s/^User=.*/User=${UNIT_USER}/" \
      -e "s|--public-base [^[:space:]]*|--public-base ${PUBLIC_BASE}|" \
      "${ROOT}/spark-compat-proxy.service" > "${tmp}"
  sudo install -m644 "${tmp}" /etc/systemd/system/spark-compat-proxy.service
  rm -f "${tmp}"
  sudo systemctl daemon-reload
  echo "installed unit User=${UNIT_USER} public-base=${PUBLIC_BASE}"
fi

echo "  ${BIN} --help"
echo "  ${BIN} write-configs --out /tmp/spark-client-configs"
