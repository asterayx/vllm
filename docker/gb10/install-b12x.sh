#!/usr/bin/env bash
# Install official b12x==1.2.6 (stock extra, not Anemll flashinfer_b12x).
# Packed GB10 venvs are uv-managed and often have no pip module.
# Do not `pip install vllm[b12x]` — that can resolve/replace the packed vLLM.
set -euo pipefail

PYTHON="${1:-/opt/venv/bin/python}"
B12X_VERSION="${B12X_VERSION:-1.2.6}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "No python at ${PYTHON}" >&2
  exit 1
fi

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return
  fi
  if [[ -x /root/.local/bin/uv ]]; then
    PATH="/root/.local/bin:${PATH}"
    export PATH
    return
  fi
  if ! command -v curl >/dev/null 2>&1; then
    echo "Need curl to install uv (packed venv has no pip)" >&2
    exit 1
  fi
  curl -LsSf https://astral.sh/uv/install.sh | sh
  PATH="/root/.local/bin:${PATH}"
  export PATH
  if ! command -v uv >/dev/null 2>&1; then
    echo "uv install failed" >&2
    exit 1
  fi
}

TORCH_PIN="$("${PYTHON}" -c 'import torch; print(torch.__version__.split("+", 1)[0])')"
CONSTRAINT="$(mktemp)"
trap 'rm -f "${CONSTRAINT}"' EXIT
printf 'torch==%s\n' "${TORCH_PIN}" > "${CONSTRAINT}"

ensure_uv
echo "Installing b12x==${B12X_VERSION} with uv (torch constrained to ${TORCH_PIN})"
uv pip install --python "${PYTHON}" --no-cache \
  -c "${CONSTRAINT}" \
  "b12x==${B12X_VERSION}"

# has_b12x() is computed at import time; verify in a fresh interpreter.
"${PYTHON}" -c 'import b12x; print("b12x", b12x.__version__)'
"${PYTHON}" -c 'from vllm.utils.b12x import has_b12x; assert has_b12x(), "has_b12x is False"'
