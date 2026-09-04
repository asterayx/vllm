#!/usr/bin/env bash
# Install humming-kernels[cu13] and verify importlib can see `humming`.
# Packed GB10 venvs are uv-managed and often have no pip module.
set -euo pipefail

if [[ "${INSTALL_HUMMING:-1}" != "1" ]]; then
  echo "INSTALL_HUMMING=${INSTALL_HUMMING:-} skipped"
  exit 0
fi

PYTHON="${1:-/opt/venv/bin/python}"
HUMMING_SPEC="${HUMMING_SPEC:-humming-kernels[cu13]==0.1.12}"

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
echo "Installing ${HUMMING_SPEC} with uv (torch constrained to ${TORCH_PIN})"
uv pip install --python "${PYTHON}" --no-cache \
  -c "${CONSTRAINT}" \
  "${HUMMING_SPEC}"

"${PYTHON}" -c '
import importlib.metadata
import humming  # noqa: F401
print("humming-kernels", importlib.metadata.version("humming-kernels"))
'
# has_humming() is computed at import time; use a fresh interpreter.
"${PYTHON}" -c 'from vllm.utils.import_utils import has_humming; assert has_humming(), "has_humming is False"'
