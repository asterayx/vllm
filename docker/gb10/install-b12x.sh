#!/usr/bin/env bash
# Install official b12x==1.2.6 (stock extra, not Anemll flashinfer_b12x).
# Do not `pip install vllm[b12x]` — that can resolve/replace the packed vLLM.
set -euo pipefail

PYTHON="${1:-/opt/venv/bin/python}"
B12X_VERSION="${B12X_VERSION:-1.2.6}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "No python at ${PYTHON}" >&2
  exit 1
fi

TORCH_PIN="$("${PYTHON}" -c 'import torch; print(torch.__version__.split("+", 1)[0])')"
CONSTRAINT="$(mktemp)"
trap 'rm -f "${CONSTRAINT}"' EXIT
printf 'torch==%s\n' "${TORCH_PIN}" > "${CONSTRAINT}"

echo "Installing b12x==${B12X_VERSION} (torch constrained to ${TORCH_PIN})"
"${PYTHON}" -m pip install --no-cache-dir \
  --upgrade-strategy only-if-needed \
  -c "${CONSTRAINT}" \
  "b12x==${B12X_VERSION}"

# has_b12x() is computed at import time; verify in a fresh interpreter.
"${PYTHON}" -c 'import b12x; print("b12x", b12x.__version__)'
"${PYTHON}" -c 'from vllm.utils.b12x import has_b12x; assert has_b12x(), "has_b12x is False"'
