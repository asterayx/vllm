#!/usr/bin/env bash
# Make `import humming` work in the serve venv. Bind-mount + leftover
# kernel order is not enough: has_humming() is False until the wheel is
# in /opt/venv, and then Triton still owns W8A8 (GB10 json warning).
#
# Called from run.sh / entrypoint. Never fails the serve.
set -euo pipefail

PYTHON="${1:-/opt/venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  echo "ensure-humming: no python at ${PYTHON}" >&2
  exit 0
fi
if "${PYTHON}" -c 'import humming' >/dev/null 2>&1; then
  echo "humming already importable"
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "SM12x leftover W8A8 needs humming-kernels; installing into ${PYTHON}"
if ! INSTALL_HUMMING=1 "${SCRIPT_DIR}/install-humming.sh" "${PYTHON}"; then
  echo "WARN: humming install failed; leftover W8A8 will use Triton" >&2
fi
