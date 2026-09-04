#!/usr/bin/env bash
# Inject humming-kernels[cu13]==0.1.12 into an existing GB10 image.
# Does not recopy the venv and does not compile vLLM.
#
#   ./docker/gb10/pack-humming.sh
#   VLLM_GB10_IMAGE=vllm-gb10:v0.28.0-dsv4-spark ./docker/gb10/pack-humming.sh
#   ./docker/gb10/sync-image.sh roccen@<worker-connectx-ip>
#
# SM121 leftover W8A8 picks Humming before the untuned Triton default.
# Do not add humming to --linear-backend b12x.
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "Pack humming on aarch64 (the Spark), not x86." >&2
  exit 1
fi

# shellcheck source=version.sh
source "$(dirname "$0")/version.sh"
IMAGE="${VLLM_GB10_IMAGE}"
if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "No image ${IMAGE}. Pack the venv first." >&2
  exit 1
fi

STAGE="$(mktemp -d)"
trap 'rm -rf "${STAGE}"' EXIT
cp docker/gb10/install-humming.sh "${STAGE}/install-humming.sh"

cat > "${STAGE}/Dockerfile" <<'EOF'
ARG BASE=vllm-gb10:v0.28.0-dsv4-spark
FROM ${BASE}
USER root
# Packed host venvs have no pip; install with uv like b12x.
RUN if [ ! -x /root/.local/bin/uv ]; then \
      curl -LsSf https://astral.sh/uv/install.sh | sh; \
    fi
ENV PATH="/root/.local/bin:${PATH}"
COPY install-humming.sh /tmp/install-humming.sh
RUN chmod +x /tmp/install-humming.sh \
    && /tmp/install-humming.sh /opt/venv/bin/python \
    && rm -f /tmp/install-humming.sh
EOF

echo "Injecting humming-kernels[cu13]==0.1.12 into ${IMAGE} (no vLLM compile)"
docker build \
  --platform linux/arm64 \
  -f "${STAGE}/Dockerfile" \
  --build-arg BASE="${IMAGE}" \
  -t "${IMAGE}" \
  "${STAGE}"
echo "Re-tagged ${IMAGE} with humming-kernels[cu13]==0.1.12."
echo "Sync to the worker. Leftover SM121 W8A8 uses humming before Triton."
