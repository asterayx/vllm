#!/usr/bin/env bash
# Inject official b12x==1.2.6 into the existing vllm-gb10:dspark image.
# Does not recopy the venv and does not compile vLLM.
#
#   ./docker/gb10/pack-b12x.sh
#   ./docker/gb10/sync-image.sh roccen@<worker-connectx-ip>
#
# Installs the stock `b12x` package (setup.py extra, --moe-backend b12x).
# Not Anemll `--moe-backend flashinfer_b12x`.
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "Pack b12x on aarch64 (the Spark), not x86." >&2
  exit 1
fi

IMAGE="${VLLM_GB10_IMAGE:-vllm-gb10:dspark}"
if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "No image ${IMAGE}. Pack the venv first." >&2
  exit 1
fi

STAGE="$(mktemp -d)"
trap 'rm -rf "${STAGE}"' EXIT
cp docker/gb10/install-b12x.sh "${STAGE}/install-b12x.sh"

cat > "${STAGE}/Dockerfile" <<'EOF'
ARG BASE=vllm-gb10:dspark
FROM ${BASE}
USER root
# Packed host venvs have no pip; install with uv like the FlashInfer overlay.
RUN if [ ! -x /root/.local/bin/uv ]; then \
      curl -LsSf https://astral.sh/uv/install.sh | sh; \
    fi
ENV PATH="/root/.local/bin:${PATH}"
COPY install-b12x.sh /tmp/install-b12x.sh
RUN chmod +x /tmp/install-b12x.sh \
    && /tmp/install-b12x.sh /opt/venv/bin/python \
    && rm -f /tmp/install-b12x.sh
EOF

echo "Injecting official b12x==1.2.6 into ${IMAGE} (no vLLM compile, no venv copy)"
docker build \
  --platform linux/arm64 \
  -f "${STAGE}/Dockerfile" \
  --build-arg BASE="${IMAGE}" \
  -t "${IMAGE}" \
  "${STAGE}"
echo "Re-tagged ${IMAGE} with b12x. Official b12x==1.2.6 pins"
echo "nvidia-cutlass-dsl==4.6.2 (may replace a packed 4.7.0)."
echo "Sync to the worker, then launch with"
echo "  --linear-backend b12x --moe-backend b12x"
