#!/usr/bin/env bash
# Inject host ibverbs into the existing vllm-gb10:dspark image.
# Does not recopy the venv and does not compile vLLM.
#
#   ./docker/gb10/pack-ibverbs.sh
#   ./docker/gb10/sync-image.sh roccen@<worker-connectx-ip>
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "Pack ibverbs on aarch64 (the Spark), not x86." >&2
  exit 1
fi

IMAGE="${VLLM_GB10_IMAGE:-vllm-gb10:dspark}"
if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "No image ${IMAGE}. Pack the venv first." >&2
  exit 1
fi

STAGE="$(mktemp -d)"
trap 'rm -rf "${STAGE}"' EXIT
./docker/gb10/collect-ibverbs.sh "${STAGE}/ibverbs"
cp docker/gb10/install-ibverbs.sh "${STAGE}/install-ibverbs.sh"

cat > "${STAGE}/Dockerfile" <<'EOF'
ARG BASE=vllm-gb10:dspark
FROM ${BASE}
USER root
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        libibverbs1 \
        ibverbs-providers \
        ibverbs-utils \
        libmlx5-1 \
        librdmacm1 \
        rdma-core \
    && rm -rf /var/lib/apt/lists/*
COPY ibverbs /tmp/ibverbs
COPY install-ibverbs.sh /tmp/install-ibverbs.sh
RUN chmod +x /tmp/install-ibverbs.sh && /tmp/install-ibverbs.sh /tmp/ibverbs \
    && rm -f /tmp/install-ibverbs.sh
EOF

echo "Injecting host ibverbs into ${IMAGE} (no vLLM compile, no venv copy)"
docker build \
  --platform linux/arm64 \
  -f "${STAGE}/Dockerfile" \
  --build-arg BASE="${IMAGE}" \
  -t "${IMAGE}" \
  "${STAGE}"
echo "Re-tagged ${IMAGE} with ibverbs. Sync to the worker, then launch with NCCL_NET=IB."
