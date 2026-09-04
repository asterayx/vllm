#!/usr/bin/env bash
# Inject humming-kernels[cu13]==0.1.12 and the SM121 leftover kernel order
# into an existing GB10 image. Does not recopy the venv and does not compile.
#
# Installing the wheel alone is not enough: baked /opt/vllm still has
# Triton before Humming, so leftover W8A8 keeps hitting the GB10 json warning.
# This overlays kernels/linear/__init__.py from this tree and restamps
# docker/gb10/VERSION.
#
#   ./docker/gb10/pack-humming.sh
#   VLLM_GB10_IMAGE=vllm-gb10:v0.28.0-dsv4-spark ./docker/gb10/pack-humming.sh
#   ./docker/gb10/sync-image.sh roccen@<worker-connectx-ip>
#
# Do not add humming to --linear-backend b12x.
# Bind-mount serve (run-vision.sh) already sees host source; still run this
# so the image venv has the humming package.
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
cp docker/gb10/VERSION "${STAGE}/VERSION"
cp docker/gb10/stamp-version.py "${STAGE}/stamp-version.py"
cp vllm/model_executor/kernels/linear/__init__.py "${STAGE}/linear_init.py"

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
# Official-image leftover order lives in baked /opt/vllm, not the wheel.
COPY linear_init.py /opt/vllm/vllm/model_executor/kernels/linear/__init__.py
COPY VERSION /opt/vllm/docker/gb10/VERSION
COPY stamp-version.py /tmp/stamp-version.py
ARG VLLM_SPARK_VERSION
ENV VLLM_SPARK_VERSION=${VLLM_SPARK_VERSION} \
    VLLM_VERSION_OVERRIDE=${VLLM_SPARK_VERSION} \
    VLLM_GB10_VERSION=${VLLM_SPARK_VERSION}
LABEL org.opencontainers.image.version="${VLLM_SPARK_VERSION}"
RUN /opt/venv/bin/python /tmp/stamp-version.py \
        "${VLLM_SPARK_VERSION}" --root /opt/vllm \
    && rm -f /tmp/stamp-version.py \
    && find /opt/vllm/vllm/model_executor/kernels/linear -name '*.pyc' -delete \
    && find /opt/vllm/vllm/model_executor/kernels/linear -name '__pycache__' -type d -exec rm -rf {} + \
    || true
EOF

echo "Injecting humming-kernels + SM121 leftover order into ${IMAGE} (${VLLM_SPARK_VERSION})"
docker build \
  --platform linux/arm64 \
  -f "${STAGE}/Dockerfile" \
  --build-arg BASE="${IMAGE}" \
  --build-arg VLLM_SPARK_VERSION="${VLLM_SPARK_VERSION}" \
  -t "${IMAGE}" \
  -t "${VLLM_GB10_IMAGE_RELEASE}" \
  "${STAGE}"
echo "Re-tagged ${IMAGE} and ${VLLM_GB10_IMAGE_RELEASE}."
echo "Sync to the worker. Expect: Selected HummingFP8ScaledMMLinearKernel"
echo "and no NVIDIA_GB10 W8A8 Block FP8 config warnings."
