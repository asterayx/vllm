#!/usr/bin/env bash
# Release: serve from the baked /opt/vllm. Does not bind-mount host source.
# Use this to verify the image, or to publish. git checkout on the host
# will not change what the container runs.
#
# Head:
#   NODE_RANK=0 ./docker/gb10/run-image.sh
# Worker:
#   NODE_RANK=1 VLLM_HOST_IP=192.168.100.11 HEADLESS=--headless \
#     ./docker/gb10/run-image.sh
#
# Default image is still vllm-gb10:dspark. Override after build.sh:
#   VLLM_GB10_IMAGE=vllm-gb10:v0.28.0-dsv4-spark ./docker/gb10/run-image.sh
set -euo pipefail
export MOUNT_VLLM_SRC=0
exec "$(cd "$(dirname "$0")" && pwd)/run.sh"
