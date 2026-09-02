#!/usr/bin/env bash
# Vision-Exp from the baked /opt/vllm (no host source mount).
# Same k=3 / next_n=4 as run-vision.sh.
#
# Head:
#   NODE_RANK=0 ./docker/gb10/run-vision-image.sh
# Worker:
#   NODE_RANK=1 VLLM_HOST_IP=192.168.100.11 HEADLESS=--headless \
#     ./docker/gb10/run-vision-image.sh
set -euo pipefail
export MOUNT_VLLM_SRC=0
exec "$(cd "$(dirname "$0")" && pwd)/run-vision.sh"
