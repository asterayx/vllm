#!/usr/bin/env bash
# Open a CUDA coredump with cuda-gdb from the GB10 image.
# cuda-gdb lives in the image at /usr/local/cuda/bin/cuda-gdb — not on the
# Spark host PATH. The serve container is already dead after IMA; do not
# docker exec it. Use a fresh one-shot container.
#
#   ./docker/gb10/gdb-dump.sh
#   ./docker/gb10/gdb-dump.sh ~/cuda-dumps/cuda_coredump_aitopard-d6d3.196.1787965675
set -euo pipefail

IMAGE="${VLLM_GB10_IMAGE:-vllm-gb10:dspark}"
CUDA_DUMP_DIR="${CUDA_DUMP_DIR:-${HOME}/cuda-dumps}"
DUMP="${1:-}"

if [ -z "${DUMP}" ]; then
  DUMP="$(ls -t "${CUDA_DUMP_DIR}"/cuda_coredump_* 2>/dev/null | head -n 1 || true)"
fi
if [ -z "${DUMP}" ] || [ ! -f "${DUMP}" ]; then
  echo "no coredump under ${CUDA_DUMP_DIR}" >&2
  echo "usage: $0 [ /path/to/cuda_coredump_* ]" >&2
  exit 1
fi

DUMP_ABS="$(cd "$(dirname "${DUMP}")" && pwd)/$(basename "${DUMP}")"
DUMP_DIR="$(dirname "${DUMP_ABS}")"
DUMP_NAME="$(basename "${DUMP_ABS}")"

echo "cuda-gdb is /usr/local/cuda/bin/cuda-gdb inside ${IMAGE}"
echo "opening ${DUMP_ABS}"

exec docker run --rm -it --gpus all --privileged \
  -v "${DUMP_DIR}:/cuda-dumps:ro" \
  --entrypoint bash \
  "${IMAGE}" \
  -lc "export PATH=/usr/local/cuda/bin:\${PATH}
       command -v cuda-gdb
       cuda-gdb --version
       exec cuda-gdb \
         -ex 'target cudacore /cuda-dumps/${DUMP_NAME}' \
         -ex 'info cuda kernels' \
         -ex 'info cuda threads' \
         -ex bt"
