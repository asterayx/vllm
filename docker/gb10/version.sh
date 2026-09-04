#!/usr/bin/env bash
# Spark GB10 image version. Source from other scripts, or run to print tags.
# Single source of truth: docker/gb10/VERSION (PEP 440, e.g. 0.28.0+dsv4.spark.1).
# Bump the trailing integer when publishing a new official image.
#
#   ./docker/gb10/version.sh
#   source docker/gb10/version.sh
set -euo pipefail

_gb10_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ver_file="${_gb10_dir}/VERSION"
if [[ -f "${_ver_file}" ]]; then
  VLLM_SPARK_VERSION="$(sed -e 's/#.*//' -e '/^[[:space:]]*$/d' "${_ver_file}" | head -1 | tr -d '[:space:]')"
else
  # Dockerfile.gb10 dockerignores VERSION during the compile RUN.
  VLLM_SPARK_VERSION="${VLLM_SPARK_VERSION:-0.28.0+dsv4.spark.2}"
  echo "WARN: ${_ver_file} missing; VLLM_SPARK_VERSION=${VLLM_SPARK_VERSION}" >&2
fi
if [[ ! "${VLLM_SPARK_VERSION}" =~ ^([0-9]+\.[0-9]+\.[0-9]+)\+dsv4\.spark\.([0-9]+)$ ]]; then
  echo "VERSION must look like 0.28.0+dsv4.spark.1, got: ${VLLM_SPARK_VERSION}" >&2
  return 1 2>/dev/null || exit 1
fi
VLLM_BASE_VERSION="${BASH_REMATCH[1]}"
SPARK_RELEASE="${BASH_REMATCH[2]}"
VLLM_GB10_IMAGE_NAME="${VLLM_GB10_IMAGE_NAME:-vllm-gb10}"
VLLM_GB10_IMAGE_FAMILY="${VLLM_GB10_IMAGE_FAMILY:-${VLLM_GB10_IMAGE_NAME}:v${VLLM_BASE_VERSION}-dsv4-spark}"
VLLM_GB10_IMAGE_RELEASE="${VLLM_GB10_IMAGE_RELEASE:-${VLLM_GB10_IMAGE_FAMILY}.${SPARK_RELEASE}}"
# Moving latest of this line. Override with VLLM_GB10_IMAGE to pin the release tag.
VLLM_GB10_IMAGE="${VLLM_GB10_IMAGE:-${VLLM_GB10_IMAGE_FAMILY}}"
export VLLM_SPARK_VERSION VLLM_BASE_VERSION SPARK_RELEASE
export VLLM_GB10_IMAGE_NAME VLLM_GB10_IMAGE_FAMILY VLLM_GB10_IMAGE_RELEASE
export VLLM_GB10_IMAGE

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  printf 'VLLM_SPARK_VERSION=%s\n' "${VLLM_SPARK_VERSION}"
  printf 'VLLM_GB10_IMAGE_FAMILY=%s\n' "${VLLM_GB10_IMAGE_FAMILY}"
  printf 'VLLM_GB10_IMAGE_RELEASE=%s\n' "${VLLM_GB10_IMAGE_RELEASE}"
  printf 'VLLM_GB10_IMAGE=%s\n' "${VLLM_GB10_IMAGE}"
fi
