#!/usr/bin/env bash
# Resolve-only checks for docker/gb10/run-qwen38.sh weight mounting.
set -euo pipefail

SCRIPT="$(cd "$(dirname "$0")/../.." && pwd)/docker/gb10/run-qwen38.sh"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT

out="$(QWEN38_RESOLVE_ONLY=1 MODEL_HOST="${tmp}/missing" "${SCRIPT}")"
printf '%s\n' "${out}" | grep -q 'serve=RadixArk/Qwen3.8-Flash-Next-NVFP4'
printf '%s\n' "${out}" | grep -q 'offline=0'
printf '%s\n' "${out}" | grep -q 'mount=$'

mkdir -p "${tmp}/empty"
out="$(QWEN38_RESOLVE_ONLY=1 MODEL_HOST="${tmp}/empty" "${SCRIPT}")"
printf '%s\n' "${out}" | grep -q 'serve=RadixArk/Qwen3.8-Flash-Next-NVFP4'
printf '%s\n' "${out}" | grep -q 'mount=$'

mkdir -p "${tmp}/local"
echo '{"architectures":["Qwen4ExpForConditionalGeneration"]}' >"${tmp}/local/config.json"
out="$(QWEN38_RESOLVE_ONLY=1 MODEL_HOST="${tmp}/local" MODEL_PATH=/models/qwen "${SCRIPT}")"
printf '%s\n' "${out}" | grep -q 'serve=/models/qwen'
printf '%s\n' "${out}" | grep -q 'offline=1'
printf '%s\n' "${out}" | grep -q 'mount=-v'

echo "ok"
