#!/usr/bin/env bash
# Overlay collected host verbs files onto the image. Used by Dockerfiles.
set -euo pipefail
src="${1:-/tmp/ibverbs}"

mkdir -p /etc/libibverbs.d /usr/lib/aarch64-linux-gnu/libibverbs /etc/rdma
if [[ -d "${src}/etc/libibverbs.d" ]]; then
  cp -a "${src}/etc/libibverbs.d/." /etc/libibverbs.d/
fi
if [[ -d "${src}/rdma" ]] && [[ -n "$(ls -A "${src}/rdma" 2>/dev/null || true)" ]]; then
  cp -a "${src}/rdma/." /etc/rdma/
fi
if [[ -d "${src}/libibverbs" ]]; then
  cp -a "${src}/libibverbs/." /usr/lib/aarch64-linux-gnu/libibverbs/
fi
if [[ -d "${src}/lib" ]]; then
  cp -a "${src}/lib/." /usr/lib/aarch64-linux-gnu/
fi
printf '%s\n' /usr/lib/aarch64-linux-gnu > /etc/ld.so.conf.d/ibverbs-host.conf
ldconfig
rm -rf "${src}"

if [[ ! -d /etc/libibverbs.d ]] || [[ -z "$(ls -A /etc/libibverbs.d)" ]]; then
  echo "ibverbs overlay left /etc/libibverbs.d empty" >&2
  exit 1
fi
echo "Installed ibverbs providers:"
ls -l /etc/libibverbs.d
