#!/usr/bin/env bash
# Copy this Spark's verbs userspace into $1 so the image matches the host driver.
set -euo pipefail

dest="${1:?usage: collect-ibverbs.sh <dest-dir>}"
mkdir -p "${dest}/etc/libibverbs.d" "${dest}/lib" "${dest}/libibverbs" "${dest}/rdma"

if [[ ! -d /etc/libibverbs.d ]] || [[ -z "$(ls -A /etc/libibverbs.d 2>/dev/null || true)" ]]; then
  echo "Host has no /etc/libibverbs.d. Install rdma-core / ibverbs-providers on the Spark first." >&2
  exit 1
fi
rsync -a /etc/libibverbs.d/ "${dest}/etc/libibverbs.d/"

if [[ -d /etc/rdma ]]; then
  rsync -a /etc/rdma/ "${dest}/rdma/"
fi

if [[ -d /usr/lib/aarch64-linux-gnu/libibverbs ]]; then
  rsync -a /usr/lib/aarch64-linux-gnu/libibverbs/ "${dest}/libibverbs/"
fi

shopt -s nullglob
for f in \
  /usr/lib/aarch64-linux-gnu/libibverbs.so* \
  /usr/lib/aarch64-linux-gnu/libmlx5.so* \
  /usr/lib/aarch64-linux-gnu/libmlx4.so* \
  /usr/lib/aarch64-linux-gnu/librdmacm.so* \
  /usr/lib/aarch64-linux-gnu/libibumad.so* \
  /usr/lib/aarch64-linux-gnu/libefa.so* \
  /usr/lib/aarch64-linux-gnu/libmana.so* \
  /usr/lib/aarch64-linux-gnu/libhns.so* \
  /lib/aarch64-linux-gnu/libibverbs.so* \
  /lib/aarch64-linux-gnu/libmlx5.so*
do
  cp -a "$f" "${dest}/lib/"
done

if [[ ! -d "${dest}/libibverbs" ]] || [[ -z "$(ls -A "${dest}/libibverbs" 2>/dev/null || true)" ]]; then
  if [[ -z "$(ls -A "${dest}/lib" 2>/dev/null || true)" ]]; then
    echo "Did not find host mlx5/ibverbs libraries." >&2
    exit 1
  fi
fi

echo "Collected host ibverbs into ${dest}"
ls -l "${dest}/etc/libibverbs.d" || true
ls "${dest}/libibverbs" 2>/dev/null | head || true
ls "${dest}/lib" 2>/dev/null | head || true
