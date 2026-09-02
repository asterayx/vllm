#!/bin/bash
set -euo pipefail

export PATH="/opt/venv/bin:/usr/local/cuda/bin:${PATH:-}"
export VIRTUAL_ENV="${VIRTUAL_ENV:-/opt/venv}"
export PYTHONPATH="/opt/vllm${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}"

# Packed host venvs leave PEP 660 finders pointing at /home/.... Those
# win over PYTHONPATH. build.sh editables already point at /opt/vllm —
# do not strip them or importlib.metadata.version("vllm") fails.
if [ "${MOUNT_VLLM_SRC:-0}" != "0" ]; then
  for site in /opt/venv/lib/python*/site-packages; do
    [[ -d "${site}" ]] || continue
    rm -f "${site}"/__editable__.vllm* "${site}"/__editable___vllm*
    printf '%s\n' /opt/vllm > "${site}/_vllm_relocated.pth"
  done
fi

# uv editable metadata is invisible when launching via PYTHONPATH=/opt/vllm.
# Write a stub dist-info so importlib.metadata.version("vllm") succeeds.
/opt/venv/bin/python -c '
import importlib.metadata as m
import os
from pathlib import Path
ver = os.environ.get("VLLM_VERSION_OVERRIDE", "0.28.0")
text = "Metadata-Version: 2.1\nName: vllm\nVersion: %s\n" % ver
try:
    print("vllm metadata", m.version("vllm"), flush=True)
except m.PackageNotFoundError:
    dests = [Path("/tmp") / ("vllm-%s.dist-info" % ver)]
    try:
        import site
        dests.extend(
            Path(p) / ("vllm-%s.dist-info" % ver)
            for p in site.getsitepackages()
            if p
        )
    except Exception:
        pass
    wrote = False
    for dest in dests:
        try:
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "METADATA").write_text(text)
            print("wrote stub", dest, flush=True)
            wrote = True
        except OSError as e:
            print("could not write", dest, e, flush=True)
    if not wrote:
        raise SystemExit("could not create vllm package metadata")
'
export PYTHONPATH="/tmp${PYTHONPATH:+:${PYTHONPATH}}"

# WORKDIR is /opt/vllm; a bare `vllm` hits the package directory.
if [[ "${1:-}" == "vllm" ]]; then
  shift
  exec /opt/venv/bin/python -m vllm.entrypoints.cli.main "$@"
fi
exec "$@"
