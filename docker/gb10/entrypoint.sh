#!/bin/bash
set -euo pipefail

export PATH="/opt/venv/bin:/usr/local/cuda/bin:${PATH:-}"
export VIRTUAL_ENV="${VIRTUAL_ENV:-/opt/venv}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}"

# Packed host venvs leave PEP 660 finders pointing at /home/.... Those
# win over PYTHONPATH. build.sh editables already point at /opt/vllm —
# do not strip them in image mode: the finder is what exposes cmake .so.
if [ "${MOUNT_VLLM_SRC:-0}" != "0" ]; then
  export PYTHONPATH="/opt/vllm${PYTHONPATH:+:${PYTHONPATH}}"
  for site in /opt/venv/lib/python*/site-packages; do
    [[ -d "${site}" ]] || continue
    rm -f "${site}"/__editable__.vllm* "${site}"/__editable___vllm*
    printf '%s\n' /opt/vllm > "${site}/_vllm_relocated.pth"
  done
else
  _pp=""
  _save_ifs="$IFS"
  IFS=:
  for _p in ${PYTHONPATH:-}; do
    [ -z "$_p" ] && continue
    [ "$_p" = "/opt/vllm" ] && continue
    _pp="${_pp:+$_pp:}$_p"
  done
  IFS="$_save_ifs"
  if [ -n "$_pp" ]; then
    export PYTHONPATH="$_pp"
  else
    unset PYTHONPATH
  fi
fi

# uv editable metadata is invisible when launching via PYTHONPATH=/opt/vllm.
# Write a stub dist-info so importlib.metadata.version("vllm") succeeds.
# Also expose cmake extensions that live under build/ rather than vllm/.
/opt/venv/bin/python - <<'PY'
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

pkg = Path("/opt/vllm/vllm")
names = (
    "_C_stable_libtorch",
    "_moe_C_stable_libtorch",
    "_C",
    "_moe_C",
)
search_roots = [pkg]
search_roots.extend(Path("/opt/vllm").glob("build*"))
search_roots.extend(Path("/opt/vllm").glob("_skbuild*"))
search_roots.extend(Path("/opt/venv").glob("lib/python*/site-packages"))
search_roots.extend(Path("/opt/venv").glob("lib/python*/site-packages/vllm"))
for name in names:
    already = sorted(pkg.glob(f"{name}*.so")) if pkg.is_dir() else []
    if already:
        print("have", already[0], flush=True)
        continue
    found = []
    for root in search_roots:
        if not root.is_dir():
            continue
        found.extend(p for p in root.glob(f"{name}*.so") if p.is_file())
        found.extend(p for p in root.glob(f"**/{name}*.so") if p.is_file())
    uniq, seen = [], set()
    for p in found:
        rp = str(p.resolve())
        if rp in seen:
            continue
        seen.add(rp)
        uniq.append(p)
    print(f"candidates {name}: {uniq}", flush=True)
    if not uniq or not pkg.is_dir():
        continue
    src = uniq[0]
    dst = pkg / src.name
    try:
        if not dst.exists():
            dst.symlink_to(src)
            print("linked", dst, "->", src, flush=True)
    except OSError as e:
        print("could not link", dst, e, flush=True)
PY

if [ -d /tmp/vllm-0.28.0.dist-info ] \
   || [ -d /tmp/vllm-"${VLLM_VERSION_OVERRIDE:-0.28.0}".dist-info ]; then
  export PYTHONPATH="/tmp${PYTHONPATH:+:${PYTHONPATH}}"
fi

# WORKDIR is /opt/vllm; a bare `vllm` hits the package directory.
if [[ "${1:-}" == "vllm" ]]; then
  shift
  exec /opt/venv/bin/python -m vllm.entrypoints.cli.main "$@"
fi
exec "$@"
