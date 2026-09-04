#!/usr/bin/env bash
# Rewrite shebangs and editable install metadata after COPY into
# /opt/venv + /opt/vllm. Host venvs are typically `uv venv` + editable
# install. uv points bin/python at a host-managed CPython that is not
# copied into the image — retarget those links at IMAGE_PYTHON
# (/usr/bin/python3.12). PEP 660 finders live in *.py (not just *.pth);
# a leftover host path makes `import vllm` fail even when /opt/vllm is
# on PYTHONPATH.
set -euo pipefail

NEW_VENV="$1"
NEW_SRC="$2"
OLD_VENV="$3"
OLD_SRC="$4"

IMAGE_PYTHON="${IMAGE_PYTHON:-/usr/bin/python3.12}"
if [[ ! -x "${IMAGE_PYTHON}" ]]; then
  echo "image python missing: ${IMAGE_PYTHON}" >&2
  exit 1
fi

if [[ -f "${NEW_VENV}/pyvenv.cfg" ]]; then
  sed -i \
    -e "s|^home = .*|home = /usr/bin|" \
    -e "s|^executable = .*|executable = ${IMAGE_PYTHON}|" \
    -e "s|^command = .*|command = ${IMAGE_PYTHON} -m venv ${NEW_VENV}|" \
    "${NEW_VENV}/pyvenv.cfg"
fi

# uv venvs symlink bin/python at a host-managed CPython
# (~/.local/share/uv/python/...). COPY keeps the symlink; the target is
# not in the image, so /opt/venv/bin/python is dangling (exit 127).
mkdir -p "${NEW_VENV}/bin"
ln -sfn "${IMAGE_PYTHON}" "${NEW_VENV}/bin/python3.12"
ln -sfn python3.12 "${NEW_VENV}/bin/python3"
ln -sfn python3 "${NEW_VENV}/bin/python"

REWRITE_PY="${REWRITE_PY:-}"
if [[ -z "${REWRITE_PY}" ]]; then
  if command -v python3.12 >/dev/null 2>&1; then
    REWRITE_PY=python3.12
  else
    REWRITE_PY="${IMAGE_PYTHON}"
  fi
fi
"${REWRITE_PY}" - <<'PY' "${NEW_VENV}" "${NEW_SRC}" "${OLD_VENV}" "${OLD_SRC}"
import pathlib
import sys

new_venv, new_src, old_venv, old_src = sys.argv[1:]
new_venv = new_venv.rstrip("/")
new_src = new_src.rstrip("/")
old_venv = old_venv.rstrip("/")
old_src = old_src.rstrip("/")

replacements = []
if old_src:
    replacements.append((old_src, new_src))
if old_venv:
    replacements.append((old_venv, new_venv))

suffixes = {".pth", ".egg-link", ".json", ".txt", ".py"}
root = pathlib.Path(new_venv)
if replacements:
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in suffixes:
            continue
        # Only rewrite Python files that belong to editable metadata.
        if path.suffix == ".py" and not path.name.startswith("__editable__"):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        updated = text
        for old, new in replacements:
            updated = updated.replace(old, new)
        if updated != text:
            path.write_text(updated, encoding="utf-8")

# Host scripts may shebang the uv-managed interpreter, not OLD_VENV.
new_shebang = f"#!{new_venv}/bin/python"
bin_dir = pathlib.Path(new_venv) / "bin"
if bin_dir.is_dir():
    for path in bin_dir.iterdir():
        if not path.is_file() or path.is_symlink():
            continue
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        if not raw.startswith(b"#!") or b"python" not in raw.split(b"\n", 1)[0]:
            continue
        nl = raw.find(b"\n")
        rest = raw[nl:] if nl >= 0 else b""
        updated = new_shebang.encode() + rest
        if updated != raw:
            path.write_bytes(updated)

# Always put the packed tree on sys.path, then drop vLLM's host
# PEP 660 finders. A finder that still maps to /home/... wins over
# PYTHONPATH and raises ModuleNotFoundError.
for site in pathlib.Path(new_venv).glob("lib/python*/site-packages"):
    (site / "_vllm_relocated.pth").write_text(new_src + "\n", encoding="utf-8")
    for leftover in site.glob("__editable__.vllm*"):
        leftover.unlink(missing_ok=True)
    for leftover in site.glob("__editable___vllm*"):
        leftover.unlink(missing_ok=True)
PY

if [[ -w /usr/local/bin ]]; then
  ln -sfn "${NEW_VENV}/bin/python" /usr/local/bin/python
fi
chmod +x "${NEW_SRC}/docker/gb10/entrypoint.sh" || true

if [[ ! -e "${NEW_VENV}/bin/python" ]]; then
  echo "missing ${NEW_VENV}/bin/python after relocate" >&2
  ls -l "${NEW_VENV}/bin" >&2 || true
  exit 1
fi
"${NEW_VENV}/bin/python" -c "import vllm; print('relocated vllm:', vllm.__file__)"
