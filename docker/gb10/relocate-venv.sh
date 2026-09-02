#!/usr/bin/env bash
# Rewrite shebangs and editable install metadata after COPY into
# /opt/venv + /opt/vllm. Host venvs are typically `pip install -e`, and
# PEP 660 finders live in *.py (not just *.pth). A leftover host path
# makes `import vllm` fail even when /opt/vllm is on PYTHONPATH.
set -euo pipefail

NEW_VENV="$1"
NEW_SRC="$2"
OLD_VENV="$3"
OLD_SRC="$4"

if [[ -f "${NEW_VENV}/pyvenv.cfg" ]]; then
  sed -i \
    -e "s|^home = .*|home = /usr/bin|" \
    -e "s|^executable = .*|executable = /usr/bin/python3.12|" \
    -e "s|^command = .*|command = /usr/bin/python3.12 -m venv ${NEW_VENV}|" \
    "${NEW_VENV}/pyvenv.cfg"
fi

if [[ -n "${OLD_VENV}" ]]; then
  find "${NEW_VENV}/bin" -type f -print0 \
    | xargs -0 sed -i "s|^#!${OLD_VENV}/bin/python|#!${NEW_VENV}/bin/python|g"
fi

python3.12 - <<'PY' "${NEW_VENV}" "${NEW_SRC}" "${OLD_VENV}" "${OLD_SRC}"
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

ln -sfn "${NEW_VENV}/bin/python" /usr/local/bin/python
chmod +x "${NEW_SRC}/docker/gb10/entrypoint.sh" || true

"${NEW_VENV}/bin/python" -c "import vllm; print('relocated vllm:', vllm.__file__)"
