#!/usr/bin/env bash
# Rewrite shebangs and editable .pth paths after COPY into /opt/venv + /opt/vllm.
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
replacements = []
if old_src:
    replacements.append((old_src.rstrip("/"), new_src.rstrip("/")))
if old_venv:
    replacements.append((old_venv.rstrip("/"), new_venv.rstrip("/")))
if not replacements:
    raise SystemExit(0)

suffixes = {".pth", ".egg-link", ".json", ".txt"}
root = pathlib.Path(new_venv)
for path in root.rglob("*"):
    if not path.is_file() or path.suffix not in suffixes:
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
PY

ln -sfn "${NEW_VENV}/bin/python" /usr/local/bin/python
chmod +x "${NEW_SRC}/docker/gb10/entrypoint.sh" || true
