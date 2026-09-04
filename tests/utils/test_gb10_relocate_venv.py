# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Host uv venvs symlink bin/python at a CPython that is not in the image."""

import os
import subprocess
import sys
from pathlib import Path

_GB10 = Path(__file__).resolve().parents[2] / "docker" / "gb10"
_RELOCATE = _GB10 / "relocate-venv.sh"


def _image_python() -> str:
    for candidate in ("/usr/bin/python3.12", "/usr/bin/python3"):
        if os.access(candidate, os.X_OK):
            return candidate
    return sys.executable


def _run_relocate(venv: Path, src: Path, old_venv: str, old_src: str) -> None:
    env = os.environ.copy()
    env["IMAGE_PYTHON"] = _image_python()
    env["REWRITE_PY"] = sys.executable
    subprocess.run(
        ["bash", str(_RELOCATE), str(venv), str(src), old_venv, old_src],
        check=True,
        env=env,
        cwd=venv.parent,
    )


def test_relocate_retargets_dangling_uv_python(tmp_path: Path):
    old_venv = "/home/roccen/.venvs/vllm028"
    old_src = "/home/roccen/src/vllm"
    uv_py = (
        tmp_path
        / "home"
        / "roccen"
        / ".local"
        / "share"
        / "uv"
        / "python"
        / "cpython-3.12.12-linux-aarch64-gnu"
        / "bin"
        / "python3.12"
    )
    venv = tmp_path / "opt" / "venv"
    src = tmp_path / "opt" / "vllm"
    (venv / "bin").mkdir(parents=True)
    (src / "vllm").mkdir(parents=True)
    (src / "vllm" / "__init__.py").write_text("__version__ = 'test'\n")
    (src / "docker" / "gb10").mkdir(parents=True)
    (src / "docker" / "gb10" / "entrypoint.sh").write_text("#!/bin/sh\n")
    (venv / "pyvenv.cfg").write_text(
        "home = /home/roccen/.local/share/uv/python/cpython-3.12.12-linux-aarch64-gnu/bin\n"
        f"executable = {uv_py}\n"
        f"command = {uv_py} -m venv {old_venv}\n"
    )
    (venv / "bin" / "python").symlink_to(uv_py)
    (venv / "bin" / "python3").symlink_to(uv_py)
    (venv / "bin" / "python3.12").symlink_to(uv_py)
    pip = venv / "bin" / "pip"
    pip.write_text(f"#!{uv_py}\nprint('pip')\n")
    pip.chmod(0o755)
    site = venv / "lib" / "python3.12" / "site-packages"
    site.mkdir(parents=True)
    (site / "__editable__.vllm-0.28.0.pth").write_text(f"{old_src}\n")
    (site / "__editable___vllm_finder.py").write_text(
        f"PATHS = [{old_src!r}]\n"
    )

    assert not (venv / "bin" / "python").exists()
    _run_relocate(venv, src, old_venv, old_src)

    image_py = Path(_image_python()).resolve()
    assert (venv / "bin" / "python").exists()
    assert (venv / "bin" / "python").resolve() == image_py
    cfg = (venv / "pyvenv.cfg").read_text()
    assert f"executable = {_image_python()}" in cfg
    assert pip.read_text().startswith(f"#!{venv}/bin/python\n")
    assert not list(site.glob("__editable__.vllm*"))
    assert (site / "_vllm_relocated.pth").read_text() == f"{src}\n"


def test_relocate_creates_python_when_copy_dropped_dangling_link(tmp_path: Path):
    venv = tmp_path / "opt" / "venv"
    src = tmp_path / "opt" / "vllm"
    (venv / "bin").mkdir(parents=True)
    (src / "vllm").mkdir(parents=True)
    (src / "vllm" / "__init__.py").write_text("__version__ = 'test'\n")
    (src / "docker" / "gb10").mkdir(parents=True)
    (src / "docker" / "gb10" / "entrypoint.sh").write_text("#!/bin/sh\n")
    (venv / "lib" / "python3.12" / "site-packages").mkdir(parents=True)

    _run_relocate(venv, src, "/old/venv", "/old/src")
    assert (venv / "bin" / "python").exists()
    assert (venv / "bin" / "python").resolve() == Path(_image_python()).resolve()
