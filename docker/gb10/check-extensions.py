#!/usr/bin/env python3
"""Fail if CUDA torch, vLLM metadata, or _C_stable_libtorch is missing."""

from __future__ import annotations

import importlib
import importlib.metadata as metadata
import os
import sys
from pathlib import Path

import torch

_SCRIPT = Path(__file__).resolve()
_REPO = Path(os.environ.get("VLLM_ROOT", _SCRIPT.parents[2])).resolve()
_VENV = Path(os.environ.get("VIRTUAL_ENV", sys.prefix)).resolve()


assert torch.version.cuda, (
    f"CPU torch installed ({torch.__version__}); need torch==2.13.0+cu130"
)
print("torch", torch.__version__, "cuda", torch.version.cuda, flush=True)

try:
    ver = metadata.version("vllm")
except metadata.PackageNotFoundError as exc:
    sites = list(_VENV.glob("lib/python*/site-packages"))
    for site in sites:
        print("site-packages", site, flush=True)
        print(sorted(p.name for p in site.glob("*vllm*")), flush=True)
    raise SystemExit(
        "vllm is not installed in the venv (no package metadata)."
    ) from exc
print("vllm dist", ver, flush=True)

hits: list[Path] = []
roots = [_REPO / "vllm", *_REPO.glob("build*")]
roots.extend(_VENV.glob("lib/python*/site-packages"))
roots.extend(_VENV.glob("lib/python*/site-packages/vllm"))
for root in roots:
    if not root.is_dir():
        continue
    hits.extend(root.glob("*_C_stable_libtorch*.so"))
    hits.extend(root.glob("**/*_C_stable_libtorch*.so"))
uniq = list(dict.fromkeys(p.resolve() for p in hits if p.is_file()))
print("extensions", uniq, flush=True)
assert uniq, "vllm._C_stable_libtorch was not compiled"

mod = importlib.import_module("vllm._C_stable_libtorch")
print("imported", getattr(mod, "__file__", mod), flush=True)
