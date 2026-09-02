#!/usr/bin/env python3
"""Fail a GB10 image build if vLLM was not actually installed/compiled."""

from __future__ import annotations

import importlib
import importlib.metadata as metadata
from pathlib import Path

import torch

assert torch.version.cuda, (
    f"CPU torch installed ({torch.__version__}); need torch==2.13.0+cu130"
)
print("torch", torch.__version__, "cuda", torch.version.cuda, flush=True)

try:
    ver = metadata.version("vllm")
except metadata.PackageNotFoundError as exc:
    raise SystemExit(
        "vllm is not installed in the venv (no package metadata). "
        "uv pip install -e . did not install the project."
    ) from exc
print("vllm dist", ver, flush=True)

hits: list[Path] = []
roots = [Path("/opt/vllm/vllm"), *Path("/opt/vllm").glob("build*")]
roots.extend(Path("/opt/venv").glob("lib/python*/site-packages"))
roots.extend(Path("/opt/venv").glob("lib/python*/site-packages/vllm"))
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
