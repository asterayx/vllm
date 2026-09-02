#!/usr/bin/env python3
"""Fail a GB10 image build if CUDA torch or vLLM C extensions are missing."""

from pathlib import Path

import torch

assert torch.version.cuda, (
    f"CPU torch installed ({torch.__version__}); need torch==2.13.0+cu130"
)
print("torch", torch.__version__, "cuda", torch.version.cuda, flush=True)

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
