# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DPSK_NVTX must cost nothing when off and push a range per call when on."""

from vllm.utils import dpsk_nvtx as mod


def _fn(x):
    return x + 1


def test_disabled_is_free(monkeypatch):
    monkeypatch.setattr(mod, "DPSK_NVTX", False)
    assert mod.dpsk_nvtx("a") is mod.dpsk_nvtx("b")
    assert mod.dpsk_nvtx_fn("x")(_fn) is _fn


def test_enabled_wraps_each_call(monkeypatch):
    pushed = []

    class _Range:
        def __init__(self, name):
            pushed.append(name)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(mod, "DPSK_NVTX", True)
    monkeypatch.setattr(mod.torch.cuda.nvtx, "range", _Range)
    wrapped = mod.dpsk_nvtx_fn("dpsk:x")(_fn)
    assert wrapped(1) == 2 and wrapped(2) == 3
    assert pushed == ["dpsk:x", "dpsk:x"]
    with mod.dpsk_nvtx("dpsk:y"):
        pass
    assert pushed[-1] == "dpsk:y"
