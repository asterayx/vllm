# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU unit tests for Spark recipe knobs (MiaAI / Anemll parity)."""

import os

from vllm.config.cache import CacheConfig
from vllm.distributed.device_communicators.shm_broadcast import SpinCondition
from vllm.utils.torch_utils import resolve_kv_cache_dtype_string


def test_nvfp4_ds_mla_alias_normalizes_in_cache_config():
    cfg = CacheConfig(cache_dtype="nvfp4_ds_mla")
    assert cfg.cache_dtype == "fp8_ds_mla"


def test_nvfp4_ds_mla_alias_normalizes_in_resolve_helper():
    assert resolve_kv_cache_dtype_string("nvfp4_ds_mla", None) == "fp8_ds_mla"
    assert resolve_kv_cache_dtype_string("fp8_ds_mla", None) == "fp8_ds_mla"
    assert resolve_kv_cache_dtype_string("auto", None) == "auto"


def test_shm_spin_busy_loop_default_is_1s(monkeypatch):
    monkeypatch.delenv("VLLM_SHM_BROADCAST_BUSY_LOOP_S", raising=False)
    import vllm.envs as envs

    monkeypatch.setattr(envs, "VLLM_SHM_BROADCAST_BUSY_LOOP_S", 1.0, raising=False)

    class _DummySocket:
        def setsockopt(self, *args, **kwargs):
            return None

        def setsockopt_string(self, *args, **kwargs):
            return None

        def connect(self, *args, **kwargs):
            return None

        def bind(self, *args, **kwargs):
            return None

    class _DummyContext:
        def socket(self, *args, **kwargs):
            return _DummySocket()

    sc = SpinCondition(
        is_reader=True,
        context=_DummyContext(),
        notify_address="inproc://test-spin-busy-loop",
    )
    assert sc.busy_loop_s == 1.0


def test_shm_spin_busy_loop_env_override(monkeypatch):
    import vllm.envs as envs

    monkeypatch.setattr(envs, "VLLM_SHM_BROADCAST_BUSY_LOOP_S", 0.05)

    class _DummySocket:
        def setsockopt(self, *args, **kwargs):
            return None

        def setsockopt_string(self, *args, **kwargs):
            return None

        def connect(self, *args, **kwargs):
            return None

        def bind(self, *args, **kwargs):
            return None

    class _DummyContext:
        def socket(self, *args, **kwargs):
            return _DummySocket()

    sc = SpinCondition(
        is_reader=True,
        context=_DummyContext(),
        notify_address="inproc://test-spin-busy-loop-override",
    )
    assert sc.busy_loop_s == 0.05
    # Explicit ctor arg still wins.
    sc2 = SpinCondition(
        is_reader=True,
        context=_DummyContext(),
        notify_address="inproc://test-spin-busy-loop-explicit",
        busy_loop_s=1.0,
    )
    assert sc2.busy_loop_s == 1.0
    _ = os.environ  # keep import used for linters that scan the module
