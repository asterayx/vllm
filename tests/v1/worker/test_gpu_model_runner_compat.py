# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard 0.28.0 worker imports after the Qwen4Exp cherry-pick."""

from pathlib import Path

import torch

from vllm.v1.kv_cache_interface import CircularBufferSpec


def test_gpu_worker_imports_without_later_main_jit_registry() -> None:
    from vllm.model_executor.warmup import jit_warmup
    from vllm.v1.worker import gpu_worker
    from vllm.v1.worker.gpu import model_runner

    assert not hasattr(jit_warmup, "JitWarmupRegistry")
    source = Path(model_runner.__file__).read_text()
    assert "JitWarmupRegistry" not in source
    assert "profile_cudagraph_memory as _" not in source
    assert gpu_worker.Worker is not None
    assert model_runner.GPUModelRunner is not None


def test_circular_buffer_spec_uses_one_block() -> None:
    spec = CircularBufferSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    assert spec.max_num_blocks_per_req(None, 262144) == 1
    assert spec.prefix_cacheable is False
