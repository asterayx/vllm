# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm B12X JIT kernels used by a loaded model."""

from collections import Counter
from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.mxfp4.b12x import (
    warmup_b12x_mxfp4_linear,
)
from vllm.model_executor.kernels.linear.mxfp8.b12x import (
    warmup_b12x_mxfp8_linear,
)
from vllm.model_executor.kernels.linear.nvfp4.b12x import (
    warmup_b12x_nvfp4_linear,
)
from vllm.model_executor.kernels.linear.scaled_mm.b12x_block import (
    warmup_b12x_block_fp8_linear,
)
from vllm.model_executor.kernels.linear.scaled_mm.b12x_tensor import (
    warmup_b12x_tensor_fp8_linear,
)
from vllm.platforms import current_platform
from vllm.utils.b12x import B12xWarmupUnit, b12x_warmup_token_counts

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


def _collect_warmup_units(
    model: torch.nn.Module,
    token_counts: tuple[int, ...],
    output_dtype: torch.dtype,
) -> Iterable[B12xWarmupUnit]:
    units: dict[object, B12xWarmupUnit] = {}
    for layer in model.modules():
        provider = getattr(layer, "b12x_warmup_provider", None)
        get_unit = getattr(provider, "get_b12x_warmup_unit", None)
        if not callable(get_unit):
            continue
        unit = get_unit(layer, token_counts, output_dtype)
        assert isinstance(unit, B12xWarmupUnit)
        units.setdefault(unit.key, unit)
    return units.values()


def _compile_warmup_units(
    units: Iterable[B12xWarmupUnit],
) -> Counter[str]:
    warmed: Counter[str] = Counter()
    with torch.inference_mode():
        for unit in units:
            unit.compile()
            warmed[unit.name] += 1
        if warmed:
            torch.accelerator.synchronize()
    return warmed


def b12x_warmup(worker: "Worker", cudagraph_capture_sizes: list[int]) -> None:
    if not current_platform.is_cuda():
        return
    if not current_platform.is_device_capability_family(120):
        return

    model = worker.get_model()
    max_num_batched_tokens = worker.scheduler_config.max_num_batched_tokens
    compile_sizes = ()
    compilation_config = getattr(
        getattr(worker, "vllm_config", None), "compilation_config", None
    )
    if compilation_config is not None:
        compile_sizes = compilation_config.compile_sizes or ()
    max_tokens = max_num_batched_tokens
    max_num_scheduled_tokens = getattr(
        worker.scheduler_config, "max_num_scheduled_tokens", None
    )
    if max_num_scheduled_tokens is not None:
        max_tokens = max(max_tokens, max_num_scheduled_tokens)
    serving_sizes = [
        max_num_batched_tokens,
        *cudagraph_capture_sizes,
        *(size for size in compile_sizes if isinstance(size, int)),
    ]
    output_dtype = getattr(
        getattr(worker, "model_config", None),
        "dtype",
        torch.bfloat16,
    )
    if output_dtype not in (torch.bfloat16, torch.float16):
        output_dtype = torch.bfloat16

    warmup_kwargs = {
        "max_tokens": max_num_batched_tokens,
        "cudagraph_capture_sizes": cudagraph_capture_sizes,
        "output_dtype": output_dtype,
    }
    providers = (
        ("block-FP8", warmup_b12x_block_fp8_linear),
        ("MXFP8", warmup_b12x_mxfp8_linear),
        ("tensor FP8", warmup_b12x_tensor_fp8_linear),
        ("MXFP4", warmup_b12x_mxfp4_linear),
        ("NVFP4", warmup_b12x_nvfp4_linear),
    )
    for name, warmup in providers:
        warmed = warmup(model, **warmup_kwargs)
        if warmed:
            logger.info_once(
                "Warmed up %d B12X %s linear GEMM signatures.",
                warmed,
                name,
            )

    token_counts = b12x_warmup_token_counts(
        max_tokens=max_tokens,
        cudagraph_capture_sizes=serving_sizes,
    )
    units = _collect_warmup_units(model, token_counts, output_dtype)
    for name, count in _compile_warmup_units(units).items():
        logger.info_once(
            "Warmed up %d b12x %s kernel signature(s).",
            count,
            name,
        )
