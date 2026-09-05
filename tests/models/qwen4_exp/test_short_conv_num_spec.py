# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect

from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)


def test_short_conv_state_shape_accepts_num_spec() -> None:
    params = inspect.signature(
        MambaStateShapeCalculator.short_conv_state_shape
    ).parameters
    assert "num_spec" in params

    no_spec = MambaStateShapeCalculator.short_conv_state_shape(
        tp_world_size=1, intermediate_size=16, conv_kernel=4, num_spec=0
    )
    with_spec = MambaStateShapeCalculator.short_conv_state_shape(
        tp_world_size=1, intermediate_size=16, conv_kernel=4, num_spec=3
    )
    if is_conv_state_dim_first():
        assert no_spec == ((16, 3),)
        assert with_spec == ((16, 6),)
    else:
        assert no_spec == ((3, 16),)
        assert with_spec == ((6, 16),)
