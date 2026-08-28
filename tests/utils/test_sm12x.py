# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.utils.sm12x import (
    SM12X_SAFE_MIN_TOKENS,
    pad_token_rows,
    sm12x_align_tokens,
    sm12x_pad_token_rows,
)


def test_sm12x_align_tokens_pads_small_batches(monkeypatch):
    from vllm.utils import sm12x as sm12x_utils

    monkeypatch.setattr(
        sm12x_utils.current_platform,
        "is_device_capability_family",
        lambda fam: fam == 120,
    )
    assert sm12x_align_tokens(8) == SM12X_SAFE_MIN_TOKENS
    assert sm12x_align_tokens(1) == SM12X_SAFE_MIN_TOKENS
    assert sm12x_align_tokens(16) == 16
    assert sm12x_align_tokens(36) == 36
    assert sm12x_align_tokens(0) == 0


def test_sm12x_align_tokens_unchanged_off_sm12x(monkeypatch):
    from vllm.utils import sm12x as sm12x_utils

    monkeypatch.setattr(
        sm12x_utils.current_platform,
        "is_device_capability_family",
        lambda fam: False,
    )
    assert sm12x_align_tokens(8) == 8


def test_sm12x_pad_token_rows(monkeypatch):
    from vllm.utils import sm12x as sm12x_utils

    monkeypatch.setattr(
        sm12x_utils.current_platform,
        "is_device_capability_family",
        lambda fam: fam == 120,
    )
    x = torch.arange(8 * 4, dtype=torch.float32).view(8, 4)
    padded, orig = sm12x_pad_token_rows(x, what="MoE")
    assert orig == 8
    assert padded.shape == (16, 4)
    assert torch.equal(padded[:8], x)
    assert torch.count_nonzero(padded[8:]) == 0
    assert pad_token_rows(x, 8).shape == (8, 4)
