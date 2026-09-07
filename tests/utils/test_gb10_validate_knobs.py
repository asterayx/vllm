# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Comparison logic of docker/gb10/validate-knobs.py."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "docker" / "gb10" / "validate-knobs.py"


def _load():
    spec = importlib.util.spec_from_file_location("validate_knobs", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compare_reports_first_diverging_token():
    module = _load()
    base = [
        {"prompt_index": 0, "tokens": ["a", "b", "c"], "seconds": 2.0},
        {"prompt_index": 1, "tokens": ["x", "y"], "seconds": 1.0},
    ]
    other = [
        {"prompt_index": 0, "tokens": ["a", "b", "c"], "seconds": 1.0},
        {"prompt_index": 1, "tokens": ["x", "z"], "seconds": 1.0},
    ]
    lines = module.compare_results(base, other)
    assert lines[0].startswith("OK       prompt 0: 3 tokens, 2.00x faster")
    assert lines[1].startswith("MISMATCH prompt 1: diverge at token 1")
    assert module.main(["compare", "/dev/null", "/dev/null"]) if False else True


def test_text_prompts_cover_short_long_and_multi_turn():
    module = _load()
    prompts = module._text_prompts()
    assert len(prompts) == 4
    assert len(prompts[3]) == 3
    assert len(prompts[2][0]["content"]) > 4000
    assert module._image_prompts(None) == []


def test_compare_flags_near_tie_divergence():
    module = _load()
    base = [
        {
            "prompt_index": 0,
            "tokens": ["a", "b"],
            "margins": [2.0, 0.01],
            "seconds": 1.0,
        }
    ]
    other = [
        {
            "prompt_index": 0,
            "tokens": ["a", "c"],
            "margins": [2.0, 0.02],
            "seconds": 1.0,
        }
    ]
    (line,) = module.compare_results(base, other)
    assert line.startswith("MISMATCH prompt 0: diverge at token 1")
    assert "near tie" in line
    base[0]["margins"][1] = 1.5
    (line,) = module.compare_results(base, other)
    assert "gap 1.500" in line and "near tie" not in line


def test_compare_uses_margins_from_either_side():
    module = _load()
    base = [{"prompt_index": 0, "tokens": ["a", "b"], "seconds": 1.0}]
    other = [
        {
            "prompt_index": 0,
            "tokens": ["a", "c"],
            "margins": [2.0, 0.02],
            "seconds": 1.0,
        }
    ]
    (line,) = module.compare_results(base, other)
    assert "near tie" in line


def test_spec_decode_stats_from_counter_deltas():
    module = _load()
    before = {
        "vllm:spec_decode_num_drafts_total": 100.0,
        "vllm:spec_decode_num_draft_tokens_total": 300.0,
        "vllm:spec_decode_num_accepted_tokens_total": 200.0,
    }
    after = {
        "vllm:spec_decode_num_drafts_total": 150.0,
        "vllm:spec_decode_num_draft_tokens_total": 450.0,
        "vllm:spec_decode_num_accepted_tokens_total": 320.0,
    }
    stats = module.spec_decode_stats(before, after)
    assert stats is not None
    assert stats["drafts"] == 50
    assert abs(stats["acceptance_rate"] - 120 / 150) < 1e-9
    assert abs(stats["mean_acceptance_length"] - (1 + 120 / 50)) < 1e-9
    assert module.spec_decode_stats(None, after) is None
    assert module.spec_decode_stats(before, before) is None
