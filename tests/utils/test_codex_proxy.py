# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "docker" / "gb10" / "codex_proxy.py"
_SPEC = importlib.util.spec_from_file_location("codex_proxy", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
codex_proxy = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(codex_proxy)


def test_alias_copies_reasoning_to_reasoning_content():
    obj = {"choices": [{"delta": {"reasoning": "think"}}]}
    codex_proxy.alias(obj)
    assert obj["choices"][0]["delta"]["reasoning_content"] == "think"
    assert obj["choices"][0]["delta"]["reasoning"] == "think"


def test_alias_does_not_overwrite_existing_reasoning_content():
    obj = {"reasoning": "new", "reasoning_content": "keep"}
    codex_proxy.alias(obj)
    assert obj["reasoning_content"] == "keep"


def test_alias_nested_list_and_message():
    obj = {
        "output": [
            {"content": [{"reasoning": "a"}]},
            {"message": {"reasoning": "b"}},
        ]
    }
    codex_proxy.alias(obj)
    assert obj["output"][0]["content"][0]["reasoning_content"] == "a"
    assert obj["output"][1]["message"]["reasoning_content"] == "b"


def test_rewrite_sse_aliases_data_and_keeps_done():
    line = 'data: {"choices":[{"delta":{"reasoning":"x"}}]}'
    out = codex_proxy.rewrite_sse(line)
    payload = out[len("data: ") :]
    obj = __import__("json").loads(payload)
    assert obj["choices"][0]["delta"]["reasoning_content"] == "x"
    assert codex_proxy.rewrite_sse("data: [DONE]") == "data: [DONE]"
    assert codex_proxy.rewrite_sse(": keep") == ": keep"


def test_alias_json_bytes_roundtrip():
    raw = b'{"reasoning":"z"}'
    out = codex_proxy._alias_json_bytes(raw)
    assert b'"reasoning_content": "z"' in out or b'"reasoning_content":"z"' in out
