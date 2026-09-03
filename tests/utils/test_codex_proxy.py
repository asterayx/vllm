# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "docker" / "gb10" / "codex_proxy.py"
_SPEC = importlib.util.spec_from_file_location("codex_proxy", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
codex_proxy = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(codex_proxy)


def test_apply_codex_message_prefers_response_content_over_reasoning():
    msg = codex_proxy.apply_codex_message(
        {
            "role": "assistant",
            "content": "",
            "response_content": "hello",
            "reasoning_content": "thinking",
        }
    )
    assert msg["content"] == "hello"


def test_apply_codex_message_falls_back_to_reasoning():
    msg = codex_proxy.apply_codex_message(
        {"role": "assistant", "reasoning_content": "only-think"}
    )
    assert msg["content"] == "only-think"


def test_apply_codex_message_keeps_existing_content():
    msg = codex_proxy.apply_codex_message(
        {
            "role": "assistant",
            "content": "keep",
            "response_content": "ignore",
        }
    )
    assert msg["content"] == "keep"


def test_transform_chat_completion_message_and_delta():
    out = codex_proxy.transform_chat_completion(
        {
            "choices": [
                {"message": {"content": None, "response_content": "ans"}},
                {"delta": {"reasoning_content": "tok"}},
            ]
        }
    )
    assert out["choices"][0]["message"]["content"] == "ans"
    assert out["choices"][1]["delta"]["content"] == "tok"


def test_transform_responses_fills_output_text():
    out = codex_proxy.transform_responses(
        {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "",
                            "response_content": "vis",
                        }
                    ],
                }
            ]
        }
    )
    assert out["output"][0]["content"][0]["text"] == "vis"


def test_transform_sse_rewrites_data_and_keeps_done():
    body = (
        'data: {"choices":[{"delta":{"response_content":"x"}}]}\n\n'
        "data: [DONE]\n\n"
    )
    out = codex_proxy.transform_sse_body(body)
    assert '"content":"x"' in out
    assert "data: [DONE]" in out


def test_transform_openai_json_dispatches():
    chat = codex_proxy.transform_openai_json(
        {"choices": [{"message": {"response_content": "a"}}]}
    )
    resp = codex_proxy.transform_openai_json(
        {"output": [{"type": "message", "response_content": "b"}]}
    )
    assert chat["choices"][0]["message"]["content"] == "a"
    assert resp["output"][0]["content"][0]["text"] == "b"
