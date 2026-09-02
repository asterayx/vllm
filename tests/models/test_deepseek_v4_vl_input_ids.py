# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Official v0.28.0 pops ``input_ids`` from the HF processor output.

``DeepseekV4VLProcessor`` only emits vision tensors, so the multimodal
processor must tokenize the prompt itself.
"""

from types import SimpleNamespace

import pytest
from PIL import Image
from transformers import BatchFeature

from vllm.models.deepseek_v4.common.mm_preprocess import (
    DeepseekV4VLMultiModalProcessor,
    DeepseekV4VLProcessor,
)
from vllm.multimodal.parse import MultiModalDataParser
from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config

pytestmark = pytest.mark.cpu_test


class _StubTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = [11, 12, 13]
        if add_special_tokens:
            return [1, *ids]
        return ids


class _StubCtx:
    def call_hf_processor(self, hf_processor, data, kwargs=None):
        allowed = {"return_tensors": "pt"}
        if kwargs:
            allowed.update(kwargs)
        return hf_processor(**data, **allowed)


class _StubInfo:
    def get_hf_config(self):
        return DeepseekV4Config(vocab_size=129280)

    def get_hf_processor(self, **kwargs):
        return DeepseekV4VLProcessor(self.get_hf_config())

    def get_tokenizer(self):
        return _StubTokenizer()

    def get_data_parser(self):
        return MultiModalDataParser()

    @property
    def ctx(self):
        return _StubCtx()


def test_vl_hf_processor_does_not_emit_input_ids():
    processor = DeepseekV4VLProcessor(DeepseekV4Config(vocab_size=129280))
    out = processor(
        text="<｜deepseek_image｜>",
        images=[Image.new("RGB", (64, 64), color=(10, 20, 30))],
    )
    assert "patches" in out
    assert "input_ids" not in out


def test_vl_multimodal_processor_adds_input_ids():
    processor = DeepseekV4VLMultiModalProcessor(_StubInfo(), None)
    assert (
        processor._hf_processor_applies_updates(
            "<｜deepseek_image｜>",
            SimpleNamespace(),  # type: ignore[arg-type]
            {},
            {},
        )
        is False
    )

    processed = processor._call_hf_processor(
        prompt="<｜deepseek_image｜>",
        mm_data={"images": [Image.new("RGB", (64, 64), color=(10, 20, 30))]},
        mm_kwargs={},
        tok_kwargs={},
    )
    assert isinstance(processed, BatchFeature)
    input_ids = processed.pop("input_ids")
    (prompt_ids,) = input_ids
    assert prompt_ids == [11, 12, 13]
    assert "patches" in processed
