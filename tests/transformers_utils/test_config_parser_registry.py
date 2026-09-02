# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest
from transformers import PretrainedConfig

from vllm.transformers_utils.config import (
    HFConfigParser,
    get_config_parser,
    register_config_parser,
)
from vllm.transformers_utils.config_parser_base import ConfigParserBase


@register_config_parser("custom_config_parser")
class CustomConfigParser(ConfigParserBase):
    def parse(
        self,
        model: str | Path,
        trust_remote_code: bool,
        revision: str | None = None,
        code_revision: str | None = None,
        **kwargs,
    ) -> tuple[dict, PretrainedConfig]:
        raise NotImplementedError


def test_register_config_parser():
    assert isinstance(get_config_parser("custom_config_parser"), CustomConfigParser)


def test_deepseek_v4_config_format_aliases_hf():
    assert isinstance(get_config_parser("deepseek_v4"), HFConfigParser)
    assert isinstance(get_config_parser("hf"), HFConfigParser)


def test_invalid_config_parser():
    with pytest.raises(ValueError):

        @register_config_parser("invalid_config_parser")
        class InvalidConfigParser:
            pass
