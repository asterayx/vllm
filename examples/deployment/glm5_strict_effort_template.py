# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Write a GLM-5.3 chat template that rejects unsupported reasoning efforts.

GLM-5.3's template only understands ``low`` and ``high``; every other value
(including ``medium`` and ``none``) silently becomes ``max``. The generated
template raises instead, which vLLM returns to the client as HTTP 400.

Usage: python glm5_strict_effort_template.py MODEL_DIR OUTPUT_PATH
"""

import json
import sys
from pathlib import Path

GUARD = (
    "{%- if reasoning_effort is defined and reasoning_effort is not none "
    "and reasoning_effort not in ['low', 'high', 'max'] -%}"
    "{{- raise_exception('reasoning_effort=' ~ reasoning_effort ~ ' is not "
    "supported by GLM-5.3; use low, high or max') -}}"
    "{%- endif -%}\n"
)


def main(model_dir: str, output: str) -> None:
    model = Path(model_dir).expanduser()
    jinja = model / "chat_template.jinja"
    if jinja.is_file():
        template = jinja.read_text()
    else:
        config = json.loads((model / "tokenizer_config.json").read_text())
        template = config.get("chat_template")
    if not isinstance(template, str) or "reasoning_effort" not in template:
        sys.exit(f"{model}: chat template does not use reasoning_effort")
    Path(output).write_text(GUARD + template)


if __name__ == "__main__":
    main(*sys.argv[1:3])
