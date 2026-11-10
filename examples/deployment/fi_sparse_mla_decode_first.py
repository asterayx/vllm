# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Write a decode-first copy of FlashInfer's SM120 sparse-MLA calibration.

FlashInfer's autotune stores a decode/prefill crossover per (heads, topk) in
``sparse_mla_sm120_cpb.json``. On the RTX PRO 5000 box it routes every
DeepSeek-V4.1 target decode (topk 128, dual cache) to ``prefill_mg_dual``,
although ``sparse_mla_decode_dsv4`` is faster there (#22 P3: attention kernel
time per step 0.43 -> 0.30 ms on PP0, 0.66 -> 0.35 ms on PP1).

The copy raises every ``dsv4|*`` crossover to 64 (the decode-kernel limit,
i.e. FlashInfer's uncalibrated default) and keeps everything else. Point the
server at it with ``FLASHINFER_AUTOTUNE_DIR=<out-dir>``; the source file is
not modified.

    python fi_sparse_mla_decode_first.py --out-dir ~/dpsk_deploy/fi_decode_first
"""

import argparse
import json
import pathlib

FILE_NAME = "sparse_mla_sm120_cpb.json"
DECODE_MAX_TOKENS = 64


def default_source() -> pathlib.Path:
    from flashinfer.mla import _sparse_mla_sm120_cpb as cpb

    return cpb.default_cache_path()


def make_decode_first(payload: dict) -> int:
    """Set every dsv4 crossover to DECODE_MAX_TOKENS; return how many."""
    changed = 0

    def walk(node):
        nonlocal changed
        if not isinstance(node, dict):
            return
        for key, value in node.items():
            if key == "decode_max_tokens" and isinstance(value, dict):
                for entry in value:
                    if entry.startswith("dsv4|"):
                        value[entry] = DECODE_MAX_TOKENS
                        changed += 1
            else:
                walk(value)

    walk(payload)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=pathlib.Path, default=None)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    src = args.src or default_source()
    out_dir = args.out_dir.expanduser()
    if out_dir.resolve() == src.parent.resolve():
        raise SystemExit("--out-dir must differ from the source directory")
    payload = json.loads(src.read_text())
    changed = make_decode_first(payload)
    if changed == 0:
        raise SystemExit(f"no dsv4 decode_max_tokens entries in {src}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / FILE_NAME
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"{src} -> {out}: {changed} dsv4 entries set to {DECODE_MAX_TOKENS}")


if __name__ == "__main__":
    main()
