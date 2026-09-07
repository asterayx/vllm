# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""docker/gb10/summarize-profile.py category grouping and totals."""

import gzip
import importlib.util
import json
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2] / "docker" / "gb10" / "summarize-profile.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("summarize_profile", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_categorize_kernel_names():
    m = _load()
    assert m.categorize("ncclDevKernel_AllReduce_Sum_bf16_RING_LL") == "nccl"
    assert m.categorize("sparse_mla_sm120_decode_dsv4_kernel") == "attention"
    assert m.categorize("b12x::MoEDynamicKernelSilu") == "moe"
    assert m.categorize("mhc_post_tilelang_kernel") == "mhc"
    assert m.categorize("humming_fp8_gemm_kernel") == "dense_gemm"
    assert m.categorize("void at::native::elementwise_kernel") == "glue"


def test_summarize_reads_gzip_trace_and_reports_per_token(tmp_path):
    m = _load()
    events = [
        {
            "ph": "X",
            "cat": "kernel",
            "name": "ncclDevKernel_AllReduce",
            "ts": 0,
            "dur": 100,
        },
        {
            "ph": "X",
            "cat": "kernel",
            "name": "sparse_mla_sm120_decode",
            "ts": 100,
            "dur": 300,
        },
        {"ph": "X", "cat": "kernel", "name": "b12x_moe", "ts": 400, "dur": 600},
        {"ph": "X", "cat": "cpu_op", "name": "aten::add", "ts": 0, "dur": 5000},
    ]
    path = tmp_path / "t.pt.trace.json.gz"
    with gzip.open(path, "wt") as handle:
        json.dump({"traceEvents": events}, handle)
    lines = m.summarize(m.load_events(path), top=5, tokens=10)
    assert lines[0].startswith(
        "GPU busy 1.0 ms over 1.0 ms profiled (100% utilization)"
    )
    assert "0.10 GPU ms" in lines[1]
    assert any(line.startswith("moe") and "60.0%" in line for line in lines)
    assert not any("aten::add" in line for line in lines)
