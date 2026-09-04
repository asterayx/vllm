# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
from pathlib import Path

_GB10 = Path(__file__).resolve().parents[2] / "docker" / "gb10"
_STAMP = _GB10 / "stamp-version.py"
_SPEC = importlib.util.spec_from_file_location("gb10_stamp_version", _STAMP)
assert _SPEC is not None and _SPEC.loader is not None
stamp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(stamp)


def test_version_file_is_spark_release():
    raw = (_GB10 / "VERSION").read_text().strip().splitlines()
    line = next(x.split("#", 1)[0].strip() for x in raw if x.split("#", 1)[0].strip())
    major, minor, patch, local = stamp.parse_spark_version(line)
    assert (major, minor, patch) == (0, 28, 0)
    assert local.startswith("dsv4.spark.")
    assert line == f"0.28.0+{local}"


def test_stamp_writes_version_module(tmp_path):
    root = tmp_path / "src"
    (root / "vllm").mkdir(parents=True)
    dest = stamp.stamp_version_py(root, "0.28.0+dsv4.spark.1")
    ns: dict = {}
    exec(dest.read_text(), ns)
    assert ns["__version__"] == "0.28.0+dsv4.spark.1"
    assert ns["__version_tuple__"] == (0, 28, 0, "dsv4.spark.1")


def test_stamp_rewrites_dist_info(tmp_path):
    meta = tmp_path / "vllm-0.28.0.dist-info" / "METADATA"
    meta.parent.mkdir(parents=True)
    meta.write_text("Metadata-Version: 2.1\nName: vllm\nVersion: 0.28.0\n")
    written = stamp.stamp_dist_info("0.28.0+dsv4.spark.1", [tmp_path])
    assert written == [meta]
    assert "Version: 0.28.0+dsv4.spark.1" in meta.read_text()
