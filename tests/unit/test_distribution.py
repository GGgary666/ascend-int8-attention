"""Check the relocated package without initializing device runtimes."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from npu_sparse_attn_sharedkv_int8._paths import SCALES_PATH, native_path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_all_imported_native_sources_match_frozen_release():
    upstream = json.loads((ROOT / "release/upstream-v65-p101.json").read_text())
    mapping = json.loads((ROOT / "release/import-map.json").read_text())["mappings"]
    checked = []
    for original, relative in mapping.items():
        if original.startswith(("kernels/", "host/")) and Path(original).suffix in (".cpp", ".h"):
            actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            assert actual == upstream["sources"][original]["sha256"], relative
            checked.append(relative)
    assert len(checked) == 45  # 43 device files and two host translation units.


def test_calibration_preserves_frozen_values_after_metadata_sanitization():
    upstream = json.loads((ROOT / "release/upstream-v65-p101.json").read_text())
    published = json.loads((ROOT / "release/publication.json").read_text())["calibration"]
    assert published["upstream_sha256"] == upstream["sources"]["data/scales_v1.pt"]["sha256"]
    assert hashlib.sha256(SCALES_PATH.read_bytes()).hexdigest() == published["published_sha256"]
    payload = torch.load(SCALES_PATH, map_location="cpu", weights_only=True)
    assert payload["provenance"]["model"] == published["model_identifier"]
    assert len(payload["scales"]) == published["scale_tensor_count"] == 125
    values = hashlib.sha256()
    for key, scale in sorted(payload["scales"].items()):
        assert scale.dtype == torch.float32 and scale.shape == (512,)
        values.update(key.encode() + b"\0")
        values.update(scale.contiguous().numpy().tobytes())
    assert values.hexdigest() == published["scale_values_sha256"]


def test_missing_native_library_has_actionable_error():
    with pytest.raises(RuntimeError, match="Run ./build.sh"):
        native_path("intentionally_missing.so")


def test_reference_and_disabled_plugin_do_not_import_device_backends(tmp_path):
    code = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from npu_sparse_attn_sharedkv_int8 import CacheQuantization, SharedKVCase
from npu_sparse_attn_sharedkv_int8.vllm_plugin import register
register()
assert 'torch_npu' not in sys.modules
assert 'vllm' not in sys.modules
import npu_sparse_attn_sharedkv_int8 as package
assert Path(package.__file__).resolve().parent.parent == Path(sys.argv[1])
"""
    env = {k: v for k, v in os.environ.items() if not k.startswith("SHAREDKV_")}
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    subprocess.run([sys.executable, "-I", "-c", code, str(ROOT)], env=env, cwd=tmp_path, check=True)
