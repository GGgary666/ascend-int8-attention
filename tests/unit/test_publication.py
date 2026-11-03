"""Prevent publication of private metadata hidden inside binary archives."""
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import torch


CHECKER = Path(__file__).resolve().parents[2] / "tools/check_publication.py"


def scan(path):
    result = subprocess.run([sys.executable, str(CHECKER), str(path)],
                            capture_output=True, text=True, check=False)
    return result, json.loads(result.stdout)


def test_detects_private_metadata_in_compressed_wheel(tmp_path):
    private_path = "/" + "/".join(("home", "example", "models", "example-model"))
    payload = io.BytesIO()
    torch.save({"provenance": {"model": private_path}}, payload)
    wheel = tmp_path / "example.whl"
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("package/data/scales.pt", payload.getvalue())
    result, report = scan(wheel)
    assert result.returncode == 1 and not report["complete"]
    assert any(item["kind"] == "machine_path" and ":pickle@" in item["file"]
               for item in report["findings"])
    assert private_path not in result.stdout  # Findings do not repeat private values.


def test_accepts_public_model_identifier_and_standard_installation_path(tmp_path):
    path = tmp_path / "public.pt"
    torch.save({"model": "example-model", "cann": "/usr/local/Ascend/cann-9.2.0"}, path)
    result, report = scan(path)
    assert result.returncode == 0 and report["complete"]
    assert report["pickle_strings_checked"] > 0 and not report["findings"]
