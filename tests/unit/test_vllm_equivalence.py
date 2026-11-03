"""Keep observed baseline variation separate from strict output equivalence."""
import json
from pathlib import Path
import subprocess
import sys


CHECKER = Path(__file__).resolve().parents[2] / "tools/check_vllm_equivalence.py"


def run_comparison(tmp_path, *, allow_variants=False, change_prompt=False):
    def row(name, output):
        return {"case": name, "dp_rank": 0, "prompt_token_ids": [1, 2],
                "token_ids": [output], "text": str(output), "finish_reason": "stop",
                "logprobs": [-0.1], "usage": {}}
    baseline = {"model": "test", "temperature": 0, "seed": 42,
                "rows": [row("cold", 10), row("cached", 11)]}
    candidate = json.loads(json.dumps(baseline))
    candidate["rows"][1] = row("cached", 10)
    if change_prompt:
        candidate["rows"][1]["prompt_token_ids"] = [3, 4]
    for name, value in (("baseline", baseline), ("candidate", candidate)):
        (tmp_path / (name + ".json")).write_text(json.dumps(value))
    output = tmp_path / "diff.json"
    command = [sys.executable, str(CHECKER), "compare",
               "--baseline", str(tmp_path / "baseline.json"),
               "--candidate", str(tmp_path / "candidate.json"), "--output", str(output)]
    if allow_variants:
        command.append("--allow-observed-prompt-variants")
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return result.returncode, json.loads(output.read_text())


def test_observed_variant_requires_explicit_mode_and_reports_strict_mismatch(tmp_path):
    status, strict = run_comparison(tmp_path)
    assert status == 1 and not strict["passed"]
    status, compatible = run_comparison(tmp_path, allow_variants=True)
    assert status == 0 and compatible["passed"]
    assert compatible["strict_same_case_matches"] == 1
    assert compatible["observed_prompt_variant_matches"] == 1
    assert compatible["rows"][1]["matched_baseline_case"] == "cold"


def test_observed_variant_never_accepts_a_different_prompt(tmp_path):
    status, report = run_comparison(tmp_path, allow_variants=True, change_prompt=True)
    assert status == 1 and not report["passed"]
    assert not report["rows"][1]["prompt_ids_equal"]
