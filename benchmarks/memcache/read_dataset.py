"""Read the frozen maximum-pressure case. No HTTP or model calls."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np


def metadata(directory):
    return [json.loads(x) for x in (Path(directory) / "requests.jsonl").open()]


def iter_requests(directory, *, run_id, model="DeepSeek-V4-Pro"):
    if not run_id or run_id == "REPLACE_WITH_RUN_ID":
        raise ValueError("fresh run_id required")
    with gzip.open(Path(directory) / "payloads.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            item["request"]["model"] = model
            item["request"]["cache_salt"] = item["request"]["cache_salt"].replace(
                "REPLACE_WITH_RUN_ID", run_id
            )
            yield item


def verify(directory):
    directory = Path(directory)
    rows = metadata(directory)
    source = {
        r["id"]: r for r in json.loads((directory / "source_features.json").read_text())["selected"]
    }
    mm = np.memmap(directory / "data/prompts.u32", dtype="<u4", mode="r")
    by = {r["id"]: r for r in rows}
    cursor = 0
    count = 0
    for i, item in enumerate(iter_requests(directory, run_id="OFFLINE_VERIFY")):
        r = item["metadata"]
        body = item["request"]
        src = source[r["id"]]
        assert (
            r == rows[i]
            and i == r["replay_order"]
            and r["source_order"] == rows[0]["source_order"] + i
        )
        a = np.asarray(body["prompt"], dtype="<u4")
        assert len(a) == r["input_tokens"] == src["input_tokens"]
        assert body["max_tokens"] == r["output_tokens"] == src["output_tokens"]
        assert r["seq_len"] == len(a) + r["output_tokens"] <= 135000
        assert (
            np.array_equal(a, mm[cursor : cursor + len(a)]) and cursor == r["prompt_offset_tokens"]
        )
        assert hashlib.sha256(a.tobytes()).hexdigest() == r["prompt_sha256"]
        assert item["headers"]["X-data-parallel-rank"] == str(r["dp_rank"])
        ref = r["prefix_reference_id"]
        if ref:
            p = by[ref]
            b = mm[p["prompt_offset_tokens"] : p["prompt_offset_tokens"] + p["input_tokens"]]
            k = r["frozen_shared_prefix_tokens"]
            assert np.array_equal(a[:k], b[:k])
            assert k >= r["source_history_copied_tokens"]
            assert p["replay_order"] < i and p["source_end_s"] <= r["source_start_s"]
            assert p["dp_rank"] == r["dp_rank"] and p["identity_scope_id"] == r["identity_scope_id"]
        cursor += len(a)
        count += 1
    assert count == len(rows) == len(source) == 324 and len(mm) == cursor
    return {
        "passed": True,
        "requests": count,
        "source_lengths_and_global_order_preserved": True,
        "frozen_prompts_and_prefixes_verified": True,
        "no_inference_executed": True,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", nargs="?", default=str(Path(__file__).resolve().parent))
    p.add_argument("--verify", action="store_true")
    a = p.parse_args()
    print(json.dumps(verify(a.directory), ensure_ascii=False, indent=2))
