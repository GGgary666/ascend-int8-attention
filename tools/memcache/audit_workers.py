"""Verify the 16 current worker audit files on this node, including source freshness."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(directory, parent_pid):
    deployment = json.loads((directory / "deployment.json").read_text())
    rank = deployment["node_rank"]
    proc = Path("/proc") / str(parent_pid)
    fields = (proc / "stat").read_text().rsplit(") ", 1)[1].split()
    if fields[0] == "Z":
        raise ValueError("The recorded parent process is a zombie")
    start_ticks = int(fields[19])
    boot = next(
        int(line.split()[1])
        for line in Path("/proc/stat").read_text().splitlines()
        if line.startswith("btime ")
    )
    started = boot + start_ticks / os.sysconf("SC_CLK_TCK")
    store = (
        Path(deployment["framework"]) / "vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store"
    )
    plugin = Path(deployment["plugin"]) / "npu_sparse_attn_sharedkv_int8"
    records = []
    for tp in range(16):
        path = directory / "worker-audits" / f"dp{rank}-tp{tp}.json"
        if path.stat().st_mtime < started:
            raise ValueError("Stale worker audit: " + path.name)
        record = json.loads(path.read_text())
        if (record["dp_rank"], record["tp_rank"]) != (rank, tp):
            raise ValueError("Worker rank mismatch")
        for name, expected in [
            ("worker_source_sha256", store / "pool_worker.py"),
            ("protocol_source_sha256", store / "cache_compatibility.py"),
        ]:
            if record[name] != sha(expected):
                raise ValueError("Worker source identity differs: " + name)
        for name, expected in [
            ("vllm_ascend", Path(deployment["framework"]) / "vllm_ascend/__init__.py"),
            ("npu_sparse_attn_sharedkv_int8", plugin / "__init__.py"),
        ]:
            if Path(record["modules"][name]).resolve() != expected.resolve():
                raise ValueError("Worker import path differs: " + name)
        for name, expected in record["native_sha256"].items():
            if sha(plugin / "_native" / name) != expected:
                raise ValueError("Worker native identity differs: " + name)
        records.append(dict(dp=rank, tp=tp, audit_sha256=sha(path), record=record))
    current = (proc / "stat").read_text().rsplit(") ", 1)[1].split()
    if current[0] == "Z" or int(current[19]) != start_ticks:
        raise ValueError("Parent changed during worker audit")
    return dict(
        passed=True,
        node_rank=rank,
        workers=records,
        parent_pid=parent_pid,
        parent_start_ticks=start_ticks,
        started=started,
        checked_at=time.time(),
        namespace=records[0]["record"]["namespace"],
        source_host_freshness_checked=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.deployment.resolve(strict=True), args.parent_pid)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(
        json.dumps(dict(passed=True, workers=len(result["workers"]), namespace=result["namespace"]))
    )
