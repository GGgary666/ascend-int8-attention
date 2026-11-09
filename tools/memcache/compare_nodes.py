"""Require matching deployment identities and a complete set of 32 fresh worker records."""

import argparse
import json
from pathlib import Path


def compare(preflights, workers):
    preflights = sorted(preflights, key=lambda row: row["node_rank"])
    workers = sorted(workers, key=lambda row: row["node_rank"])
    if [row["node_rank"] for row in preflights] != [0, 1] or [
        row["node_rank"] for row in workers
    ] != [0, 1]:
        raise ValueError("Provide exactly one record from each DP node")
    for row in preflights:
        if not row["passed"] or not row["model_content_verified"]:
            raise ValueError("Both nodes need a complete preflight, including model content")
    for key in (
        "packages",
        "model_digest",
        "scales_digest",
        "identity_digest",
        "protocol",
        "patch_sources_sha256",
        "plugin_sources_sha256",
        "native_sha256",
        "driver_metadata_sha256",
    ):
        if preflights[0][key] != preflights[1][key]:
            raise ValueError("Node identity mismatch: " + key)
    records = []
    for row in workers:
        if not row["passed"] or not row["source_host_freshness_checked"]:
            raise ValueError("Worker freshness was not checked on its source host")
        records.extend(row["workers"])
    if {(row["dp"], row["tp"]) for row in records} != {
        (dp, tp) for dp in (0, 1) for tp in range(16)
    } or len(records) != 32:
        raise ValueError("Expected all 32 unique workers")
    namespaces = {row["record"]["namespace"] for row in records}
    if len(namespaces) != 1:
        raise ValueError("Worker namespaces differ across DP or TP")
    for row in records:
        if row["record"]["native_sha256"] != preflights[0]["native_sha256"]:
            raise ValueError("Worker native libraries differ from preflight")
        if row["record"]["protocol"] != preflights[0]["protocol"]:
            raise ValueError("Worker protocol differs from preflight")
    return dict(
        passed=True,
        workers=32,
        namespace=namespaces.pop(),
        identity_digest=preflights[0]["identity_digest"],
        protocol=preflights[0]["protocol"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", nargs=2, type=Path, required=True)
    parser.add_argument("--workers", nargs=2, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(
        [json.loads(path.read_text()) for path in args.preflight],
        [json.loads(path.read_text()) for path in args.workers],
    )
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps(result))
