"""Check source integrity and the imported p101 device sources without NPU access."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_files(root):
    excluded = {".git", "build", "dist", "__pycache__", ".pytest_cache", ".ruff_cache",
                ".venv", "validation", "benchmark-results", "_native"}
    for directory, names, files in os.walk(root):
        names[:] = sorted(n for n in names if n not in excluded and not n.endswith(".egg-info"))
        for filename in sorted(files):
            relative = (Path(directory) / filename).relative_to(root).as_posix()
            if relative not in ("release/source_manifest.json", "PKG-INFO") and not filename.endswith(".pyc"):
                yield relative


def verify(root, include_build=False):
    manifest = json.loads((root / "release/source_manifest.json").read_text())
    upstream = json.loads((root / "release/upstream-v65-p101.json").read_text())
    imported = json.loads((root / "release/import-map.json").read_text())
    failures = []
    actual = set(source_files(root))
    listed = set(manifest["sha256"])
    failures.extend(f"Unlisted source: {name}" for name in sorted(actual - listed))
    for name, expected in manifest["sha256"].items():
        path = root / name
        if not path.is_file():
            failures.append(f"Missing source: {name}")
        elif digest(path) != expected:
            failures.append(f"Changed source: {name}")
    unchanged_native = 0
    for old, new in imported["mappings"].items():
        if old.startswith(("kernels/", "host/")) and Path(old).suffix in (".cpp", ".h"):
            unchanged_native += 1
            if not (root / new).is_file() or digest(root / new) != upstream["sources"][old]["sha256"]:
                failures.append(f"Changed upstream native source: {new}")
    device_identity = {}
    if include_build:
        native = root / "npu_sparse_attn_sharedkv_int8/_native"
        record = native / "manifest.json"
        if not record.is_file():
            failures.append("Missing native build; run ./build.sh")
        else:
            build = json.loads(record.read_text())
            if not build["complete"]:
                failures.append("Incomplete native build")
            for name, expected in build["source_sha256"].items():
                if not (root / name).is_file() or digest(root / name) != expected:
                    failures.append(f"Stale build input: {name}")
            for name, expected in build["libraries"].items():
                if not (native / name).is_file() or digest(native / name) != expected:
                    failures.append(f"Changed native library: {name}")
            for name in ("libsharedkv_c8.so", "libsharedkv_int8.so"):
                path = native / name
                matches = path.is_file() and digest(path) == upstream["binaries"]["build/" + name]["sha256"]
                device_identity[name] = {"matches_upstream": matches}
                if not matches:
                    failures.append(f"Device library differs from frozen p101: {name}")
    return {
        "release": upstream["release"], "complete": not failures,
        "source_count": len(manifest["sha256"]), "unchanged_native_sources": unchanged_native,
        "build_checked": include_build, "device_identity": device_identity, "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-build", action="store_true")
    args = parser.parse_args()
    result = verify(Path(__file__).resolve().parents[1], args.include_build)
    print(json.dumps(result, indent=2))
    raise SystemExit(not result["complete"])


if __name__ == "__main__":
    main()
