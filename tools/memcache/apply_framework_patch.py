"""Check or apply the paired source patch, refusing unknown or partially patched inputs."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def apply(framework, patch_dir, write=False):
    identity = json.loads((patch_dir / "identity.json").read_text())
    patch = patch_dir / "0001-compact-int8-memcache.patch"
    if digest(patch) != identity["patch_sha256"]:
        raise ValueError("Patch content differs from its release identity")
    files = identity["files"]
    if all(digest(framework / row["path"]) == row["after_sha256"] for row in files):
        return dict(passed=True, status="already_applied", files=len(files))
    changed = [
        row["path"] for row in files if digest(framework / row["path"]) != row["before_sha256"]
    ]
    if changed:
        raise ValueError("Framework inputs differ from the paired baseline: " + ", ".join(changed))
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=framework, check=True)
    if write:
        subprocess.run(["git", "apply", str(patch)], cwd=framework, check=True)
        if not all(digest(framework / row["path"]) == row["after_sha256"] for row in files):
            raise RuntimeError("Applied files differ from validated candidate")
    return dict(passed=True, status="applied" if write else "check_only", files=len(files))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    patch_dir = Path(__file__).resolve().parents[2] / "patches/vllm-ascend"
    print(json.dumps(apply(args.framework.resolve(strict=True), patch_dir, args.apply)))
