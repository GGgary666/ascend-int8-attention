"""Hash model configuration and every indexed weight shard once, before deployment."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def freeze(model):
    model = model.resolve(strict=True)
    indexes = sorted(model.glob("*.safetensors.index.json"))
    if len(indexes) != 1:
        raise ValueError("Expected one safetensors shard index")
    index_bytes = indexes[0].read_bytes()
    shards = set(json.loads(index_bytes)["weight_map"].values())
    names = sorted({"config.json", "quant_model_description.json", indexes[0].name, *shards})
    files = []
    for name in names:
        path = model / name
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Weight index contains a non-relative shard name")
        before = path.stat()
        checksum = digest(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ino,
        ):
            raise ValueError("Model changed while hashing: " + name)
        files.append(dict(path=name, size=after.st_size, sha256=checksum))
    if (
        next(row["sha256"] for row in files if row["path"] == indexes[0].name)
        != hashlib.sha256(index_bytes).hexdigest()
    ):
        raise ValueError("Weight index changed during model hashing")
    value = dict(format_version=1, files=files)
    value["model_digest"] = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = freeze(args.model)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps(dict(files=len(result["files"]), model_digest=result["model_digest"])))
