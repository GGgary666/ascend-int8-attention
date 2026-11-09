"""Check one node's prepared deployment on CPU before starting any model worker."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def check(directory, metadata_only=False):
    config = json.loads((directory / "deployment.json").read_text())
    sys.path[:0] = [config["plugin"], config["framework"]]
    import torch
    from npu_sparse_attn_sharedkv_int8.memcache_manifest import scale_values_digest

    contract_path = (
        Path(config["framework"])
        / "vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/cache_compatibility.py"
    )
    spec = importlib.util.spec_from_file_location("memcache_preflight_contract", contract_path)
    contract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contract)
    manifest = contract.load_manifest(config["manifest"])
    contract.validate_memcache_config(directory / "mmc-client.conf")
    if "ock.mmc.local_service.dram.size = 256GB" not in (directory / "mmc-pool.conf").read_text():
        raise ValueError("This deployment requires one 256 GiB pool per node")
    plugin = Path(config["plugin"]) / "npu_sparse_attn_sharedkv_int8"
    for category, parent in [
        ("plugin_sources_sha256", plugin),
        ("native_sha256", plugin / "_native"),
    ]:
        for name, expected in manifest[category].items():
            if digest(parent / name) != expected:
                raise ValueError("Paired plugin checksum mismatch: " + name)
    scales = torch.load(
        config["environment"]["SHAREDKV_SCALES"], weights_only=True, map_location="cpu"
    )["scales"]
    if scale_values_digest(scales) != manifest["scales_digest"]:
        raise ValueError("Scale values differ from the compatibility manifest")
    for item in manifest["weight_manifest"]:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid relative model path")
        path = Path(config["model"]) / relative
        if path.stat().st_size != item["size"]:
            raise ValueError("Model file size mismatch: " + item["path"])
        if not metadata_only or item["path"] in manifest["metadata_files"]:
            if digest(path) != item["sha256"]:
                raise ValueError("Model content mismatch: " + item["path"])
    driver = {}
    for name in ("/etc/ascend_install.info", "/usr/local/Ascend/driver/version.info"):
        driver[name] = digest(Path(name))
    modules = {}
    for name, root in [
        ("vllm_ascend", config["framework"]),
        ("npu_sparse_attn_sharedkv_int8", config["plugin"]),
    ]:
        origin = Path(importlib.util.find_spec(name).origin).resolve()
        if not origin.is_relative_to(Path(root).resolve()):
            raise ValueError("Unexpected import origin: " + name)
        modules[name] = str(origin)
    packages = {
        name: importlib.metadata.version(name)
        for name in (
            "torch",
            "torch_npu",
            "vllm",
            "vllm-ascend",
            "memcache-hybrid",
            "memfabric-hybrid",
            "npu-sparse-attn-sharedkv-int8",
        )
    }
    expected_packages = {
        "torch": "2.10.0+cpu",
        "torch_npu": "2.10.0.post7",
        "vllm": "0.30.0+empty",
        "vllm-ascend": "0.30.0rc2",
        "memcache-hybrid": "1.3.0",
        "memfabric-hybrid": "1.3.0",
        "npu-sparse-attn-sharedkv-int8": "0.65.1+memcache.10",
    }
    for name, expected in expected_packages.items():
        if packages[name] != expected:
            raise ValueError(
                f"Package version differs from validated deployment: {name}={packages[name]}, expected {expected}"
            )
    progress = contract_path.parents[4] / "patch/platform/patch_compact_load_progress.py"
    spec = importlib.util.spec_from_file_location("memcache_progress_contract", progress)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    vllm_path = Path(importlib.util.find_spec("vllm").origin).parent
    if digest(vllm_path / "v1/engine/core.py") != module.ENGINE_CORE_SHA256:
        raise ValueError("The vLLM engine loop does not match the tested receive barrier")
    return dict(
        passed=True,
        node_rank=config["node_rank"],
        modules=modules,
        packages=packages,
        model_content_verified=not metadata_only,
        model_digest=manifest["model_digest"],
        scales_digest=manifest["scales_digest"],
        identity_digest=manifest["identity_digest"],
        protocol=manifest["protocol"],
        driver_metadata_sha256=driver,
        patch_sources_sha256=manifest["patch_sources_sha256"],
        plugin_sources_sha256=manifest["plugin_sources_sha256"],
        native_sha256=manifest["native_sha256"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Skip weight rehash only after an earlier complete two-node content check",
    )
    args = parser.parse_args()
    result = check(args.deployment.resolve(strict=True), args.metadata_only)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(
        json.dumps(
            dict(
                passed=True,
                model_content_verified=result["model_content_verified"],
                protocol=result["protocol"],
                identity_digest=result["identity_digest"],
            )
        )
    )
