"""Offline cache identity and runtime validation for the paired connector."""

import hashlib
import os
from pathlib import Path
import struct


def validate_transfer_mode(config, compact_pages):
    """Reject unversioned KV transfer for the plugin's padded INT8 layout."""
    if getattr(config, "kv_transfer_config", None) is not None and not compact_pages:
        raise ValueError("INT8 KV transfer requires compact pages and the paired MemCache protocol")


def scale_values_digest(scales):
    """Hash names, dtype, shape and little-endian FP32 values, not .pt metadata."""
    import torch  # CPU only; keep disabled plugin imports device independent.
    digest = hashlib.sha256()
    for name, tensor in sorted(scales.items()):
        if tensor.dtype != torch.float32 or tensor.shape != (512,):
            raise ValueError(f"Invalid frozen scale dtype/shape: {name}")
        if not bool(torch.isfinite(tensor).all() & (tensor > 0).all()):
            raise ValueError(f"Invalid frozen scale values: {name}")
        digest.update(name.encode() + b"\0float32\0[512]\0")
        digest.update(struct.pack("<512f", *tensor.tolist()))
    if not scales:
        raise ValueError("Frozen scale collection is empty")
    return digest.hexdigest()


def make_manifest(model_manifest, scales, connector_contract):
    """Consume the precomputed weight manifest without re-reading weights."""
    files = model_manifest["files"]
    if connector_contract.canonical_digest({"format_version": 1, "files": files}) != model_manifest["model_digest"]:
        raise ValueError("Model content manifest checksum mismatch")
    identity = {
        "format_version": 1, "protocol": connector_contract.SHAREDKV_CACHE_PROTOCOL,
        "encoding": connector_contract.SHAREDKV_ENCODING,
        "compute_revision": connector_contract.SHAREDKV_COMPUTE_REVISION,
        "layout": "compact-c8-v1", "model_digest": model_manifest["model_digest"],
        "scales_digest": scale_values_digest(scales),
    }
    package = Path(__file__).parent
    return {**identity, "identity_digest": connector_contract.canonical_digest(identity),
            "patch_sources_sha256": connector_contract.patch_source_hashes(),
            "plugin_sources_sha256": {name: connector_contract.content_digest(package / name)
                                      for name in ("memcache_manifest.py", "cache_layout.py", "vllm_adapter.py")},
            "native_sha256": {path.name: connector_contract.content_digest(path)
                              for path in (package / "_native").glob("*.so")},
            "metadata_files": {entry["path"]: entry["sha256"] for entry in files
                               if entry["path"] in ("config.json", "quant_model_description.json")},
            "weight_manifest": files}


def validate_transfer(config, scales):
    # Importing the paired framework happens only on the enabled serving path.
    try:
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import cache_compatibility as contract
    except ImportError as exc:
        raise ValueError("Compact C8 MemCache requires the paired vLLM-Ascend cache protocol patch") from exc
    path = config.kv_transfer_config.kv_connector_extra_config.get("cache_compatibility_manifest")
    if not path:
        raise ValueError("Compact C8 MemCache requires cache_compatibility_manifest")
    manifest = contract.load_manifest(Path(path))
    contract.validate_configuration(config, manifest)
    package = Path(__file__).parent
    expected_sources = manifest.get("plugin_sources_sha256", {})
    expected_native = manifest.get("native_sha256", {})
    if set(expected_sources) != {"memcache_manifest.py", "cache_layout.py", "vllm_adapter.py"}:
        raise ValueError("Missing paired INT8 plugin source checksums")
    if set(expected_native) != {"libsharedkv_c8.so", "libsharedkv_int8.so", "sharedkv_host.so", "sharedkv_c8_host.so"}:
        raise ValueError("Missing frozen p101 native library checksums")
    for name, digest in expected_sources.items():
        if contract.content_digest(package / name) != digest:
            raise ValueError(f"Paired INT8 plugin source checksum mismatch: {name}")
    for name, digest in expected_native.items():
        if contract.content_digest(package / "_native" / name) != digest:
            raise ValueError(f"Frozen p101 native library checksum mismatch: {name}")
    path = os.environ.get("MMC_LOCAL_CONFIG_PATH")
    if not path:
        raise ValueError("Compact C8 MemCache requires MMC_LOCAL_CONFIG_PATH")
    contract.validate_memcache_config(path)
    if manifest["scales_digest"] != scale_values_digest(scales):
        raise ValueError("Frozen scale values differ from the cache compatibility manifest")
    return manifest
