"""Reject incomplete identities and unsafe deployment inputs before touching services."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools/memcache" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_files(path):
    path.mkdir()
    for name in ("config.json", "quant_model_description.json"):
        (path / name).write_text("{}")
    (path / "weights.safetensors").write_bytes(b"weight-content")
    (path / "weights.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.weight": "weights.safetensors"}})
    )
    return path


def test_model_hash_ignores_timestamps_but_rejects_changed_weight(tmp_path):
    import os

    model = model_files(tmp_path / "model")
    freeze = load("freeze_model").freeze
    first = freeze(model)
    os.utime(model / "weights.safetensors", (1, 1))
    assert freeze(model) == first
    (model / "weights.safetensors").write_bytes(b"other--content")
    assert freeze(model)["model_digest"] != first["model_digest"]


@pytest.mark.parametrize("name", ["../outside.safetensors", "/outside.safetensors"])
def test_weight_index_cannot_select_absolute_or_parent_paths(tmp_path, name):
    model = model_files(tmp_path / "model")
    (model / "weights.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": name}}))
    with pytest.raises(ValueError, match="non-relative"):
        load("freeze_model").freeze(model)


def test_prepare_keeps_client_and_pool_separate_and_refuses_overwrite(tmp_path):
    model = model_files(tmp_path / "model")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    args = SimpleNamespace(
        model=model,
        manifest=manifest,
        framework=tmp_path,
        plugin=tmp_path,
        output=tmp_path / "deployment",
        meta_host="meta-host",
        local_ip="node-host",
        dp_address="primary-host",
        interface="eth-test",
        node_rank=1,
        listen_host="localhost",
        port=18000,
        rpc_port=14399,
        meta_port=15000,
        config_port=16000,
        metrics_port=17000,
        early_shared_mlp=False,
        validation_endpoints=False,
    )
    prepare = load("prepare_deployment").prepare
    output = prepare(args)
    assert "dram.size = 0GB\n" in (output / "mmc-client.conf").read_text()
    assert "dram.size = 256GB\n" in (output / "mmc-pool.conf").read_text()
    deployment = json.loads((output / "deployment.json").read_text())
    command = deployment["command"]
    assert command[command.index("--data-parallel-start-rank") + 1] == "1"
    connector = json.loads(command[command.index("--kv-transfer-config") + 1])
    assert connector["kv_load_failure_policy"] == "fail"
    assert connector["kv_connector_extra_config"]["load_async"] is False
    before = (output / "deployment.json").read_bytes()
    with pytest.raises(FileExistsError):
        prepare(args)
    assert (output / "deployment.json").read_bytes() == before


def identities():
    preflights, workers = [], []
    for dp in (0, 1):
        preflights.append(
            dict(
                node_rank=dp,
                passed=True,
                model_content_verified=True,
                packages={},
                model_digest="model",
                scales_digest="scale",
                identity_digest="identity",
                protocol="c8",
                patch_sources_sha256={},
                plugin_sources_sha256={},
                native_sha256={"library": "native"},
                driver_metadata_sha256={},
            )
        )
        workers.append(
            dict(
                node_rank=dp,
                passed=True,
                source_host_freshness_checked=True,
                workers=[
                    dict(
                        dp=dp,
                        tp=tp,
                        record=dict(
                            namespace="namespace",
                            protocol="c8",
                            native_sha256={"library": "native"},
                        ),
                    )
                    for tp in range(16)
                ],
            )
        )
    return preflights, workers


@pytest.mark.parametrize(
    "corruption", ["missing_worker", "wrong_namespace", "stale", "unhashed_weights", "wrong_scale"]
)
def test_two_node_acceptance_rejects_incomplete_or_inconsistent_proof(corruption):
    compare = load("compare_nodes").compare
    preflights, workers = identities()
    assert compare(preflights, workers)["workers"] == 32
    if corruption == "missing_worker":
        workers[1]["workers"].pop()
    elif corruption == "wrong_namespace":
        workers[1]["workers"][15]["record"]["namespace"] = "other"
    elif corruption == "stale":
        workers[1]["source_host_freshness_checked"] = False
    elif corruption == "unhashed_weights":
        preflights[1]["model_content_verified"] = False
    elif corruption == "wrong_scale":
        preflights[1]["scales_digest"] = "other"
    with pytest.raises(ValueError):
        compare(preflights, workers)
