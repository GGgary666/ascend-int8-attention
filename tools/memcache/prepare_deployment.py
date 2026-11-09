"""Generate an isolated two-node DP2/TP16 MemCache configuration without starting services."""

import argparse
import json
from pathlib import Path


def prepare(args):
    for host in (args.meta_host, args.local_ip, args.dp_address):
        if not host or any(c.isspace() for c in host) or any(c in host for c in "\r\n=/#"):
            raise ValueError("Provide a single host/address without URL syntax")
    out = args.output.resolve()
    model = args.model.resolve(strict=True)
    manifest = args.manifest.resolve(strict=True)
    framework = args.framework.resolve(strict=True)
    plugin = args.plugin.resolve(strict=True)
    out.mkdir(parents=True, exist_ok=False)
    common = (
        f"ock.mmc.meta_service_url = tcp://{args.meta_host}:{args.meta_port}\n"
        f"ock.mmc.local_service.config_store_url = tcp://{args.meta_host}:{args.config_port}\n"
        "ock.mmc.log_level = info\nock.mmc.local_service.world_size = 256\n"
        "ock.mmc.local_service.protocol = device_sdma\n"
        "ock.mmc.local_service.max.dram.size = 1024GB\n"
        "ock.mmc.local_service.storage.enabled = false\n"
    )
    for role, size in (("client", 0), ("pool", 256)):
        (out / f"mmc-{role}.conf").write_text(
            common + f"ock.mmc.local_service.dram.size = {size}GB\n"
        )
    (out / "mmc-meta.conf").write_text(
        f"ock.mmc.meta_service_url = tcp://{args.meta_host}:{args.meta_port}\n"
        f"ock.mmc.meta_service.config_store_url = tcp://{args.meta_host}:{args.config_port}\n"
        f"ock.mmc.meta_service.metrics_url = http://{args.meta_host}:{args.metrics_port}\n"
        "ock.mmc.log_level = info\nock.mmc.evict_threshold_high = 70\nock.mmc.evict_threshold_low = 60\n"
    )
    custom = framework / "vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
    environment = dict(
        VLLM_USE_V2_MODEL_RUNNER="0",
        VLLM_ENGINE_READY_TIMEOUT_S="7200",
        ASCEND_VISIBLE_DEVICES=",".join(map(str, range(16))),
        LOCAL_WORLD_SIZE="16",
        HCCL_IF_IP=args.local_ip,
        GLOO_SOCKET_IFNAME=args.interface,
        TP_SOCKET_IFNAME=args.interface,
        HCCL_SOCKET_IFNAME=args.interface,
        HCCL_OP_EXPANSION_MODE="AIV",
        HCCL_BUFFSIZE="1024",
        HCCL_CONNECT_TIMEOUT="7200",
        ASCEND_CONNECT_TIMEOUT="10000",
        ASCEND_TRANSFER_TIMEOUT="10000",
        VLLM_RPC_TIMEOUT="1800000",
        PYTORCH_NPU_ALLOC_CONF="expandable_segments:True",
        OMP_PROC_BIND="false",
        OMP_NUM_THREADS="10",
        TASK_QUEUE_ENABLE="2",
        VLLM_PREFIX_CACHE_RETENTION_INTERVAL="4096",
        PYTHONHASHSEED="0",
        VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT="480",
        VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="1800",
        MMC_LOCAL_CONFIG_PATH=str(out / "mmc-client.conf"),
        VLLM_LOGGING_LEVEL="INFO",
        ASCEND_CUSTOM_OPP_PATH=str(custom),
        SHAREDKV_INT8_MODE="int8",
        SHAREDKV_COMPACT_PAGES="1",
        SHAREDKV_HOST_QUEUE="1",
        SHAREDKV_SKIP_SAS_METADATA="1",
        SHAREDKV_SCALES=str(plugin / "npu_sparse_attn_sharedkv_int8/data/scales_v1.pt"),
        VLLM_CACHE_ROOT=str(out / "compiler-cache/vllm"),
        TORCHINDUCTOR_CACHE_DIR=str(out / "compiler-cache/inductor"),
    )
    if args.validation_endpoints:
        environment["VLLM_SERVER_DEV_MODE"] = "1"
    connector = dict(
        kv_connector="AscendStoreConnector",
        kv_role="kv_both",
        kv_load_failure_policy="fail",
        kv_connector_extra_config=dict(
            backend="memcache",
            use_layerwise=False,
            load_async=False,
            cache_compatibility_manifest=str(manifest),
            cache_audit_dir=str(out / "worker-audits"),
        ),
    )
    command = [
        "serve",
        str(model),
        "--host",
        args.listen_host,
        "--port",
        str(args.port),
        "--data-parallel-address",
        args.dp_address,
        "--data-parallel-rpc-port",
        str(args.rpc_port),
        "--data-parallel-size",
        "2",
        "--data-parallel-size-local",
        "1",
        "--tensor-parallel-size",
        "16",
        "--enable-expert-parallel",
        "--served-model-name",
        "DeepSeek-V4-Pro",
        "--max-model-len",
        "135000",
        "--max-num-batched-tokens",
        "8192",
        "--max-num-seqs",
        "16",
        "--gpu-memory-utilization",
        "0.9",
        "--block-size",
        "32",
        "--enable-prefix-caching",
        "--enable-prompt-tokens-details",
        "--tokenizer-mode",
        "deepseek_v4",
        "--tool-call-parser",
        "deepseek_v4",
        "--enable-auto-tool-choice",
        "--reasoning-parser",
        "deepseek_v4",
        "--attention_config.indexer_kv_dtype",
        "int8",
        "--quantization",
        "ascend",
        "--model-loader-extra-config",
        json.dumps(dict(enable_multithread_load=True, num_threads=128)),
        "--speculative-config",
        json.dumps(dict(num_speculative_tokens=5, method="dspark", enforce_eager=True)),
        "--compilation-config",
        json.dumps(dict(cudagraph_mode="FULL_DECODE_ONLY")),
        "--additional-config",
        json.dumps(
            dict(
                ascend_compilation_config=dict(
                    enable_npugraph_ex=True, enable_static_kernel=True, enable_super_kernel=True
                ),
                enable_cpu_binding=True,
                enable_flashcomm1=True,
                multistream_overlap_shared_expert=True,
            )
        ),
        "--kv-transfer-config",
        json.dumps(connector),
    ]
    if args.node_rank:
        command.extend(["--headless", "--data-parallel-start-rank", "1"])
    value = dict(
        node_rank=args.node_rank,
        model=str(model),
        manifest=str(manifest),
        framework=str(framework),
        plugin=str(plugin),
        environment=environment,
        pythonpath_prepend=[str(plugin), str(framework)],
        librarypath_prepend=[str(custom / "op_api/lib"), "/usr/local/lib"],
        command=command,
        working_directory=str(out / "generated-kernels"),
    )
    (out / "deployment.json").write_text(json.dumps(value, indent=2) + "\n")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "manifest", "framework", "plugin", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("meta-host", "local-ip", "dp-address", "interface"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--node-rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--listen-host", default="localhost")
    for name, default in [
        ("meta-port", 15000),
        ("config-port", 16000),
        ("metrics-port", 17000),
        ("port", 18000),
        ("rpc-port", 14399),
    ]:
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--validation-endpoints", action="store_true")
    print(prepare(parser.parse_args()))
