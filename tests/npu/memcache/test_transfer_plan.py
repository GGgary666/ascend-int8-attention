"""Isolated SDK byte-contract and alternating scatter coalescing experiment."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace as NS

import torch
import torch_npu  # noqa: F401

from .model_layout import make_views, model_groups
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import (
    MemcacheBackend,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.cache_compatibility import (
    canonical_digest,
    describe_block_view,
    key_info_matches,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ChunkedTokenDatabase,
    KeyMetadata,
    PoolKey,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker
from vllm_ascend.compilation.task_queue import startup_graph_task_queue

parser = argparse.ArgumentParser()
parser.add_argument("--rounds", type=int, default=20)
parser.add_argument("--device", type=int, required=True)
parser.add_argument("--participants", type=int, default=1)
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--sync-dir", type=Path, required=True)
args = parser.parse_args()
assert args.rounds >= 10
args.output.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)
torch.npu.set_device(args.device)
assert torch.npu.get_task_queue_enable() == 2
model_root = args.model
model = json.loads((model_root / "config.json").read_text())
groups = model_groups(model_root)
aliases = {}
for group in groups:
    for name in group.layer_names:
        parts = name.split(".")
        index = int(parts[2])
        if index >= model["num_hidden_layers"]:
            aliases[name] = f"mtp.{index - model['num_hidden_layers']}." + ".".join(parts[3:])
groups = [
    replace(
        group,
        layer_names=[aliases.get(name, name) for name in group.layer_names],
        kv_cache_spec=replace(
            group.kv_cache_spec,
            kv_cache_specs={
                aliases.get(name, name): spec
                for name, spec in group.kv_cache_spec.kv_cache_specs.items()
            },
        ),
    )
    for group in groups
]
n = 2 * len(groups) + 3
allocation, backing, views = make_views(groups, n, f"npu:{args.device}")
generator = torch.Generator().manual_seed(731092)
initial = torch.randint(0, 256, (backing.numel(),), generator=generator, dtype=torch.uint8)
worker = KVPoolWorker.__new__(KVPoolWorker)
worker.hf_config = NS(num_hidden_layers=model["num_hidden_layers"])
worker.num_layers = model["num_hidden_layers"]
worker._global_to_local_layer = {}
worker.kv_caches = views
worker._get_cache_block_metadata = lambda cache: describe_block_view(cache, n)
for field in (
    "group_kv_caches_base_addr",
    "group_block_len",
    "group_block_stride",
    "group_layer_cache_entry_offsets",
    "group_num_layers",
):
    setattr(worker, field, {})
sources, targets, sizes = [], [], []
expected = initial.clone()
for gid, group in enumerate(groups):
    worker._infer_cache_group_metadata(gid, group.layer_names)
    bases = worker.group_kv_caches_base_addr[gid]
    strides = worker.group_block_stride[gid]
    lengths = worker.group_block_len[gid]
    source = [base + (gid + 1) * stride for base, stride in zip(bases, strides, strict=True)]
    target = [
        base + (gid + len(groups) + 1) * stride for base, stride in zip(bases, strides, strict=True)
    ]
    for src, dst, length in zip(source, target, lengths, strict=True):
        src -= backing.data_ptr()
        dst -= backing.data_ptr()
        expected[dst : dst + length] = initial[src : src + length]
    sources.append(source)
    targets.append(target)
    sizes.append(lengths)
ranges = [(backing.data_ptr(), backing.data_ptr() + backing.numel())]
database = ChunkedTokenDatabase(
    [
        KeyMetadata("registered-plan", 0, 0, 0, gid, cache_namespace="a" * 64)
        for gid in range(len(groups))
    ],
    [g.kv_cache_spec.block_size for g in groups],
    None,
)
database.set_group_buffers(
    worker.group_kv_caches_base_addr,
    worker.group_block_len,
    worker.group_block_stride,
    registered_regions=ranges,
    num_blocks=n,
)
plans = database._full_block_transfer_plans


def describe(target, copies=1, planned=True):
    database._full_block_transfer_plans = plans if planned else {}
    addresses, lengths = [], []
    try:
        for _ in range(copies):
            for gid in range(len(groups)):
                block = gid + (len(groups) + 1 if target else 1)
                size = database.get_block_size(gid)
                addr, length, _ = database.prepare_value(0, size, [block], gid)
                addresses.append(addr)
                lengths.append(length)
    finally:
        database._full_block_transfer_plans = plans
    return addresses, lengths


merged_sources, merged_sizes = describe(False)
merged_targets, target_sizes = describe(True)
assert merged_sizes == target_sizes
assert describe(False, planned=False) == (sources, sizes)
assert describe(True, planned=False) == (targets, sizes)
assert sum(map(sum, sizes)) == sum(map(sum, merged_sizes))
assert sum(map(len, sources)) == 276 and sum(map(len, merged_sources)) == 125
backend = MemcacheBackend(
    NS(data_parallel_size=2),
    device_id=args.device,
    lazy_init=True,
    extra_config={"cache_compatibility_manifest": "coalesce-byte-experiment"},
)
backend.register_buffer([backing.data_ptr()], [backing.numel()])
nonce = str(time.time_ns())
namespace = canonical_digest({"prototype": "registered-plan-wire-order", "nonce": nonce})
owned_keys = []


def make_keys(label, copies=1):
    keys = [
        PoolKey(
            KeyMetadata("registered-plan-probe", 0, 0, 0, gid, cache_namespace=namespace),
            f"{label}-{copy}",
        ).to_string()
        for copy in range(copies)
        for gid in range(len(groups))
    ]
    owned_keys.extend(keys)
    return keys


def check_infos(keys, lengths):
    infos = backend.batch_get_key_info(keys)
    assert len(infos) == len(keys)
    assert all(key_info_matches(info, sum(row)) for info, row in zip(infos, lengths, strict=True))


def restore_initial():
    backing.copy_(initial)
    torch.npu.synchronize()


def check_bytes():
    actual = backing.cpu()
    assert torch.equal(actual, expected), "Payload, unrelated page, group or padding changed"
    return hashlib.sha256(actual.numpy().tobytes()).hexdigest()


report = dict(
    device=args.device,
    participants=args.participants,
    passed=False,
    production_modified=False,
    timing_acceptance=False,
    nonce=nonce,
    byte_order="Actual KVPoolWorker group order, real compact model specs",
    components=sum(map(len, sources)),
    merged_segments=sum(map(len, merged_sources)),
    payload_bytes=sum(map(sum, sizes)),
    groups=len(groups),
    cases=[],
    timings=[],
)
try:
    for merge_put in (False, True):
        for merge_get in (False, True):
            restore_initial()
            keys = make_keys(f"put{int(merge_put)}-get{int(merge_get)}")
            results = backend.put(
                keys, merged_sources if merge_put else sources, merged_sizes if merge_put else sizes
            )
            assert results == [0] * len(keys), results
            check_infos(keys, sizes)
            results = backend.get(
                keys, merged_targets if merge_get else targets, target_sizes if merge_get else sizes
            )
            assert results == [0] * len(keys), results
            digest = check_bytes()
            consumer = torch.empty_like(backing)
            stream = torch.npu.Stream(device=args.device)
            with torch.npu.stream(stream):
                consumer.copy_(backing)
            stream.synchronize()
            assert torch.equal(consumer.cpu(), expected)
            graph = torch.npu.NPUGraph()
            with startup_graph_task_queue(torch.npu):
                with torch.npu.graph(graph):
                    consumer.copy_(backing)
            assert torch.npu.get_task_queue_enable() == 2
            for _ in range(3):
                graph.replay()
            torch.npu.synchronize()
            assert torch.equal(consumer.cpu(), expected)
            report["cases"].append(
                dict(
                    merged_put=merge_put,
                    merged_get=merge_get,
                    whole_backing_sha256=digest,
                    streams=True,
                    graph_replay=True,
                )
            )
            print(json.dumps(report["cases"][-1]), flush=True)
    for copies in (1, 16):
        restore_initial()
        keys = make_keys(f"timing-{copies}", copies)
        assert backend.put(keys, sources * copies, sizes * copies) == [0] * len(keys)
        check_infos(keys, sizes * copies)
        samples = {"legacy": [], "registered": []}
        modes = tuple(samples)
        for trial in range(args.rounds + 4):
            order = modes[trial % 2 :] + modes[: trial % 2]
            for mode in order:
                barrier = args.sync_dir / f"{copies}-{trial}-{mode}"
                barrier.mkdir(parents=True, exist_ok=True)
                (barrier / str(args.device)).touch()
                deadline = time.monotonic() + 120
                while len(list(barrier.iterdir())) != args.participants:
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"Concurrent peers missing: {barrier}")
                    time.sleep(0.001)
                start = time.perf_counter_ns()
                transfer_addresses, transfer_sizes = describe(
                    True, copies, planned=mode == "registered"
                )
                results = backend.get(keys, transfer_addresses, transfer_sizes)
                duration = (time.perf_counter_ns() - start) / 1e6
                assert results == [0] * len(keys), results
                if trial >= 4:
                    samples[mode].append(duration)
        digest = check_bytes()
        plain, merged = (
            statistics.median(samples["legacy"]),
            statistics.median(samples["registered"]),
        )
        report["timings"].append(
            dict(
                copies=copies,
                keys=len(keys),
                payload_bytes=sum(map(sum, sizes)) * copies,
                plain_ms=samples["legacy"],
                merged_ms=samples["registered"],
                plain_median_ms=plain,
                merged_median_ms=merged,
                merged_over_plain=merged / plain,
                whole_backing_sha256=digest,
            )
        )
        print(
            json.dumps(
                {
                    k: v
                    for k, v in report["timings"][-1].items()
                    if k not in ("plain_ms", "merged_ms", "planning_ms")
                }
            ),
            flush=True,
        )
    report["passed"] = True
finally:
    report["stats"] = backend.drain_transfer_stats()
    if backend.store is not None:
        if owned_keys:
            removed = backend.store.remove_batch(owned_keys)
            report["cleanup_result"] = str(removed)
        backend.store.close()
    (args.output / f"device{args.device}.json").write_text(json.dumps(report, indent=2) + "\n")
assert report["passed"]
print("REGISTERED_PLAN_NPU_BYTE_CONTRACT_PASSED", flush=True)
