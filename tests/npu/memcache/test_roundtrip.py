#!/usr/bin/env python3
"""Whole-model compact scatter/gather, guards, streams and fresh-client GET."""

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace as NS
import torch
from .model_layout import model_groups, make_views
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import (
    MemcacheBackend,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import KeyMetadata, PoolKey
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.cache_compatibility import (
    describe_block_view,
    canonical_digest,
    key_info_matches,
)

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--mode", choices=["write", "read"], required=True)
p.add_argument("--device", type=int, default=0)
p.add_argument("--model", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
root = a.output.resolve()
root.mkdir(parents=True, exist_ok=True)
model = a.model
torch.npu.set_device(a.device)
torch.set_num_threads(4)
groups = model_groups(model)
n = 2 * len(groups) + 3
allocation, backing, views = make_views(groups, n, f"npu:{a.device}")
stream = torch.npu.Stream(device=a.device)
with torch.npu.stream(stream):
    for gid, group in enumerate(groups):
        for name in group.layer_names:
            for ci, view in enumerate(views[name]):
                view[gid + 1].fill_((gid + ci) % 13 + 1)
    event = torch.npu.Event()
    event.record()
event.synchronize()  # same fence used before saving an INT8 writer's pages.
expected = backing.cpu().clone()
keys, addrs, sizes, source_bytes = [], [], [], []
record_path = root / "npu-byte-objects.json"
if a.mode == "write":
    record = {"nonce": str(time.time_ns())}
    record_path.write_text(json.dumps(record))
else:
    record = json.loads(record_path.read_text())
namespace = canonical_digest({"test": "real-model-compact", "nonce": record["nonce"]})
for gid, group in enumerate(groups):
    key = PoolKey(
        KeyMetadata("byte-contract", 0, 0, 0, gid, cache_namespace=namespace), "fixed-token-hash"
    ).to_string()
    ptrs, lens = [], []
    # Match the worker's physical layer order and ordered tensor components.
    names = sorted(
        group.layer_names, key=lambda name: (int(name.split("layers.")[1].split(".")[0]), name)
    )
    for name in names:
        for view in views[name]:
            length, stride, _, _ = describe_block_view(view, n)
            source = gid + 1
            target = gid + len(groups) + 1
            ptrs.append(view.data_ptr() + (source if a.mode == "write" else target) * stride)
            lens.append(length)
            if a.mode == "read":
                start = view.data_ptr() - backing.data_ptr()
                expected[start + target * stride : start + target * stride + length] = expected[
                    start + source * stride : start + source * stride + length
                ].clone()
    keys.append(key)
    addrs.append(ptrs)
    sizes.append(lens)
backend = MemcacheBackend(
    NS(data_parallel_size=2),
    device_id=a.device,
    lazy_init=True,
    extra_config={"cache_compatibility_manifest": "byte-test"},
)
backend.register_buffer([backing.data_ptr()], [backing.numel()])
started = time.perf_counter()
if a.mode == "write":
    results = backend.put(keys, addrs, sizes)
else:
    # This is intentionally the first operation on a new process's backend.
    assert not backend._store_initialized
    results = backend.get(keys, addrs, sizes)
assert results == [0] * len(keys), results
infos = backend.batch_get_key_info(keys)
assert len(infos) == len(keys) and all(
    key_info_matches(info, sum(lengths)) for info, lengths in zip(infos, sizes, strict=True)
)
if a.mode == "read":
    assert torch.equal(backing.cpu(), expected), (
        "GET changed unrelated pages, padding, another group, or payload"
    )
    # Consume restored values on a second stream, then in captured/replayed NPU work.
    restored = next(iter(views.values()))[0]
    consumer = torch.empty_like(restored)
    other = torch.npu.Stream(device=a.device)
    with torch.npu.stream(other):
        consumer.copy_(restored)
    other.synchronize()
    assert torch.equal(consumer.cpu(), restored.cpu())
    graph = torch.npu.NPUGraph()
    from vllm_ascend.compilation.task_queue import startup_graph_task_queue

    with startup_graph_task_queue(torch.npu):
        with torch.npu.graph(graph):
            consumer.copy_(restored)
    for _ in range(3):
        graph.replay()
    torch.npu.synchronize()
    assert torch.equal(consumer.cpu(), restored.cpu())
    backend.store.remove_batch(keys)
result = dict(
    passed=True,
    mode=a.mode,
    device=a.device,
    groups=len(groups),
    components=sum(map(len, addrs)),
    payload=sum(map(sum, sizes)),
    seconds=time.perf_counter() - started,
    first_get_without_put=a.mode == "read",
    host_queue=os.environ.get("SHAREDKV_HOST_QUEUE"),
    stats=backend.drain_transfer_stats(),
)
(root / f"npu-bytes-{a.mode}.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
backend.store.close()
