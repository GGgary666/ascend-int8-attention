#!/usr/bin/env python3
"""GET immediately followed by an auxiliary-stream or graph attention read."""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
import torch
from tests.support.attention_cases import prepare, layout
from tests.support.serving_shapes import ActivationCorpus
from npu_sparse_attn_sharedkv_int8.reference import tiled_quantized_reference
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import (
    MemcacheBackend,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--device", type=int, default=1)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
torch.npu.set_device(args.device)
torch.set_num_threads(4)
corpus = ActivationCorpus()
records = []
nonce = str(time.time_ns())
for ratio in (1, 4, 128):
    for mode in ("auxiliary", "graph"):
        case = dict(
            id=f"immediate-{ratio}-{mode}",
            ratio=ratio,
            queries=[6],
            lengths=[1024],
            capacity=6,
            mode="eager",
            draft=ratio == 1,
        )
        state = prepare(corpus, case, f"npu:{args.device}")
        entry = state["original"]
        n, g, splits = layout(entry, case)
        expected, lse = tiled_quantized_reference(
            state["quant"],
            *state["quantizers"],
            tile_size=n,
            query_group=g,
            splits=splits,
            return_lse=True,
            joint_dense=getattr(entry, "c8_dynamic", False),
            joint_sparse=ratio == 4 and n <= 256 and g == 1 and state["live"] >= 24,
        )
        backend = MemcacheBackend(
            NS(data_parallel_size=2),
            device_id=args.device,
            lazy_init=True,
            extra_config={"cache_compatibility_manifest": "byte-order-test"},
        )
        retained = []
        replacements = {}
        keys = []
        srcs = []
        dsts = []
        sizes = []
        checks = []
        registered_ptrs = []
        registered_sizes = []
        for kind in ("ori", "cmp"):
            original = getattr(state["dev"], kind)
            if original is None:
                continue
            stride = original.stride(0)
            size = original.shape[0] * stride
            pair = []
            for _ in range(2):
                allocation = torch.full(
                    (size + 2 * 1024 * 1024,), 53, dtype=torch.int8, device=original.device
                )
                shift = (-allocation.data_ptr()) % (2 * 1024 * 1024)
                backing = allocation[shift : shift + size]
                view = backing.as_strided(original.shape, original.stride())
                retained.append(allocation)
                pair.append((backing, view))
            (source_raw, source), (target_raw, target) = pair
            source.copy_(original)
            event = torch.npu.Event()
            event.record()
            event.synchronize()
            registered_ptrs.extend([source_raw.data_ptr(), target_raw.data_ptr()])
            registered_sizes.extend([size, size])
            length = original[0].numel()
            for page in range(original.shape[0]):
                keys.append(f"{nonce}-{ratio}-{mode}-{kind}-{page}")
                srcs.append([source.data_ptr() + page * stride])
                dsts.append([target.data_ptr() + page * stride])
                sizes.append([length])
            replacements[kind] = target
            checks.append((target, original, target_raw, length, stride))
        backend.register_buffer(registered_ptrs, registered_sizes)
        assert backend.put(keys, srcs, sizes) == [0] * len(keys)
        dev = replace(state["dev"], **replacements)
        entry._rebind_same_geometry(dev)
        stream = torch.npu.Stream(device=args.device)
        graph = None
        if mode == "graph":
            from vllm_ascend.compilation.task_queue import startup_graph_task_queue

            with startup_graph_task_queue(torch.npu):
                with torch.npu.graph(graph := torch.npu.NPUGraph()):
                    entry.run()
        torch.npu.synchronize()  # Setup/warm-up fence occurs BEFORE the measured GET.
        assert backend.get(keys, dsts, sizes) == [0] * len(keys)
        # No CPU tensor readback, metadata RPC or device synchronization here.
        if graph is not None:
            graph.replay()
        else:
            with torch.npu.stream(stream):
                entry.run()
        if graph is not None:
            torch.npu.synchronize()
        else:
            stream.synchronize()
        torch.testing.assert_close(
            entry.out[: state["live"]].cpu().float(), expected.float(), rtol=0.008, atol=0.0003
        )
        torch.testing.assert_close(entry.lse[: state["live"]].cpu(), lse, rtol=2e-5, atol=2e-5)
        for target, original, raw, length, stride in checks:
            assert torch.equal(target.cpu(), original.cpu())
            assert (raw.view(-1, stride)[:, length:].cpu() == 53).all()
        backend.store.remove_batch(keys)
        backend.store.close()
        records.append(
            dict(
                ratio=ratio,
                mode=mode,
                passed=True,
                no_intervening_readback=True,
                payload=sum(map(sum, sizes)),
            )
        )
        print(json.dumps(records[-1]), flush=True)
        del (
            state,
            entry,
            dev,
            backend,
            retained,
            replacements,
            checks,
            source,
            target,
            source_raw,
            target_raw,
            allocation,
            backing,
            view,
            original,
        )
        torch.npu.empty_cache()
(args.output / "npu-immediate-consumer.json").write_text(
    json.dumps(dict(passed=True, records=records), indent=2) + "\n"
)
