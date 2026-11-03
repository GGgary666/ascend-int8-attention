"""Queued launch ordering and argument lifetime under rapid pointer rebinding."""
import argparse
from dataclasses import replace
import os
from pathlib import Path
import sys
import torch
import torch_npu  # noqa: F401

from npu_sparse_attn_sharedkv_int8.attention import PreparedAttention
from npu_sparse_attn_sharedkv_int8.cache import PreparedCacheWrite
from tests.support.attention_cases import layout
from npu_sparse_attn_sharedkv_int8.reference import CacheQuantization, make_case, tiled_quantized_reference


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=1)
    args = parser.parse_args()
    assert os.environ.get("SHAREDKV_HOST_QUEUE") == "1"
    torch.set_num_threads(4)
    device = f"npu:{args.device}"
    torch.npu.set_device(device)
    for ratio in (1, 4, 128):
        cpu = make_case(query_lengths=(6,), seq_lens=(513,), ratio=ratio, topk=1024, seed=2801)
        oq = CacheQuantization.calibrate([cpu.ori])
        cq = CacheQuantization.calibrate([cpu.cmp]) if cpu.cmp is not None else None
        cpu.ori = oq.quantize(cpu.ori)
        cpu.cmp = cq.quantize(cpu.cmp) if cq else None
        dev = cpu.to(device)
        op = PreparedAttention(dev, oq.scale.to(device), cq.scale.to(device) if cq else None)
        tile, group, splits = layout(op, dict(queries=[6], lengths=[513], ratio=ratio))
        assert getattr(op, "c8_dynamic", False) == (ratio in (4, 128))
        factors = (1.0, -0.75, 0.5, -1.25)
        expected = [tiled_quantized_reference(replace(cpu, q=(cpu.q * factor).bfloat16()), oq, cq,
                    tile_size=tile, splits=splits, query_group=group,
                    joint_dense=getattr(op, 'c8_dynamic', False)).float() for factor in factors]
        for auxiliary in (False, True):
            stream = torch.npu.Stream(device=device) if auxiliary else torch.npu.current_stream()
            stream.wait_stream(torch.npu.current_stream())
            outputs = []
            with torch.npu.stream(stream):
                for index in range(32):
                    # The producer and consumer must stay ordered without
                    # stream() draining the host queue between them.
                    query = dev.q * factors[index % 4]
                    output = torch.empty_like(query)
                    op._rebind_same_geometry(replace(dev, q=query), output=output)
                    op.run()
                    outputs.append(output)
                    op.case = None  # Exercise activation lifetime/reuse too.
                stream.synchronize()
            for index, output in enumerate(outputs):
                torch.testing.assert_close(output.cpu().float(), expected[index % 4], rtol=0.008, atol=0.0003)
            print(f"ratio={ratio} auxiliary_stream={auxiliary}: 32 queued producer/attention/rebind launches PASS", flush=True)

    cache = torch.zeros((4, 32, 1, 512), dtype=torch.int8, device=device)
    updates = torch.zeros((1, 1, 512), dtype=torch.bfloat16, device=device)
    slots = torch.zeros(1, dtype=torch.int32, device=device)
    inverse = torch.full((512,), 10.0, device=device)
    writer = PreparedCacheWrite(cache, updates, slots, inverse)
    expected = torch.zeros((128, 512), dtype=torch.int8)
    for index in range(32):
        count = index % 3 + 1
        value = (index % 19 - 9) / 10
        updates = torch.full((count, 1, 512), value, dtype=torch.bfloat16, device=device)
        slots = torch.arange(index * 3, index * 3 + count, dtype=torch.int32, device=device)
        writer._rebind_same_geometry(cache, updates, slots)
        writer.run()
        expected[index * 3:index * 3 + count] = (torch.tensor(value).bfloat16().float() * 10).round().to(torch.int8)
        writer.updates = writer.slots = None
    torch.testing.assert_close(cache.cpu().view(128, 512), expected, rtol=0, atol=0)
    print("Queued cache writer: varying row counts, pointers and source lifetimes PASS", flush=True)


if __name__ == "__main__":
    main()
