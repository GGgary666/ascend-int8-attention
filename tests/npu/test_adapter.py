"""Clean C8 through the real vLLM adapter, queue and cache page views."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
from unittest.mock import patch

import torch
import torch_npu  # noqa: F401
from vllm.plugins import load_general_plugins

from npu_sparse_attn_sharedkv_int8.reference import CacheQuantization, make_case, tiled_quantized_reference
from npu_sparse_attn_sharedkv_int8 import vllm_adapter as adapter


def oracle(case, oq, cq, op, **kw):
    if getattr(op, 'c8_dynamic', False):
        from tests.support.attention_cases import layout
        n,g,parts=layout(op,dict(ratio=case.ratio,queries=case.cu_q.diff().tolist(),
                               lengths=case.seq_lens.tolist()))
        kw.setdefault('joint_dense', True)
        kw.setdefault('joint_sparse', case.ratio==4 and n<=256 and g==1 and len(case.q)>=24)
    else:n,g,parts=op.tile_size,op.query_group,op.splits
    return tiled_quantized_reference(case,oq,cq,tile_size=n,query_group=g,splits=parts,**kw)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact-pages", action="store_true")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--queries", type=int, default=5)
    parser.add_argument("--length", type=int, default=260)
    args = parser.parse_args()
    assert os.environ.get("SHAREDKV_C8_DECODE", "1") == "1"
    if args.compact_pages:
        os.environ["SHAREDKV_COMPACT_PAGES"] = "1"
    else:
        os.environ.pop("SHAREDKV_COMPACT_PAGES", None)
    page_bytes = 16384 if args.compact_pages else 32768
    stride = 622592 if args.compact_pages else 32768
    offset = 16384 if args.compact_pages else 0
    torch.set_num_threads(4)
    torch.npu.set_device(1)
    device = "npu:1"
    load_general_plugins()
    cases, quantizers, scales = {}, {}, {}
    for ratio in (1, 4, 128):
        source = make_case(query_lengths=(args.queries,)*args.batch, seq_lens=(args.length,)*args.batch,
                           heads=8, ratio=ratio, topk=1024, seed=991)
        oq = CacheQuantization.calibrate([source.ori])
        cq = CacheQuantization.calibrate([source.cmp]) if source.cmp is not None else None
        cases[ratio], quantizers[ratio] = source, (oq, cq)
        scales[f"unit.r{ratio}.ori"] = oq.scale
        if cq:
            scales[f"unit.r{ratio}.cmp"] = cq.scale
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "scales.pt"
        torch.save(dict(protocol="sharedkv-c8-v1", scales=scales), path)
        os.environ["SHAREDKV_SCALES"] = str(path)
        adapter.load_scales.cache_clear()
        adapter.install()
        if os.environ.get("SHAREDKV_SKIP_SAS_METADATA") == "1":
            from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder
            buffer = torch.zeros(1024, dtype=torch.int32, device=device)
            builder = SimpleNamespace(sas_metadata_buffer=buffer)
            saved = {}
            for bound in (260, 8192):
                value = AscendDSAMetadataBuilder._build_sas_metadata(builder, saved, "unit",
                    None, None, 5, bound, None, None, 127, 0, False)
                assert value is buffer and saved["unit"] is buffer
                assert value._sharedkv_max_seq_len == bound
            print("Elided main metadata retains identity and updates the eager sequence-length bound: PASS", flush=True)
        config = SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16),
                                 cache_config=SimpleNamespace(block_size=32, cache_dtype="bfloat16"),
                                 parallel_config=SimpleNamespace(decode_context_parallel_size=1, prefill_context_parallel_size=1))
        from vllm_ascend.models.deepseek_v4.model import AscendDeepseekV4SWACache
        from vllm_ascend.models.layer.attention.layer import DSAAttention
        from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
        fake_swa = SimpleNamespace(head_dim=512, dtype=torch.bfloat16, window_size=128,
                                   block_size=32, cache_config=config.cache_config)
        swa_spec = AscendDeepseekV4SWACache.get_kv_cache_spec(fake_swa, config)
        assert swa_spec.dtype == torch.int8 and swa_spec.page_size_bytes == page_bytes
        for ratio in (4, 128):
            fake = SimpleNamespace(compress_ratio=ratio, kv_cache_dtype="bfloat16", head_size=512)
            spec = DSAAttention.get_kv_cache_spec(fake, config)
            assert spec.dtype == torch.int8 and spec.page_size_bytes == page_bytes
            assert spec.block_size == 32 * ratio
        print(f"Actual vLLM cache specs retain scheduler blocks, payload/page={page_bytes}, stride={stride}: PASS", flush=True)

        for ratio, source in cases.items():
            oq, cq = quantizers[ratio]
            dev = source.to(device)
            buffers, write_inputs = [], []
            for kind in ("ori", "cmp"):
                tensor = getattr(source, kind)
                if tensor is None:
                    continue
                raw = torch.full((tensor.shape[0] * stride,), 73, dtype=torch.int8, device=device)
                # Use the real runner's page-view function, including dtype
                # conversion and stride handling, without constructing a model.
                cache = NPUModelRunner._adjust_kv_layout(None, raw[offset:], [tensor.shape], [torch.int8], stride)[0]
                setattr(dev, kind, cache)
                updates = tensor.reshape(-1, 1, 512).to(device)
                slots = torch.arange(updates.shape[0], dtype=torch.int32, device=device)
                write_inputs.append((cache, updates, slots, kind))
                buffers.append(raw)
            impl = SimpleNamespace(swa_cache_layer=SimpleNamespace(prefix=f"unit.r{ratio}.swa_cache"),
                                   head_dim=512, turboquant=None, vllm_config=config)
            with patch("vllm_ascend.ascend_config.get_ascend_config", return_value=SimpleNamespace(enable_dsv4_kv_int8=False)):
                state = adapter.LayerRuntime(impl)
            metadata = torch.empty(1, device=device)
            metadata._sharedkv_max_seq_len = args.length
            kw = dict(ori_kv=dev.ori, cmp_kv=dev.cmp, ori_block_table=dev.ori_block_table,
                      cmp_block_table=dev.cmp_block_table, cmp_sparse_indices=dev.cmp_indices,
                      cu_seqlens_q=dev.cu_q, seqused_kv=dev.seq_lens, sinks=dev.sink,
                      metadata=metadata, cmp_ratio=ratio, ori_mask_mode=4, cmp_mask_mode=3,
                      ori_win_left=127, ori_win_right=0, softmax_scale=dev.softmax_scale,
                      layout_q="TND", layout_kv="PA_ND", return_softmax_lse=True)

            def run(query):
                for args in write_inputs:
                    state.write(*args)
                return state.attention(query, **kw)

            for multiplier in (1, -0.75):
                query = (dev.q * multiplier).contiguous()
                output, lse = run(query)
                quant = replace(source, q=source.q * multiplier, ori=oq.quantize(source.ori),
                                cmp=cq.quantize(source.cmp) if cq else None)
                op = next(reversed(state.attention_ops.values()))
                assert getattr(op,"c8_dynamic",False) == (ratio in (4,128))
                expected, expected_lse = oracle(quant, oq, cq, op, return_lse=True)
                torch.testing.assert_close(output.cpu().float(), expected.float(), atol=0.0003, rtol=0.008)
                assert lse.shape == (*query.shape[:2], 1)
                torch.testing.assert_close(lse.cpu().squeeze(-1), expected_lse, atol=2e-5, rtol=2e-5)
            if os.environ.get("SHAREDKV_HOST_QUEUE") == "1":
                # Keep producers, cache writes and attention queued together.
                # Vary source values and release temporary input references;
                # a raw-handle lookup used to hide ordering/lifetime defects.
                factors = (0.5, -0.75, 1.0, -1.25)
                expected_burst = []
                for factor in factors:
                    burst_case = replace(source, q=(source.q * factor).bfloat16(),
                        ori=oq.quantize((source.ori * factor).bfloat16()),
                        cmp=cq.quantize((source.cmp * factor).bfloat16()) if cq else None)
                    expected_burst.append(oracle(burst_case, oq, cq, op))
                stream_ids = set()
                for stream in (torch.npu.current_stream(), torch.npu.Stream(device=device)):
                    stream.wait_stream(torch.npu.current_stream())
                    outputs = []
                    with torch.npu.stream(stream):
                        stream_ids.add(adapter.stream_key(torch.device(device)))
                        for iteration in range(12):
                            factor = factors[iteration % 4]
                            for cache, updates, slots, kind in write_inputs:
                                state.write(cache, updates * factor, slots, kind)
                            outputs.append(state.attention(dev.q * factor, **kw)[0])
                        stream.synchronize()
                    for iteration, output in enumerate(outputs):
                        torch.testing.assert_close(output.cpu().float(), expected_burst[iteration % 4].float(),
                                                   atol=0.0003, rtol=0.008)
                assert len(stream_ids) == 2, "Auxiliary stream must receive a separate cache/scratch key"
                print(f"ratio={ratio}: 24 queued producer/write/attention calls on distinct streams PASS", flush=True)
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                output, lse = run(query)
            query.copy_(dev.q)
            # Reusing the Python launch object must not replace captured
            # argument values or keep all eager activation storage alive.
            state.attention((-query).contiguous(), **kw)
            graph.replay()
            quant.q = source.q
            op = state.captured_ops[-1]
            expected, expected_lse = oracle(quant, oq, cq, op, return_lse=True)
            torch.testing.assert_close(output.cpu().float(), expected.float(), atol=0.0003, rtol=0.008)
            torch.testing.assert_close(lse.cpu().squeeze(-1), expected_lse, atol=2e-5, rtol=2e-5)
            _, empty_lse = state.attention(query, **{**kw, "return_softmax_lse": False})
            assert empty_lse.shape == (0,) and empty_lse.dtype == torch.float32 and empty_lse.device == query.device
            for cache, updates, slots, kind in write_inputs:
                state.write(cache, updates[:3], slots[:3], kind)
                scale = oq if kind == "ori" else cq
                torch.testing.assert_close(cache.cpu(), scale.quantize(getattr(source, kind)), atol=0, rtol=0)
            for raw in buffers:
                assert (raw.view(-1, stride)[:, :offset].cpu() == 73).all()
                assert (raw.view(-1, stride)[:, offset + 16384:].cpu() == 73).all()
            assert all(op.case is None and op.out is None for op in state.attention_ops.values())
            print(f"ratio={ratio}: real page views, cache writes, new pointers, shared scratch and graph replay PASS", flush=True)


if __name__ == "__main__":
    main()
