"""Default routing, real-row arithmetic, compatibility paths and optional timing."""
import argparse
import copy
import ctypes
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import statistics
import torch

from npu_sparse_attn_sharedkv_int8.attention import PreparedAttention
from npu_sparse_attn_sharedkv_int8 import c8_attention
from npu_sparse_attn_sharedkv_int8.reference import tiled_quantized_reference
from tests.support import baseline
from tests.support.attention_cases import prepare, layout, guards, metrics
from tests.support.serving_shapes import ActivationCorpus


def cases():
    rows = []
    for ratio in (4, 128):
        for queries, lengths, capacity, mode in (
            ([6], [1024], 48, 'graph'),
            ([6]*4, [8192]*4, 48, 'graph'),
            ([6]*8, [135000]*8, 48, 'graph'),
            ([1, 2, 3, 4, 5, 6], [129, 1028, 8193, 32769, 65537, 133333], 21, 'eager'),
            ([1], [129], 1, 'eager'),
            ([4], [257], 4, 'eager'),
            ([1, 2, 3, 4, 5, 6, 4, 6], [129, 1027, 4097, 8193, 32769, 65537, 131071, 133333], 31, 'eager'),
            ([5]*8, [8192]*8, 40, 'eager'),
        ):
            rows.append(dict(id=f'c{ratio}_{len(rows)}', ratio=ratio, queries=queries,
                             lengths=lengths, capacity=capacity, mode=mode, draft=False))
    rows.extend([
        dict(id='draft_slots', ratio=1, queries=[5]*4, lengths=[8192]*4, capacity=20, mode='eager', draft=True),
        dict(id='c4_capacity96', ratio=4, queries=[12]*8, lengths=[8192]*8, capacity=96, mode='eager', draft=False),
        dict(id='c128_prefill', ratio=128, queries=[64], lengths=[4096], capacity=64, mode='eager', draft=False),
    ])
    return rows


def compare_graphs(entry, reference_library, depth, golden):
    control = copy.copy(entry)
    control.native = None
    suffix = 'sparse_small' if entry.variant == 22 else ('sparse' if entry.variant == 20 else 'dense')
    control.launch = getattr(reference_library, 'launch_sharedkv_c8_' + suffix)
    control.launch.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(type(entry.args))]
    control.launch.restype = None
    assert control.args is entry.args  # All inputs, outputs and scratch share addresses.
    entry.run(); expected = entry.out.clone(); expected_lse = entry.lse.clone()
    control.run(); torch.npu.synchronize()
    bitexact = torch.equal(entry.out, expected) and torch.equal(entry.lse, expected_lse)
    frozen_error = metrics(entry.out[:len(golden)].cpu().float(), golden)
    assert frozen_error['relative_l2'] < .04 and frozen_error['min_head_cosine'] > .999
    assert entry.lse.isfinite().all()
    graphs = {}
    for name, op in [('frozen', control), ('default', entry)]:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            for _ in range(depth): op.run()
        graphs[name] = graph
    for _ in range(3):
        for graph in graphs.values(): graph.replay()
    torch.npu.synchronize()
    samples = {name: [] for name in graphs}
    for round_index in range(9):
        order = list(graphs) if round_index % 2 == 0 else list(reversed(graphs))
        for name in order:
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            start.record(); graphs[name].replay(); end.record(); end.synchronize()
            samples[name].append(start.elapsed_time(end)*1000/depth)
    medians = {name: statistics.median(values) for name, values in samples.items()}
    return dict(depth=depth, samples_us=samples, median_us=medians,
                ratio=medians['default']/medians['frozen'], bitexact=bitexact, frozen_bf16=frozen_error, shared_addresses=True)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('validation/attention.json'))
    parser.add_argument('--reference-library', type=Path, help='Optional frozen V65 control; never a runtime dependency')
    parser.add_argument('--depth', type=int, default=256)
    args = parser.parse_args()
    assert not args.output.exists()
    assert 'SHAREDKV_C8_DECODE' not in os.environ, 'Exercise the actual unset/default policy'
    baseline.load_ops()
    torch.set_num_threads(4); torch.npu.set_device(1)
    corpus = ActivationCorpus(); records = []
    frozen = ctypes.CDLL(str(args.reference_library.resolve())) if args.reference_library else None
    for case in cases():
        state = prepare(corpus, case, 'npu:1'); entry = state['original']
        expected_route = case['ratio'] in (4, 128)
        assert getattr(entry, 'c8_dynamic', False) == expected_route
        n, group, splits = layout(entry, case)
        oracle, oracle_lse = tiled_quantized_reference(state['quant'], *state['quantizers'],
            tile_size=n, query_group=group, splits=splits, return_lse=True, joint_dense=expected_route,
            joint_sparse=expected_route and case['ratio']==4 and n<=256 and group==1 and state['live']>=24)
        entry.run(); actual, actual_lse = entry.out.cpu(), entry.lse.cpu(); guards(state)
        live = state['live']
        torch.testing.assert_close(actual[:live].float(), oracle.float(), rtol=.008, atol=.0003)
        torch.testing.assert_close(actual_lse[:live], oracle_lse, rtol=2e-5, atol=2e-5)
        assert actual.isfinite().all() and actual_lse.isfinite().all()
        bf = replace(state['dev'], ori=state['source'].ori.to('npu:1'),
                     cmp=state['source'].cmp.to('npu:1') if state['source'].cmp is not None else None)
        meta = baseline.prepare(bf, max_query_length=max(case['queries']), max_seq_length=max(case['lengths']))
        golden = baseline.run(bf, meta)[0][:live].cpu().float()
        error = metrics(actual[:live].float(), golden)
        assert error['relative_l2'] < .04 and error['min_head_cosine'] > .999, error
        record = dict(case=case, v65=expected_route, oracle_pass=True, guards_pass=True, bf16=error)
        if frozen is not None and expected_route and case['capacity']<=48 and len(case['queries'])<=8:
            record['timing'] = compare_graphs(entry, frozen, args.depth, golden)
        # Explicit rollback remains available without modifying global defaults.
        legacy = PreparedAttention(state['dev'], *[q.scale.to('npu:1') if q else None for q in state['quantizers']],
                                   implementation='fallback')
        assert not getattr(legacy, 'c8_dynamic', False)
        legacy.run(); assert legacy.out.cpu().isfinite().all()
        records.append(record); print(json.dumps(record), flush=True)
        del state, entry, legacy, bf, meta, oracle, oracle_lse
        torch.npu.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(complete=True, records=records,
        default_policy='unset SHAREDKV_C8_DECODE selects V65 when supported',
        device_sha256=hashlib.sha256(Path('build/libsharedkv_c8.so').read_bytes()).hexdigest(),
        reference_library=str(args.reference_library) if args.reference_library else None), indent=2) + '\n')


if __name__ == '__main__':
    main()
