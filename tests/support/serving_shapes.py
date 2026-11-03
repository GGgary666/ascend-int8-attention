"""Standalone serving geometries populated from saved model activations.

Shape expansion resamples BF16 rows from one real layer and uses its frozen C8
scales. It does not claim to reproduce a real long-context model forward. The
full original capture is archived; these fixtures keep one real sample per layer.
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch

from npu_sparse_attn_sharedkv_int8.reference import CacheQuantization, SharedKVCase


ROOT = Path(__file__).resolve().parents[2]
MAX_MODEL_LEN = 135000
INT8_PAGE_STRIDE = 622592


@dataclass(frozen=True)
class Geometry:
    name: str
    ratio: int
    queries: tuple[int, ...]
    lengths: tuple[int, ...]
    padded_tokens: int | None = None
    draft_slots: bool = False
    graph: bool = True


class ActivationCorpus:
    def __init__(self, directory=ROOT/'tests/fixtures/activations',
                 scales=ROOT/'npu_sparse_attn_sharedkv_int8/data/scales_v1.pt'):
        self.directory = Path(directory)
        self.scales = torch.load(scales, map_location='cpu', weights_only=True)['scales']
        self.records = [json.loads(x) for x in (self.directory/'metrics.jsonl').read_text().splitlines()]
        self.records = [r for r in self.records if r['type']=='attention']
        self.pools = {}

    def layer(self, ratio, index):
        names = sorted({r['layer'] for r in self.records if r['ratio']==ratio})
        return names[index % len(names)]

    def pool(self, layer):
        if layer not in self.pools:
            rows = [r for r in self.records if r['layer']==layer]
            samples = [torch.load(self.directory/f"{layer}.call_{r['call']}.pt", weights_only=True) for r in rows]
            assert all(torch.equal(s['sink'], samples[0]['sink']) for s in samples)
            self.pools[layer] = dict(
                query=torch.stack([s['query'] for s in samples]).bfloat16(),
                original=torch.cat([s['original'] for s in samples]).bfloat16(),
                compressed=torch.cat([s['compressed'] for s in samples]).bfloat16(),
                sink=samples[0]['sink'].float(), softmax_scale=samples[0]['softmax_scale'],
                calls=[r['call'] for r in rows])
        return self.pools[layer]

    def make(self, geometry, layer_index=0, seed=461055):
        g = geometry
        assert len(g.queries)==len(g.lengths) and all(0 < q <= s <= MAX_MODEL_LEN for q,s in zip(g.queries,g.lengths))
        assert sum(g.queries) > 0
        generator = torch.Generator().manual_seed(seed)
        layer = self.layer(g.ratio, layer_index); pool = self.pool(layer)
        oq = CacheQuantization(self.scales[layer+'.ori'])
        cq = CacheQuantization(self.scales[layer+'.cmp']) if g.ratio>1 else None
        batch = len(g.queries); tokens = sum(g.queries)

        def cache(kind, ranges, table_width):
            page_count = sum(len(r) for r in ranges)
            # One safe page also handles the very first token (no C4 keys yet).
            order = torch.randperm(max(1,page_count), generator=generator, dtype=torch.int32)
            table = torch.zeros((batch,table_width),dtype=torch.int32)
            offset = 0
            for b, indices in enumerate(ranges):
                if indices:
                    table[b, list(indices)] = order[offset:offset+len(indices)]
                    offset += len(indices)
            values = pool[kind]
            ids = torch.randint(len(values), (len(order)*32,), generator=generator)
            return values[ids].view(len(order),32,1,512), table

        # Keep only original pages touched by these queries and the 128-token
        # window, as a real sliding-window allocator does. Logical table rows
        # still have the model's full 4219-column capacity.
        ori_ranges = [range(max(0,s-q-127)//32, (s-1)//32+1) for q,s in zip(g.queries,g.lengths)]
        ori, ot = cache('original', ori_ranges, math.ceil(MAX_MODEL_LEN/32))
        cmp = ct = indices = None
        if cq:
            cmp_ranges = [range(math.ceil((s//g.ratio)/32)) for s in g.lengths]
            cmp, ct = cache('compressed', cmp_ranges, math.ceil(MAX_MODEL_LEN/(g.ratio*32)))
        if g.ratio==4:
            indices = torch.full((tokens,1,1024),-1,dtype=torch.int32)
        original_slots = torch.full((tokens,1,256),-1,dtype=torch.int32) if g.draft_slots else None
        offset = 0
        for b,(query,length) in enumerate(zip(g.queries,g.lengths)):
            for local in range(query):
                position = length-query+local
                if indices is not None:
                    count = (position+1)//4
                    selected = torch.randperm(count,generator=generator,dtype=torch.int32)[:1024]
                    indices[offset+local,0,:len(selected)] = selected
                if original_slots is not None:
                    logical = torch.arange(max(0,position+1-128),length,dtype=torch.int32)
                    assert len(logical)<=256
                    original_slots[offset+local,0,:len(logical)] = ot[b,logical.long()//32]*32+logical%32
            offset += query
        qids = torch.randint(len(pool['query']), (tokens,), generator=generator)
        q = pool['query'][qids]
        cu = torch.tensor([0,*torch.tensor(g.queries).cumsum(0).tolist()],dtype=torch.int32)
        case = SharedKVCase(q=q,ori=ori,ori_block_table=ot,cmp=cmp,cmp_block_table=ct,
            cu_q=cu,seq_lens=torch.tensor(g.lengths,dtype=torch.int32),sink=pool['sink'],
            cmp_indices=indices,ori_indices=original_slots,ratio=g.ratio,window=128,
            softmax_scale=pool['softmax_scale'])
        return case,oq,cq,dict(layer=layer,source_calls=pool['calls'],seed=seed,
            data_kind='geometry-expanded real BF16 activation rows; frozen serving C8 scales',
            caveat='Row resampling does not preserve complete real long-context Q/KV correlations.')

