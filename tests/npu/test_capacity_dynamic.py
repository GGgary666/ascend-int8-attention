"""Reuse a B128/T768 graph across live batches, splits and padding forms."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch
from tests.support import baseline
from tests.support.attention_cases import prepare, layout, guards, metrics
from tests.support.serving_shapes import ActivationCorpus, Geometry
from tests.npu.test_capacity import oracle_checks


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    output = parser.parse_args().output
    assert not output.exists()
    baseline.load_ops()
    torch.set_num_threads(4)
    torch.npu.set_device(1)
    corpus = ActivationCorpus()
    records = []
    for ratio, maximum in [(4,8193), (128,135000)]:
        captured = dict(id=f'capacity_graph_c{ratio}', ratio=ratio, queries=[6]*128,
            lengths=[maximum]*128, capacity=768, mode='eager', draft=False, index=0)
        state = prepare(corpus,captured,'npu:1')
        entry, dev = state['original'], state['dev']
        assert entry.c8_dynamic and entry.partial.numel() == 12*8*4*520
        # Keep only fixed device storage, including the real strided KV views.
        state.pop('source'); state.pop('quant')
        initial_args = bytes(entry.args)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            entry.run()
        modes = [
            ([6]*128,[maximum]*128,'synthetic'),
            ([6]*23,[8192 if ratio==4 else 65536]*23,'negative'),
            ([5]*9,[1028 if ratio==4 else 32896]*9,'synthetic'),
            ([6]*8,[8192 if ratio==4 else 32768]*8,'negative'),
            ([1],[maximum],'synthetic'),
            ([3]*2,[maximum]*2,'negative'),
            ([1]*64,[513 if ratio==4 else 16384]*64,'synthetic'),
            ([1]*64,[3 if ratio==4 else 127]*64,'synthetic'),
            ([1]*64,[4 if ratio==4 else 128]*64,'synthetic'),
            ([1]*64,[3 if ratio==4 else 127]*64,'synthetic'),
            *[([1]*64,[length]*64,'synthetic') for length in (1,33,65,97,129,1)],
            ([1+i%9 for i in range(17)],
             [([129,513,1028,8193] if ratio==4 else [129,16384,32896,133333])[i%4] for i in range(17)],'negative'),
            ([49],[maximum],'synthetic'),
            ([6]*8,[513,maximum,129,maximum,1028,maximum,257,maximum],'holes'),
            ([1]*64,[513]*64,'holes'),
            ([2]*64,[513]*64,'holes'),
            # Equal live totals do not prove q1. One q2 request plus a
            # zero-query metadata row must still cover every token once.
            ([2]+[1]*126,[513]*127,'holes'),
            ([2]+[1]*126,[[129, ratio*513, 1, ratio*128][i%4] for i in range(127)],'holes'),
            # Cross the retained-task batch boundary (24*5 + 1 queries),
            # then change KV-kind count and request indexing in the same
            # graph. This exercises old-PV GM reuse before output finishes.
            ([1]*121,[1]*121,'synthetic'),
            ([1]*121,[ratio]*121,'negative'),
            ([1]*64,[ratio*512]*64,'holes'),
            # Same graph/workspace crosses retained L1 capacity and changes
            # live per-task extents, including empty compressed contributions.
            *[([1]*97,[[1,127,ratio*128,ratio*keys][i%4] for i in range(97)],'synthetic')
              for keys in (513,768,769,832,895,896,897,769)],
            ([1]*121,[1]*121,'negative'),
            # Change packing and key-kind decisions inside one captured
            # graph, with unequal lengths and a second macro iteration.
            ([1]*121,[1+i%(3 if ratio==4 else 6) for i in range(121)],'synthetic'),
            ([1]*121,[4 if ratio==4 else 7]*121,'negative'),
            ([1]*97,[1+i%(3 if ratio==4 else 6) for i in range(97)],'synthetic'),
            ([],[],'negative'),
            ([6]*128,[maximum]*128,'synthetic'),
        ]
        for queries,lengths,padding in modes:
            batch,live = len(queries),sum(queries)
            state['live'] = live
            dev.q.fill_(float('nan'));dev.seq_lens.zero_()
            dev.cu_q.fill_(-1);dev.cu_q[0] = 0
            for name in ('ori_block_table','cmp_block_table','cmp_indices'):
                value=getattr(dev,name)
                if value is not None:value.fill_(-1)
            if batch:
                source,oq,cq,_ = corpus.make(Geometry('replay',ratio,tuple(queries),tuple(lengths)),layer_index=0)
                quant=replace(source,ori=oq.quantize(source.ori),cmp=cq.quantize(source.cmp))
                for name in ('ori','cmp','q','cmp_indices','ori_block_table','cmp_block_table'):
                    src,dst=getattr(quant,name),getattr(dev,name)
                    if src is None:continue
                    assert src.shape[0]<=dst.shape[0],(name,src.shape,dst.shape)
                    dst[:len(src)].copy_(src)
                dev.seq_lens[:batch].copy_(quant.seq_lens)
                dev.cu_q[:batch+1].copy_(quant.cu_q)
            if padding=='synthetic' and batch<128:
                # Same synthetic-request convention as the serving runner.
                dev.cu_q[batch+1:].copy_(torch.arange(live+1,live+129-batch,
                    dtype=torch.int32,device='npu:1'))
            if padding=='holes':
                # More metadata rows than query rows, with zero-query requests
                # between live ones. Exercise the bounded two-wave planner's
                # cache of nonempty requests, not a fixed-size batch array.
                places=[i*127//(batch-1) for i in range(batch)]
                counts=[0]*128;seq=[0]*128
                dev.seq_lens.zero_()
                for b,place in enumerate(places):
                    counts[place]=queries[b];seq[place]=lengths[b]
                    for name in ('ori_block_table','cmp_block_table'):
                        getattr(dev,name)[place].copy_(getattr(quant,name)[b])
                if live == 128 and batch == 127:
                    # An inactive metadata row may have a large KV length;
                    # the per-core plan must not assign it a token or extent.
                    for place, count in enumerate(counts):
                        if not count: seq[place] = maximum
                dev.cu_q.copy_(torch.tensor([0,*torch.tensor(counts).cumsum(0).tolist()],dtype=torch.int32))
                dev.seq_lens.copy_(torch.tensor(seq,dtype=torch.int32))
            entry.workspace.fill_(0xA5);entry.partial.fill_(float('nan'))
            entry.out.fill_(float('nan'));entry.lse.fill_(float('nan'))
            for _ in range(3):graph.replay()
            actual,actual_lse=entry.out.cpu(),entry.lse.cpu()
            guards(state)
            assert bytes(entry.args)==initial_args
            assert actual.isfinite().all() and actual_lse.isfinite().all()
            error=None
            if batch:
                case=dict(ratio=ratio,queries=queries,lengths=lengths,
                          metadata_batch=128 if padding=='holes' else batch)
                n,g,s=layout(entry,dict(ratio=ratio,queries=counts,lengths=seq) if padding=='holes' else case)
                oracle_state=dict(quant=quant,quantizers=(oq,cq),live=live)
                checked=oracle_checks(oracle_state,case,n,g,s,actual,actual_lse)
                bf=source.to('npu:1')
                meta=baseline.prepare(bf,max_query_length=max(queries),max_seq_length=max(lengths))
                expected=baseline.run(bf,meta)[0][:live].cpu().float()
                error=metrics(actual[:live].float(),expected)
                assert error['relative_l2']<.04 and error['min_head_cosine']>.999,error
                del source,quant,bf,meta,expected
            row=dict(ratio=ratio,batch=batch,queries=live,padding=padding,
                capacity=768,metadata_capacity=128,graph_replays=3,args_unchanged=True,
                guards_pass=True,partial_bytes=entry.partial.numel()*4,bf16=error)
            records.append(row);print(json.dumps(row),flush=True)
        del graph,state,entry,dev
        torch.npu.empty_cache()
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(complete=True,records=records),indent=2)+'\n')


if __name__=='__main__':main()
