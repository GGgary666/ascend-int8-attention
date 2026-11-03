"""One captured graph crosses live-batch/length plans, with poisoned padding."""
from dataclasses import replace
import argparse
import copy
import json
from pathlib import Path
import sys

import torch
from tests.support import baseline
from tests.support.attention_cases import metrics
from tests.support.attention_cases import prepare, layout, guards
from npu_sparse_attn_sharedkv_int8.reference import tiled_quantized_reference
from tests.support.serving_shapes import ActivationCorpus, Geometry


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('validation/dynamic.json'))
    output=parser.parse_args().output
    assert not output.exists()
    torch.set_num_threads(4)
    torch.npu.set_device(1)
    baseline.load_ops()
    corpus = ActivationCorpus()
    records = []
    for ratio,maximum,lengths in [(4,8193,[1024,1027,1028,8192]),
                                  (128,135000,[16384,32768,32896,65536])]:
        c = dict(id=f'dynamic_c{ratio}',ratio=ratio,queries=[6]*8,lengths=[maximum]*8,
            capacity=48,mode='graph',draft=False,index=0,cu_q=list(range(0,49,6)),seq_lens=[maximum]*8)
        state = prepare(corpus,c,'npu:1')
        entry = state['entries']['latest']
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            entry.run()
        # No new graph, launcher or argument values after capture.
        states = [(8,lengths[0]),(4,lengths[1]),(1,lengths[2]),(6,lengths[3]),(0,maximum),(8,maximum)]
        states.append((5,[1024,8192,1028,4097,777] if ratio==4 else [12289,133333,65536,9001,53129]))
        for batch,length in states:
            sequence_lengths=length if isinstance(length,list) else [length]*batch
            dev=state['dev'];dev.seq_lens.zero_();dev.q.fill_(float('nan'))
            if batch:
                g=Geometry('dynamic',ratio,(6,)*batch,tuple(sequence_lengths))
                source,oq,cq,_=corpus.make(g,layer_index=0)
                quant=replace(source,ori=oq.quantize(source.ori),cmp=cq.quantize(source.cmp))
                for name in ('ori','cmp','q','cmp_indices','ori_block_table','cmp_block_table'):
                    src=getattr(quant,name);dst=getattr(dev,name)
                    if src is None:continue
                    assert src.shape[0]<=dst.shape[0],(name,src.shape,dst.shape)
                    dst[:len(src)].copy_(src)
                dev.seq_lens[:batch].copy_(torch.tensor(sequence_lengths,dtype=torch.int32))
                bf=replace(dev,ori=source.ori.to('npu:1'),cmp=source.cmp.to('npu:1'))
                meta=baseline.prepare(bf,max_query_length=6,max_seq_length=max(sequence_lengths))
                expected=baseline.run(bf,meta)[0][:batch*6].cpu().float()
                n,gp,s=layout(entry,dict(queries=[6]*batch,lengths=sequence_lengths,ratio=ratio))
                oracle,lse=tiled_quantized_reference(quant,oq,cq,tile_size=n,query_group=gp,splits=s,
                    return_lse=True,residual_quantization=True,
                    joint_dense=getattr(entry, 'c8_dynamic', False))
            entry.workspace.fill_(0xA5);entry.partial.fill_(float('nan'))
            entry.out.fill_(float('nan'));entry.lse.fill_(float('nan'))
            for _ in range(3):graph.replay()
            actual=entry.out.cpu();actual_lse=entry.lse.cpu();guards(state)
            assert actual.isfinite().all() and actual_lse.isfinite().all()
            assert torch.count_nonzero(actual[batch*6:])==0
            assert torch.count_nonzero(actual_lse[batch*6:])==0
            err=None
            if batch:
                torch.testing.assert_close(actual[:batch*6].float(),oracle.float(),rtol=.008,atol=.0003)
                torch.testing.assert_close(actual_lse[:batch*6],lse,rtol=2e-5,atol=2e-5)
                err=metrics(actual[:batch*6].float(),expected)
                assert err['relative_l2']<.04 and err['min_head_cosine']>.999
            records.append(dict(ratio=ratio,batch=batch,length=length,graph_reused=True,
                poisoned_padding_scratch=True,guards_pass=True,bf16=err))
            print(records[-1],flush=True)
        del graph,state,entry
        torch.npu.empty_cache()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(complete=True,records=records),indent=2)+'\n')


if __name__=='__main__':main()
