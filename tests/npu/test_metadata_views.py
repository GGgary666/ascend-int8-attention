"""Full attention with4B/60B metadata views and multi-chunk summaries."""
import argparse
import json
from pathlib import Path

import torch
from tests.support import baseline
from tests.support.attention_cases import prepare, guards, layout, metrics
from tests.support.serving_shapes import ActivationCorpus
from tests.npu.test_capacity import oracle_checks


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    output=parser.parse_args().output;assert not output.exists()
    baseline.load_ops();torch.set_num_threads(4);torch.npu.set_device(1)
    corpus=ActivationCorpus();records=[]
    for ratio in (4,128):
      for batch,queries in ((65,1),(513,1),(65,5)):
       for offset in (1,15):
        length=max(3,queries) if ratio==4 else 127
        case=dict(id=f'metadata_views_c{ratio}_b{batch}_q{queries}_offset{offset}',ratio=ratio,
            queries=[queries]*batch,lengths=[length]*batch,
            capacity=(batch*queries+15)//16*16,mode='graph',draft=False,index=0)
        state=prepare(corpus,case,'npu:1');entry=state['original'];storage=[]
        for attr,field in (('seq_lens','seqLens'),('cu_q','cuQ')):
            original=getattr(state['dev'],attr)
            raw=torch.full((offset+len(original)+16,),-777,dtype=torch.int32,device='npu:1')
            view=raw[offset:offset+len(original)];view.copy_(original)
            setattr(state['dev'],attr,view);setattr(entry.args,field,view.data_ptr())
            storage.append((raw,len(original)))
        initial=bytes(entry.args);graph=torch.npu.NPUGraph()
        with torch.npu.graph(graph):entry.run()
        for repeat in range(3):
            # AIC must invalidate prior summary DCache before the new read.
            entry.partial.fill_(float('nan'));entry.workspace.fill_(0xa5)
            graph.replay();actual,actual_lse=entry.out.cpu(),entry.lse.cpu()
            assert actual.isfinite().all() and actual_lse.isfinite().all()
            guards(state);assert bytes(entry.args)==initial
        n,g,s=layout(entry,case)
        checked=oracle_checks(state,case,n,g,s,actual,actual_lse)
        source=state['source'].to('npu:1')
        meta=baseline.prepare(source,max_query_length=queries,max_seq_length=length)
        expected=baseline.run(source,meta)[0].cpu().float()
        error=metrics(actual[:state['live']].float(),expected)
        assert error['relative_l2']<.04 and error['min_head_cosine']>.999,error
        for raw,size in storage:
            assert (raw[:offset].cpu()==-777).all() and (raw[offset+size:].cpu()==-777).all()
        records.append(dict(case=case,oracle=checked,bf16=error,args_unchanged=True,guards_pass=True,graph_replays=3))
        print(case['id'],'PASS',flush=True)
        del graph,state,entry,storage,raw,view,original,source,meta,expected
        torch.npu.empty_cache()
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(complete=True,records=records),indent=2)+'\n')


if __name__=='__main__':main()
