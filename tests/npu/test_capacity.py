"""Unrestricted C4/C128 capacity: arithmetic, actual page views and paired timing."""
import argparse
import copy
import ctypes
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

import torch
from npu_sparse_attn_sharedkv_int8.attention import PreparedAttention
from npu_sparse_attn_sharedkv_int8.reference import tiled_quantized_reference
from tests.support import baseline
from tests.support.attention_cases import prepare, layout, guards, metrics
from tests.support.serving_shapes import ActivationCorpus


def cases():
    rows=[]
    for ratio in (4,128):
        shapes=[
            ('b9_q5',[5]*9,[1028]*9,48),
            ('b9_q6',[6]*9,[8192]*9,64),
            ('b16_q1',[1]*16,[65536]*16,32),
            ('b16_q5',[5]*16,[8192]*16,96),
            ('b16_q6',[6]*16,[135000]*16,96),
            ('b23_q6',[6]*23,[8192]*23,144),
            ('b32_q1',[1]*32,[32768]*32,48),
            ('b32_q6',[6]*32,[65536]*32,192),
            ('b64_q1',[1]*64,[8192]*64,64),
            ('b64_q6',[6]*64,[8192]*64,384),
            ('b128_q1',[1]*128,[4096]*128,128),
            ('b128_q6',[6]*128,[4096]*128,768),
            ('b256_q1',[1]*256,[513]*256,256),
            ('mixed23',[1+i%9 for i in range(23)],
                [[129,1028,8193,32769,65537,133333][i%6] for i in range(23)],128),
            ('capacity49',[49],[8192],49),
            ('prefill256',[256],[256],256),
            ('prefill1024',[1024],[1024],1024),
            ('prefill4096',[4096],[4096],4096),
            ('chunk4096_long',[4096],[65536],4096),
            ('prefill8192',[8192],[8192],8192),
            ('b9_q1_small_image',[1]*9,[129]*9,9),
            ('b23_q1_small_image',[1]*23,[8193]*23,23),
            ('b48_q1_two_waves',[1]*48,[1028]*48,48),
            ('b49_q1_three_waves',[1]*49,[1028]*49,49),
            ('b512_q1',[1]*512,[129]*512,512),
            ('prefill16384',[16384],[16384],16384),
        ]
        if ratio==128:
            shapes += [('b9_q7_group_tail',[7]*9,[8193]*9,64),
                       ('b17_q3_group_tail',[3]*17,[8192]*17,64)]
        for i,(name,q,s,capacity) in enumerate(shapes):
            rows.append(dict(id=f'c{ratio}_{name}',ratio=ratio,queries=q,lengths=s,
                             capacity=capacity,mode='eager',draft=False,index=i%3))
    return rows


def heldout_cases():
    """Extension task counts, tile boundaries, draft tails and mixed lengths.

    Initially held out, then used during refinement; not a final blind set.
    """
    rows=[]
    for ratio in (4,128):
        shapes=[
            ('b7_q1_s129',[1]*7,[129]*7,16),
            ('b11_q1_s511',[1]*11,[511]*11,16),
            ('b13_q1_s512',[1]*13,[512]*13,16),
            ('b25_q1_s513',[1]*25,[513]*25,32),
            ('b31_q1_s16383',[1]*31,[16383]*31,32),
            ('b47_q1_s16385',[1]*47,[16385]*47,64),
            ('b73_q1_s32767',[1]*73,[32767]*73,80),
            ('b17_q5_s1028',[5]*17,[1028]*17,96),
            ('b25_q6_s8193',[6]*25,[8193]*25,160),
            ('b37_q5_s32769',[5]*37,[32769]*37,192),
            ('b67_q3_s513',[3]*67,[513]*67,224),
            ('mixed31',[1+(i*7)%9 for i in range(31)],
                [[129,1028,16384,32769,65536,133333][i%6] for i in range(31)],192),
        ]
        for i,(name,q,s,cap) in enumerate(shapes):
            rows.append(dict(id=f'heldout_c{ratio}_{name}',ratio=ratio,queries=q,
                lengths=s,capacity=cap,mode='graph',draft=False,index=i%3))
    return rows


def independent_query_oracle_checks(state, case, tile, retained, actual, actual_lse):
    """Independent per-query integer model, including each core's ends."""
    quant = state['quant']; oq, cq = state['quantizers']; total = state['live']
    owners, positions, lengths = [], [], []
    for b, (count, length) in enumerate(zip(case['queries'], case['lengths'])):
        owners.extend([b] * count)
        positions.extend(range(length - count + 1, length + 1))
        lengths.extend([length] * count)
    if total <= 192:
        selected = range(total)
    else:
        # Cover both ends of every core's ownership as well as the old
        # first/middle/last sample. Full BF16 output comparison is unchanged.
        selected = sorted({0, total // 2, total - 1,
            *range(min(24, total)), *range(max(0, total - 24), total)})
    for token in selected:
        b = owners[token]
        sub = replace(quant, q=quant.q[token:token+1],
            ori_block_table=quant.ori_block_table[b:b+1], cmp_block_table=quant.cmp_block_table[b:b+1],
            cmp_indices=quant.cmp_indices[token:token+1] if quant.cmp_indices is not None else None,
            cu_q=torch.tensor([0, 1], dtype=torch.int32),
            seq_lens=torch.tensor([positions[token]], dtype=torch.int32))
        out, lse = tiled_quantized_reference(sub, oq, cq, tile_size=tile, query_group=1,
            splits=1, return_lse=True, residual_quantization=True,
            joint_dense=retained, joint_sparse=retained and case['ratio']==4, joint_tile_limit=tile)
        torch.testing.assert_close(actual[token:token+1].float(), out.float(), rtol=.008, atol=.0003)
        torch.testing.assert_close(actual_lse[token:token+1], lse, rtol=2e-5, atol=2e-5)
    return len(selected)


def oracle_checks(state, case, n, group, splits, actual, actual_lse):
    quant=state['quant'];oq,cq=state['quantizers'];total=state['live']
    task_batch = (group==1 and (total>24 or (case['ratio']==4 and total==24))
                  and 0<max(case['lengths'])//case['ratio']<=960
                  and (n<=256 or all(v<=1 for v in case['queries'])))
    if total > 48 and group == 1 and all(v == 1 for v in case['queries']):
        return independent_query_oracle_checks(state, case, n, task_batch, actual, actual_lse)
    joint_sparse = case['ratio']==4 and task_batch
    joint_tile_limit = 960 if task_batch else 256
    if total<=192:
        out,lse=tiled_quantized_reference(quant,oq,cq,tile_size=n,query_group=group,
            splits=splits,return_lse=True,residual_quantization=True,joint_dense=True,joint_sparse=joint_sparse,
            joint_tile_limit=joint_tile_limit)
        torch.testing.assert_close(actual[:total].float(),out.float(),rtol=.008,atol=.0003)
        torch.testing.assert_close(actual_lse[:total],lse,rtol=2e-5,atol=2e-5)
        return total
    # Full BF16 comparison is still performed. Sample complete quantization
    # groups for the more expensive independent integer oracle, preserving each
    # group's absolute causal position and original physical cache page table.
    starts=[0]
    for count in case['queries']:starts.append(starts[-1]+count)
    checked=0
    for b in sorted({0,len(case['queries'])//2,len(case['queries'])-1}):
        count=case['queries'][b]
        for g in sorted({0,(count//2)//group,(count-1)//group}):
            begin=g*group;rows=min(group,count-begin);token=starts[b]+begin
            sub=replace(quant,q=quant.q[token:token+rows],
                ori_block_table=quant.ori_block_table[b:b+1],cmp_block_table=quant.cmp_block_table[b:b+1],
                cmp_indices=quant.cmp_indices[token:token+rows] if quant.cmp_indices is not None else None,
                cu_q=torch.tensor([0,rows],dtype=torch.int32),
                seq_lens=torch.tensor([case['lengths'][b]-count+begin+rows],dtype=torch.int32))
            out,lse=tiled_quantized_reference(sub,oq,cq,tile_size=n,query_group=group,
                splits=splits,return_lse=True,residual_quantization=True,joint_dense=True,joint_sparse=joint_sparse,
                joint_tile_limit=joint_tile_limit)
            torch.testing.assert_close(actual[token:token+rows].float(),out.float(),rtol=.008,atol=.0003)
            torch.testing.assert_close(actual_lse[token:token+rows],lse,rtol=2e-5,atol=2e-5)
            checked+=rows
    return checked


def profile_once(op):
    """Single eager launch; external diagnostics may supply a marked range."""
    op()
    return dict(mode='eager',target_launches=1)


def paired_timing(ops, depth):
    graphs={}
    for name,op in ops.items():
        graph=torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            for _ in range(depth):op()
        graphs[name]=graph
    for _ in range(3):
        for graph in graphs.values():graph.replay()
    torch.npu.synchronize()
    samples={name:[] for name in ops}
    names=list(ops)
    for round_index in range(9):
        order=names[round_index%len(names):]+names[:round_index%len(names)]
        if round_index%2:order=order[::-1]
        for name in order:
            start,end=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
            start.record();graphs[name].replay();end.record();end.synchronize()
            samples[name].append(start.elapsed_time(end)*1000/depth)
    medians={name:statistics.median(values) for name,values in samples.items()}
    result=dict(depth=depth,samples_us=samples,median_us=medians,
                candidate_vs_bf16=medians['candidate']/medians['bf16'],
                candidate_vs_original=medians['candidate']/medians['original'])
    if 'control' in medians:result['candidate_vs_control']=medians['candidate']/medians['control']
    return result


@torch.inference_mode()
def main(case_provider=None, *, op_adapter=None):
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--heldout',action='store_true',help='Run the additional 24-case extension matrix')
    p.add_argument('--case',action='append');p.add_argument('--profile-only',choices=('candidate','original','bf16','control'))
    p.add_argument('--control-device',type=Path,help='Frozen expanded-capacity C8 library; test only')
    p.add_argument('--time-lse', action='store_true',
                   help='Enable LSE for EVERY timed/profiled role; default matches serving (off)')
    a=p.parse_args();assert not a.output.exists()
    source = case_provider() if case_provider else (heldout_cases() if a.heldout else cases())
    selected=[c for c in source if not a.case or c['id'] in a.case]
    assert selected
    if a.profile_only:assert len(selected)==1
    if a.profile_only=='control':assert a.control_device
    # Match the serving bridge by default even for direct module invocations.
    # Explicit queue=0 remains available for a labeled bridge-only diagnostic.
    os.environ.setdefault('SHAREDKV_HOST_QUEUE','1')
    # Register custom OPP before the first NPU allocation/kernel initializes
    # CANN's tiling registry (especially in a fresh SSH environment).
    baseline.load_ops()
    torch.set_num_threads(4);torch.npu.set_device(1);corpus=ActivationCorpus();records=[]
    device_path=None
    a.output.parent.mkdir(parents=True,exist_ok=True)
    for case in selected:
        state=prepare(corpus,case,'npu:1');entry=state['original'];oq,cq=state['quantizers']
        assert entry.c8_dynamic and entry.variant in (20,21,22),'Capacity fell back from the maintained kernel'
        loaded_path=Path(entry._c8_library._name).resolve()
        assert device_path is None or device_path==loaded_path, 'Candidate library changed during this run'
        device_path=loaded_path
        assert entry.partial.numel()<=12*8*4*520
        legacy=PreparedAttention(state['dev'],oq.scale.to('npu:1'),cq.scale.to('npu:1'),
            output=entry.out,return_lse=True,lse_output=entry.lse,implementation='fallback')
        # Integer variants share cache/input/output addresses and a common
        # guarded workspace. BF16 uses the same real source before KV encoding.
        extent=max(entry.workspace.numel(),legacy.workspace.numel())
        wg=torch.full((extent+1024,),0xA5,dtype=torch.uint8,device='npu:1')
        state['guards']=(*state['guards'][:2],wg)
        for op in (entry,legacy):
            op.workspace=wg[512:-512];op.args.workspace=op.workspace.data_ptr()
        bf=replace(state['dev'],ori=state['source'].ori.to('npu:1'),cmp=state['source'].cmp.to('npu:1'))
        meta=baseline.prepare(bf,max_query_length=max(case['queries']),max_seq_length=max(case['lengths']))
        ops=dict(candidate=entry.run,original=legacy.run,
                 bf16=lambda:baseline.run(bf,meta,return_lse=a.time_lse))
        integer_ops=[entry,legacy]
        if a.control_device:
            control=copy.copy(entry);control.args=type(entry.args).from_buffer_copy(entry.args)
            control.argument_address=ctypes.addressof(control.args);control.native=None
            control._control_library=ctypes.CDLL(str(a.control_device.resolve()))
            suffix={20:'sparse',21:'dense',22:'sparse_small'}[entry.variant]
            control.launch=getattr(control._control_library,'launch_sharedkv_c8_'+suffix)
            control.launch.argtypes=[ctypes.c_uint32,ctypes.c_void_p,ctypes.POINTER(type(entry.args))]
            control.launch.restype=None
            ops['control']=control.run
            integer_ops.append(control)
        # External diagnostics may equalize launch bridges without replacing
        # kernel arguments, correctness checks or the default test protocol.
        if op_adapter is not None:
            ops=op_adapter(ops)
        contract=dict(version=2,return_lse=a.time_lse,oracle_return_lse=True,
                      roles_return_lse={name:a.time_lse for name in ops},
                      host_queue=os.environ.get('SHAREDKV_HOST_QUEUE')=='1')
        contract['roles_launch']={name:(
            op.__self__.native.attention.__name__ if op.__self__.native is not None else 'ctypes'
            ) if hasattr(op,'__self__') else 'vendor_torch_op' for name,op in ops.items()}
        def configure_timing_outputs():
            # Correctness below always checks LSE. Timed/profiled kernels
            # must have the SAME optional outputs, including the BF16 role.
            torch.npu.synchronize()
            for op in integer_ops:
                op.args.lse=entry.lse.data_ptr() if a.time_lse else 0
            entry.lse.fill_(123)
            torch.npu.synchronize()
        if a.profile_only:
            configure_timing_outputs()
            torch.npu.synchronize();execution=profile_once(ops[a.profile_only]);torch.npu.synchronize()
            a.output.write_text(json.dumps(dict(complete=True,diagnostic_only=True,case=case,
                entry=a.profile_only,timing_contract=contract,profile_execution=execution))+'\n')
            return
        entry.workspace.fill_(0xA5);entry.partial.fill_(float('nan'));entry.run()
        actual,actual_lse=entry.out.cpu(),entry.lse.cpu();guards(state)
        assert actual.isfinite().all() and actual_lse.isfinite().all()
        n,group,splits=layout(entry,case)
        checked=oracle_checks(state,case,n,group,splits,actual,actual_lse)
        golden=baseline.run(bf,meta)[0][:state['live']].cpu().float()
        error=metrics(actual[:state['live']].float(),golden)
        assert error['relative_l2']<.04 and error['min_head_cosine']>.999,error
        legacy.run();old_error=metrics(legacy.out[:state['live']].cpu().float(),golden)
        depth=max(1,min(128,math.ceil(4096/case['capacity'])))
        configure_timing_outputs()
        timing=paired_timing(ops,depth)
        if not a.time_lse:
            assert (entry.lse.cpu()==123).all(), 'A timed INT8 role wrote disabled LSE'
        timing['contract']=contract
        row=dict(case=case,route='V65',variant=entry.variant,layout=[n,group,splits],
            integer_oracle_queries=checked,bf16=error,original_bf16=old_error,
            activation=state['provenance'],workspace_bytes=extent,
            partial_bytes=entry.partial.numel()*4,guards_pass=True,timing=timing)
        if a.control_device:row['control_device']=str(a.control_device.resolve())
        records.append(row)
        with a.output.with_suffix('.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(dict(id=case['id'],layout=row['layout'],bf16=error,
            median_us=timing['median_us'],vs_bf16=timing['candidate_vs_bf16'],vs_original=timing['candidate_vs_original'])),flush=True)
        del state,entry,legacy,bf,meta,ops,integer_ops,configure_timing_outputs,wg,actual,actual_lse,golden
        torch.npu.empty_cache()
    a.output.write_text(json.dumps(dict(complete=True,records=records,
        device_library=str(device_path),
        device_sha256=hashlib.sha256(device_path.read_bytes()).hexdigest()),indent=2)+'\n')


if __name__=='__main__':main()
