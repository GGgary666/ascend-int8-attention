"""C4 selected-multiset robustness: integer + FP64 formula are the golden.

Vendor BF16 is additionally gated for full valid duplicate selections. Synthetic
holes/empty/over-capacity selections are diagnostic-only against vendor SCFA:
it limits inspected slots and can terminate gathering at an interior -1.
Normal serving geometry is strictly compared with BF16 in test_capacity.
"""
import argparse
import copy
import ctypes
from dataclasses import replace
import json
from pathlib import Path

import torch
from npu_sparse_attn_sharedkv_int8.reference import sharedkv_reference, tiled_quantized_reference
from tests.support import baseline
from tests.support.attention_cases import prepare, layout, guards, metrics
from tests.support.serving_shapes import ActivationCorpus


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--control-device',type=Path)
    p.add_argument('--wide',action='store_true',help='Additional retained-KV key-domain boundaries')
    p.add_argument('--retained',action='store_true',help='Resource-boundary sparse masks above512 keys')
    args=p.parse_args();out=args.output;assert not out.exists()
    baseline.load_ops();torch.set_num_threads(4);torch.npu.set_device(1)
    corpus=ActivationCorpus();records=[]
    shapes = ([(49,1028,64),(25,1284,32),(49,1924,64),(73,2048,80)] if args.wide else
              [(7,129,16),(13,513,16),(25,513,32),(49,129,64),(65,513,80),
               (73,517,80),(97,1023,128)])
    if args.retained:
        shapes=[(25,4*513,32),(49,4*544,64),(97,4*737,112),(73,4*768,80),(25,4*769,32),
                (65,4*800,80),(97,4*895,112),(73,4*896,80),(25,4*897,32),
                (33,4*928,48),(65,4*959,80),(101,4*960,112),(73,4*961,80)]
    for batch,length,capacity in shapes:
        case=dict(id='sparse_edges',ratio=4,queries=[1]*batch,lengths=[length]*batch,
                  capacity=capacity,mode='graph',draft=False,index=0)
        state=prepare(corpus,case,'npu:1');entry=state['original']
        control=None
        if args.control_device:
            control=copy.copy(entry);control.args=type(entry.args).from_buffer_copy(entry.args)
            control.argument_address=ctypes.addressof(control.args);control.native=None
            control._control_library=ctypes.CDLL(str(args.control_device.resolve()))
            suffix={20:'sparse',22:'sparse_small'}[entry.variant]
            control.launch=getattr(control._control_library,'launch_sharedkv_c8_'+suffix)
            control.launch.argtypes=[ctypes.c_uint32,ctypes.c_void_p,ctypes.POINTER(type(entry.args))]
            control.launch.restype=None
        n,g,s=layout(entry,case)
        modes=('duplicates','holes','empty','extended_duplicates','single','one_key_duplicates')
        if args.retained: modes += ('permutation',)
        for mode in modes:
            indices=torch.full_like(state['quant'].cmp_indices,-1)
            extent=min(1024,length//4)
            if mode=='duplicates':
                indices[:,:,:extent]=torch.arange(extent).div(2,rounding_mode='floor').reshape(1,1,-1)
            elif mode=='permutation':
                indices[:,:,:extent]=torch.randperm(extent,generator=torch.Generator().manual_seed(length)).reshape(1,1,-1)
            elif mode=='extended_duplicates':
                # Additional robustness contract: more selected entries than
                # the causal key extent. SCFA caps inspected slots at extent
                # (GetSparseActualSeqLen); use the independent FP64 formula
                # for this extension and record vendor differences explicitly.
                indices[:,:,::2]=torch.arange(512).remainder(length//4).reshape(1,1,-1)
            elif mode=='holes':
                indices[:,:,:extent:3]=torch.arange((extent+2)//3).reshape(1,1,-1)
                indices[:,:,1:extent:7]=length//4+3
            elif mode in ('single','one_key_duplicates'):
                # A singleton may occur in any key domain and after holes.
                # Two copies of that key must retain log(2) multiplicity.
                selected=torch.arange(batch).remainder(length//4).reshape(batch,1)
                indices[:,:,1023]=selected
                if mode=='one_key_duplicates':indices[:,:,0]=selected
            state['quant'].cmp_indices.copy_(indices)
            state['dev'].cmp_indices[:batch].copy_(indices)
            entry.workspace.fill_(0xA5);entry.partial.fill_(float('nan'))
            entry.run();actual=entry.out[:batch].cpu();lse=entry.lse[:batch].cpu();guards(state)
            oracle,olse=tiled_quantized_reference(state['quant'],*state['quantizers'],
                tile_size=n,query_group=g,splits=s,return_lse=True,residual_quantization=True,
                joint_sparse=(batch>=24 and length//4<=960 and g==1),
                joint_tile_limit=960 if batch>=24 and length//4<=960 else 256)
            torch.testing.assert_close(actual.float(),oracle.float(),rtol=.008,atol=.0003)
            torch.testing.assert_close(lse,olse,rtol=2e-5,atol=2e-5)
            bf=replace(state['dev'],ori=state['source'].ori.to('npu:1'),cmp=state['source'].cmp.to('npu:1'))
            meta=baseline.prepare(bf,max_query_length=1,max_seq_length=length)
            error=metrics(actual.float(),baseline.run(bf,meta)[0][:batch].cpu().float())
            float_out,_=sharedkv_reference(replace(state['source'],cmp_indices=indices))
            float_error=metrics(actual.float(),float_out)
            assert float_error['relative_l2']<.04 and float_error['min_head_cosine']>.999,float_error
            if mode in ('duplicates','permutation'):
                assert error['relative_l2']<.04 and error['min_head_cosine']>.999,error
            control_error=None
            if control:
                control.run();control_error=metrics(control.out[:batch].cpu().float(),float_out)
                assert control_error['relative_l2']<.04 and control_error['min_head_cosine']>.999,control_error
                guards(state)
            row=dict(batch=batch,length=length,capacity=capacity,mode=mode,layout=[n,g,s],
                     guards_pass=True,oracle_pass=True,float_reference=float_error,
                     bf16_diagnostic=error,control_float_reference=control_error,
                     vendor_bf16_gated=mode in ('duplicates','permutation'))
            records.append(row);print(json.dumps(row),flush=True)
        del state,entry,bf,meta;torch.npu.empty_cache()
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(complete=True,records=records),indent=2)+'\n')


if __name__=='__main__':main()
