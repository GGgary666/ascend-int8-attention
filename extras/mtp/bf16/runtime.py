"""BF16 raw launcher; explicit same-ordered-slot contract, no device readbacks."""
from pathlib import Path
from functools import lru_cache
import ctypes,importlib.util
ROOT=Path(__file__).resolve().parent
SHARED_SLOT_CONTRACT=object()
WORKSPACE_PER_CORE=16*256*4+16*256*2+16*512*4+256*512*2

@lru_cache(maxsize=1)
def bridge():
    name='sharedkv_pointer_host';spec=importlib.util.spec_from_file_location(name,ROOT/'host_build'/f'{name}.so')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

@lru_cache(maxsize=4)
def loaded(group):
    from npu_sparse_attn_sharedkv_int8.attention import _Args
    lib=ctypes.CDLL(str(ROOT/'build'/f'libbf16_mtp_g{group}.so'))
    fn=lib.launch_sharedkv;fn.argtypes=[ctypes.c_uint32,ctypes.c_void_p,ctypes.POINTER(_Args)];fn.restype=None
    return lib,ctypes.cast(fn,ctypes.c_void_p).value

class Prepared:
    def __init__(self,case,*,contract=None,group=2,return_lse=False,output=None):
        import torch
        from npu_sparse_attn_sharedkv_int8.attention import _Args
        if contract is not SHARED_SLOT_CONTRACT:raise ValueError('same ordered physical slots must be established by caller')
        assert group == 2 and case.ratio==1 and case.ori_indices is not None
        assert case.q.dtype==torch.bfloat16 and case.q.shape[1:]==(8,512) and case.q.is_contiguous()
        assert case.ori.dtype==torch.bfloat16 and case.ori.shape[2:]==(1,512) and case.ori.stride(1)==512
        assert case.ori_indices.dtype==torch.int32 and case.ori_indices.shape==(case.q.shape[0],1,256) and case.ori_indices.is_contiguous()
        self.blocks=min(24,len(case.q));self.device=case.q.device.index;self.group=group
        self.workspace=torch.empty((self.blocks,WORKSPACE_PER_CORE),dtype=torch.uint8,device=case.q.device)
        self.args=_Args();self.address=ctypes.addressof(self.args);self.lib,self.pointer=loaded(group)
        assert bridge().attention_args_size==ctypes.sizeof(self.args)
        self.bind(case,return_lse=return_lse,output=output)

    def bind(self,case,*,return_lse=False,output=None):
        import torch
        p=self.args;self.case=case;self.out=output if output is not None else torch.empty_like(case.q)
        self.lse=torch.empty(case.q.shape[:2],dtype=torch.float32,device=case.q.device) if return_lse else None
        for name,t in [('q',case.q),('ori',case.ori),('oriIndices',case.ori_indices),('cuQ',case.cu_q),('seqLens',case.seq_lens),('sinks',case.sink),('out',self.out),('workspace',self.workspace)]:setattr(p,name,t.data_ptr())
        p.lse=self.lse.data_ptr() if self.lse is not None else 0
        p.tokens=len(case.q);p.heads=8;p.batch=len(case.seq_lens);p.oriBlock=case.ori.shape[1];p.oriPages=len(case.ori)
        p.oriStride=case.ori.stride(0);p.oriTopk=256;p.ratio=1;p.softmaxScale=case.softmax_scale
        return self

    def run(self):
        bridge().attention_queued(self.address,self.blocks,0,self.device,self.pointer)
        return (self.out,self.lse)
