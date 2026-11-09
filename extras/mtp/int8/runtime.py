"""MTP slot-sharing launch adapter; acceptance is tracked in the paired manifest.

The caller must establish equal, ordered physical slots within each request.
This module never infers that fact from Q=5 or a context-length threshold.
"""
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import copy,ctypes,sys
ROOT=Path(__file__).resolve().parent
SHARED_SLOT_CONTRACT=object()
WORKSPACE_PER_CORE=254464

@lru_cache(maxsize=8)
def loaded(path):
    return ctypes.CDLL(str(Path(path).resolve()))

@lru_cache(maxsize=1)
def runtime():
    import importlib.util
    name='sharedkv_pointer_host'
    spec=importlib.util.spec_from_file_location(name,ROOT/'host_build'/f'{name}.so')
    bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(bridge)
    return SimpleNamespace(loaded=loaded),bridge

def bind(original,*,contract=None):
    if contract is not SHARED_SLOT_CONTRACT:return original
    p=original.args
    if p.ratio!=1 or p.heads!=8 or not p.oriIndices or p.oriTopk!=256 or original.variant!=0:return original
    import torch
    r,bridge=runtime();result=copy.copy(original)
    result.args=type(p).from_buffer_copy(p);result.argument_address=ctypes.addressof(result.args)
    path=ROOT/'lib/libsharedkv_mtp.so';result._decode_library=r.loaded(str(path))
    launch=result._decode_library.launch_sharedkv
    launch.argtypes=[ctypes.c_uint32,ctypes.c_void_p,ctypes.POINTER(type(p))];launch.restype=None
    assert bridge.attention_args_size==ctypes.sizeof(p)
    pointer=ctypes.cast(launch,ctypes.c_void_p).value
    result.native=SimpleNamespace(attention=lambda address,blocks,variant,device:bridge.attention_queued(address,blocks,variant,device,pointer))
    result.workspace=torch.empty((result.blocks,WORKSPACE_PER_CORE),dtype=torch.uint8,device=original.workspace.device)
    result.args.workspace=result.workspace.data_ptr();result.optimized_library_path=str(path);result.optimization_route='mtp_shared_slots_g2_n256_online128_packed'
    return result

def rebind(candidate,original):
    """Copy current launch arguments while preserving the larger owned arena."""
    assert candidate is not original
    ctypes.memmove(ctypes.addressof(candidate.args),ctypes.addressof(original.args),ctypes.sizeof(original.args))
    candidate.args.workspace=candidate.workspace.data_ptr()
    candidate.case=original.case;candidate.out=original.out;candidate.lse=original.lse
    return candidate

def clear_inputs(candidate):
    candidate.case=candidate.out=candidate.lse=None
