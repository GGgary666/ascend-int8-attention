"""Default V65 attention preparation; live tiling is decided on device.

The legacy prepared op supplies validation and rebind semantics. This module
owns the stable ABI boundary, scratch requirements and three launchers.
All positive batch/query capacities use this core when the KV ABI matches.
"""
import copy
import ctypes
from functools import lru_cache
import importlib.util
import os
from types import SimpleNamespace

import torch

from ._paths import native_path

CORES = 24
HEADS = 8
DIM = 512
MAX_SPLITS = 4
def _workspace_bytes(rows, tile):
    return rows * DIM + rows * tile * 5 + rows * DIM * 4 + tile * DIM + 512


# Five retained M16 tasks: separate Q0/Q1, with scores sharing the P/PV phase.
_TASK_BATCH_WORKSPACE_BYTES = 5 * 16 * (2 * 512 + 10 * DIM)
WORKSPACE_BYTES = CORES * max(
    _workspace_bytes(16, 512), _workspace_bytes(64, 256), _TASK_BATCH_WORKSPACE_BYTES
)


def enabled():
    value = os.environ.get('SHAREDKV_C8_DECODE', '1')
    if value not in ('0', '1'):
        raise ValueError('SHAREDKV_C8_DECODE must be 0 or 1')
    return value == '1'


def argument_type():
    from .attention import _Args
    return _Args


@lru_cache(maxsize=1)
def library():
    path = native_path('libsharedkv_c8.so')
    lib = ctypes.CDLL(str(path))
    for suffix in ('sparse', 'dense', 'sparse_small'):
        fn = getattr(lib, 'launch_sharedkv_c8_' + suffix)
        fn.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(argument_type())]
        fn.restype = None
    return lib


def supported(entry):
    if not getattr(entry, 'blocks', 0) or not hasattr(entry, 'args'):
        return False
    p = entry.args
    if not isinstance(p, argument_type()):
        return False
    return (
        p.heads == HEADS
        and p.tokens > 0
        and p.batch > 0
        and p.ratio in (4, 128)
        and p.window == 128
        and p.oriBlock == p.cmpBlock == 32
        and not p.oriIndices
        and (p.ratio != 4 or p.topk == 1024)
    )


def prepare(entry, *, scratch_allocator=None, shared_storage=False):
    """Prepare outside replay, with no CPU query lengths or NPU readback.

    Tests can reserve common guarded buffers and require shared_storage.
    Serving uses its per-stream allocator, retaining buffers of captured graphs.
    """
    if getattr(entry, 'c8_dynamic', False) or not supported(entry):
        return entry
    p = entry.args
    allocate = scratch_allocator or (
        lambda name, shape, dtype, device: torch.empty(shape, dtype=dtype, device=device)
    )
    workspace, partial = entry.workspace, entry.partial
    # Split-KV is selected only for <=12 live queries, even inside a much
    # larger captured graph. Reserve that reachable extent, not graph capacity.
    partial_queries = min(p.tokens, CORES // 2)
    partial_shape = (partial_queries, HEADS, MAX_SPLITS, 520)
    partial_elements = partial_queries * HEADS * MAX_SPLITS * 520
    if shared_storage:
        if (workspace.numel() < WORKSPACE_BYTES
                or partial is None or partial.numel() < partial_elements):
            raise ValueError('C8 shared controls require maximum workspace and four partial slots')
    else:
        if workspace.numel() < WORKSPACE_BYTES:
            workspace = allocate('workspace', (WORKSPACE_BYTES,), torch.uint8, entry.case.q.device)
        if partial is None or partial.numel() < partial_elements:
            partial = allocate('partial', partial_shape, torch.float32, entry.case.q.device)

    out = copy.copy(entry)
    out.args = type(p).from_buffer_copy(p)
    out.args.workspace = workspace.data_ptr()
    out.args.partial = partial.data_ptr()
    out.args.splits = 1
    out.workspace, out.partial = workspace, partial
    out.blocks, out.splits = CORES, 1
    out.argument_address = ctypes.addressof(out.args)
    out._c8_library = library()
    small = p.ratio == 4 and p.tokens < CORES
    suffix = 'sparse_small' if small else ('sparse' if p.ratio == 4 else 'dense')
    out.launch = getattr(out._c8_library, 'launch_sharedkv_c8_' + suffix)
    out.variant = 22 if small else (20 if p.ratio == 4 else 21)
    out.native = native_host(os.environ.get('SHAREDKV_HOST_QUEUE') == '1')
    if out.native.attention_args_size != ctypes.sizeof(type(p)):
        raise RuntimeError('C8 host/device argument ABI mismatch')
    out.c8_dynamic = True
    return out


@lru_cache(maxsize=1)
def host_module():
    path = native_path('sharedkv_c8_host.so')
    spec = importlib.util.spec_from_file_location('sharedkv_c8_host', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=2)
def native_host(queued=False):
    module = host_module()
    return SimpleNamespace(
        attention=module.attention_queued if queued else module.attention,
        attention_args_size=module.attention_args_size,
        _owner=module,
    )
