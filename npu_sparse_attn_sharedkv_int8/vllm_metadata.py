"""Elide only the BF16 SAS tiling plan unused by the standalone INT8 kernel.

Keep builder buffers, metadata tasks, DSpark indices and event frontiers intact.
This hook is selected only inside the INT8 plugin and is separately opt-in.
"""
from dataclasses import replace

import torch

from .runtime import native_launcher

_installed = False
_unused = {}


def unused_sas_metadata(**kwargs):
    if kwargs["head_dim"] != 512 or kwargs["num_heads_kv"] != 1 or kwargs["layout_kv"] != "PA_ND":
        raise ValueError("SAS elision is restricted to the standalone D=512 shared-KV path")
    tensor = next((kwargs.get(name) for name in ("cu_seqlens_q", "seqused_kv", "seqused_q")
                   if kwargs.get(name) is not None), None)
    device = tensor.device if tensor is not None else torch.device(kwargs["device"])
    if device.type != "npu":
        raise ValueError("SAS elision requires an NPU metadata buffer")
    stream = native_launcher().stream_id(device.index if device.index is not None else -1)
    key = device, stream
    result = _unused.get(key)
    if result is None:
        # Draft builders retain their original in-place copies and event IDs.
        # A separate constant per stream avoids an initialization dependency.
        result = torch.zeros(1024, dtype=torch.int32, device=device)
        _unused[key] = result
    return result


def build_unused_sas_metadata(self, metadata_cache, layer_name, query_start_loc, seq_lens,
                              max_seqlen_q, max_seqlen_kv, cu_seqlens_ori_kv, cu_seqlens_cmp_kv,
                              ori_win_left, ori_win_right, dspark_swa):
    result = self.sas_metadata_buffer
    if result.dtype != torch.int32 or result.numel() != 1024:
        raise ValueError("The verified persistent SAS buffer contract changed")
    # This cache entry also controls decode-cu-seqlens preparation in upstream.
    # Keep that control flow and each builder's persistent identity unchanged.
    metadata_cache[layer_name] = result
    return result


def install():
    global _installed
    if _installed:
        return
    from vllm_ascend.attention import dsa_v1
    original_plan = dsa_v1.get_dsa_attn_kv_plan

    def plan(config, compress_ratio=1):
        result = original_plan(config, compress_ratio)
        # General plugins load before the worker's custom-op library. Resolve
        # this packet lazily, after the original selector has resolved it.
        if result.sparse_attn_metadata_op is not torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata:
            raise ValueError("SAS elision requires the verified ordinary PA_ND shared-KV plan")
        return replace(result, sparse_attn_metadata_op=unused_sas_metadata)

    dsa_v1.get_dsa_attn_kv_plan = plan
    dsa_v1.AscendDSAMetadataBuilder._build_sas_metadata = build_unused_sas_metadata
    _installed = True
