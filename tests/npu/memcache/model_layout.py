"""Small allocator fixture driven by the deployed model's real cache specs."""

import json
from pathlib import Path
from types import SimpleNamespace as NS
import torch


def model_groups(model_root):
    from vllm_ascend.core.kv_cache_interface import (
        AscendMLAAttentionSpec,
        AscendSlidingWindowMLASpec,
    )
    from vllm_ascend.models.deepseek_v4.compressor import AscendCompressorStateCache
    from vllm_ascend.models.deepseek_v4.indexer import AscendDeepseekV4IndexerCache
    from vllm_ascend.models.layer.attention.layer import DSV4_BLOCK_SIZES
    from vllm_ascend.patch.platform.patch_kv_cache_utils import group_and_unify_kv_cache_specs
    from npu_sparse_attn_sharedkv_int8.cache_layout import group_specs

    model = json.loads((Path(model_root) / "config.json").read_text())
    config = NS(cache_config=NS(block_size=32, cache_dtype="auto"))
    sizes = DSV4_BLOCK_SIZES[32][0]
    specs = {}
    for index, ratio in enumerate(model["compress_ratios"]):
        prefix = f"model.layers.{index}.self_attn"
        specs[f"{prefix}.swa_cache"] = AscendSlidingWindowMLASpec(
            block_size=sizes[1],
            num_kv_heads=1,
            head_size=model["head_dim"],
            dtype=torch.int8,
            sliding_window=model["sliding_window"],
            model_version="deepseek_v4",
            indexes_kv_by_block_stride=True,
        )
        if not ratio:
            continue
        specs[f"{prefix}.attn"] = AscendMLAAttentionSpec(
            block_size=sizes[0] * ratio,
            num_kv_heads=1,
            head_size=model["head_dim"],
            dtype=torch.int8,
            tokens_per_state=ratio,
            model_version="deepseek_v4",
            indexes_kv_by_block_stride=True,
        )
        for role, dim in [("compressor", model["head_dim"])] + (
            [("indexer.compressor", model["index_head_dim"])] if ratio == 4 else []
        ):
            state = NS(
                state_dim=2 * (1 + (ratio == 4)) * dim,
                dtype=torch.float32,
                compress_ratio=ratio,
                block_size=sizes[2] if ratio == 4 else sizes[3],
                sliding_window=ratio * (1 + (ratio == 4)),
            )
            specs[f"{prefix}.{role}.state_cache"] = AscendCompressorStateCache.get_kv_cache_spec(
                state, config
            )
        if ratio == 4:
            indexer = NS(
                head_dim=model["index_head_dim"],
                dtype=torch.int8,
                compress_ratio=ratio,
                cache_config=config.cache_config,
            )
            specs[f"{prefix}.indexer.k_cache"] = AscendDeepseekV4IndexerCache.get_kv_cache_spec(
                indexer, config
            )
    return group_specs(group_and_unify_kv_cache_specs(specs))


def make_views(groups, num_blocks, device):
    from vllm_ascend.core.kv_cache_interface import get_storage_block_size
    from npu_sparse_attn_sharedkv_int8.cache_layout import packed_layout

    stride, offsets = packed_layout(groups)
    alignment = 2 * 1024 * 1024
    allocation = torch.full(
        (num_blocks * stride + alignment,), 53, dtype=torch.uint8, device=device
    )
    shift = (-allocation.data_ptr()) % alignment
    backing = allocation[shift : shift + num_blocks * stride]
    views = {}
    for group in groups:
        for name in group.layer_names:
            spec = group.kv_cache_spec.kv_cache_specs[name]
            offset = offsets[name]
            tensors = []
            for dim, dtype in [(spec.head_size, spec.dtype)] + (
                [(spec.scale_dim, spec.scale_dtype)] if getattr(spec, "scale_dim", 0) else []
            ):
                rows = get_storage_block_size(spec)
                shape = (num_blocks, 1, rows, dim)
                element = torch.empty((), dtype=dtype).element_size()
                raw = backing.view(dtype)
                tensor = raw.as_strided(
                    shape,
                    (stride // element, rows * dim, dim, 1),
                    (backing.storage_offset() + offset) // element,
                )
                tensors.append(tensor)
                offset += rows * dim * element
            views[name] = tensors
    return allocation, backing, views
