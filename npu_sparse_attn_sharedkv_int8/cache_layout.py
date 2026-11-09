# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compact C8 pages while preserving FP32 compressor state values/shapes.

The common-stride ownership scheme follows vllm-ascend's TurboQuant planner:
86add7adfd1194c225fd8880699d5be7a248c3ce,
vllm_ascend/quantization/methods/kv_cache/turboquant/cache.py.
Only its allocation scheme is used; attention remains sharedkv-c8-v1.
Installation is explicit and scoped to this plugin's D=512 C8 specifications.
"""
import math
import logging
from dataclasses import replace

import torch
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.core.kv_cache_utils import may_override_num_blocks
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, KVCacheTensor, UniformTypeKVCacheSpecs

_installed = False


def uses_compact(groups):
    return any(spec.dtype == torch.int8 and getattr(spec, "head_size", None) == 512
               and getattr(spec, "model_version", None) == "deepseek_v4"
               for group in groups if group is not None and getattr(group, "kv_cache_spec", None) is not None
               for spec in (group.kv_cache_spec.kv_cache_specs.values()
                            if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs) else (group.kv_cache_spec,)))


def group_specs(grouped_specs):
    """Preserve the complete C4/indexer and C128 groups; split state groups."""
    largest_page = max(spec.page_size_bytes for group in grouped_specs for spec in group.kv_cache_specs.values())
    # Ascend's input order is C4, C128, then sliding-window/state groups.
    budget = round_up(max(grouped_specs[0].page_size_bytes, grouped_specs[1].page_size_bytes), largest_page)
    groups = []
    for original in grouped_specs:
        chunk, used = {}, 0

        def append():
            merged = UniformTypeKVCacheSpecs.from_specs(chunk)
            if merged is None:
                raise ValueError("Incompatible cache specifications in a compact group")
            groups.append(KVCacheGroupSpec(list(chunk), merged))

        for name, spec in original.kv_cache_specs.items():
            # The old tuple allocator padded C4 FP32 states from 16 to
            # 32 KiB (indexer states from 4096 to 4160 bytes). The common
            # stride does not need those empty regions. The compressor
            # consumes its explicit shape/stride; its values stay FP32.
            if spec.dtype == torch.float32 and getattr(spec, "page_size_padded", None) is not None:
                spec = replace(spec, page_size_padded=None)
            if chunk and used + spec.page_size_bytes > budget:
                append()
                chunk, used = {}, 0
            chunk[name] = spec
            used += spec.page_size_bytes
        if chunk:
            append()
    return groups


def packed_layout(groups):
    """Every physical block ID has the same stride in every cache group."""
    offsets, maximum = {}, 0
    for group in groups:
        specs = group.kv_cache_spec.kv_cache_specs
        used = 0
        for name in sorted(group.layer_names, key=lambda name: -specs[name].page_size_bytes):
            spec = specs[name]
            alignment = get_dtype_size(spec.dtype)
            if getattr(spec, "scale_dim", 0):
                alignment = math.lcm(alignment, get_dtype_size(spec.scale_dtype))
            if spec.dtype == torch.int8 and spec.head_size == 512:
                alignment = math.lcm(alignment, 512)
            offset = round_up(used, alignment)
            if name in offsets:
                raise ValueError(f"Cache layer appears in multiple groups: {name}")
            offsets[name] = offset
            used = offset + spec.page_size_bytes
        maximum = max(maximum, used)
    # Integer row offsets in the fused sparse gather require a whole number
    # of 512-byte rows; FP32 states and indexer scales also remain aligned.
    return round_up(maximum, 512), offsets


def pool_bytes_per_block(groups):
    return packed_layout(groups)[0]


def max_memory_usage(config, groups):
    return pool_bytes_per_block(groups) * sum(group.kv_cache_spec.max_memory_usage_pages(config) for group in groups)


def cache_config(config, groups, available_memory):
    parallel = config.parallel_config
    if any(getattr(parallel, name, 1) != 1 for name in (
        "pipeline_parallel_size", "decode_context_parallel_size", "prefill_context_parallel_size")):
        raise ValueError("Compact C8 pages are integrated for single-stage, non-context-parallel serving")
    if getattr(config, "kv_transfer_config", None) is not None:
        from .memcache_manifest import validate_transfer
        from .vllm_adapter import load_scales
        validate_transfer(config, load_scales())
    stride, offsets = packed_layout(groups)
    blocks = may_override_num_blocks(config, available_memory // stride)
    size = blocks * stride
    descriptors = [KVCacheTensor(size=size, layers=[name], offset=offset,
                                layer_stride=0, block_stride=stride)
                   for name, offset in offsets.items()]
    return blocks, descriptors


def allocate(runner, config):
    """Allocate one backing; descriptors retain offsets into that allocation."""
    specs = runner._get_layer_kv_cache_specs(config)
    stride, offsets = packed_layout(config.kv_cache_groups)
    size = stride * config.num_blocks
    actual = {}
    for descriptor in config.kv_cache_tensors:
        if (len(descriptor.layers) != 1 or descriptor.layer_stride != 0
                or descriptor.block_stride != stride or descriptor.size != size):
            raise ValueError("Invalid compact C8 backing descriptor")
        name = descriptor.layers[0]
        if name in actual or descriptor.offset != offsets[name]:
            raise ValueError("Invalid compact C8 layer offset")
        actual[name] = descriptor.offset
    if set(actual) != set(specs):
        raise ValueError("Compact C8 descriptor/spec coverage differs")
    runner.hybrid_with_attn_and_mamba = False
    backing = runner._allocate_int8_cache_tensor(size, 2 * 1024 * 1024)
    logging.getLogger(__name__).info("Compact INT8 sharedKV: %d blocks, %d bytes/block, %d groups, one %d-byte backing",
        config.num_blocks, stride, len(config.kv_cache_groups), size)
    return {name: backing[offset:] for name, offset in actual.items()}


def reshape(runner, config, raw, kernel_block_sizes=None):
    from vllm_ascend.core.kv_cache_interface import get_storage_block_size
    specs = runner._get_layer_kv_cache_specs(config)
    stride = pool_bytes_per_block(config.kv_cache_groups)
    result = {}
    for group in runner._kv_cache_spec_attn_group_iterator():
        for name in group.layer_names:
            if name in runner.runner_only_attn_layers:
                continue
            spec = specs[name]
            shape = group.backend.get_kv_cache_shape(config.num_blocks, get_storage_block_size(spec),
                                                     spec.num_kv_heads, spec.head_size)
            shapes, dtypes = [shape], [spec.dtype]
            if getattr(spec, "scale_dim", 0):
                shapes.append(group.backend.get_kv_cache_shape(config.num_blocks, get_storage_block_size(spec),
                                                               spec.num_kv_heads, spec.scale_dim))
                dtypes.append(spec.scale_dtype)
            # DSV4 v1 binds a list of views, even for a one-tensor cache.
            result[name] = runner._adjust_kv_layout(raw[name], shapes, dtypes, stride)
    expected = set(specs) - set(runner.runner_only_attn_layers)
    if set(result) != expected:
        raise ValueError("Compact C8 runner views do not cover the cache specifications")
    return result


def install():
    """Wrap only the installed DSV4 planner and runner paths for C8 specs."""
    global _installed
    if _installed:
        return
    from vllm.v1.core import kv_cache_utils as core
    from vllm_ascend.patch.platform import patch_kv_cache_utils as upstream
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    old_groups = upstream._get_kv_cache_groups_uniform_groups
    old_config = upstream._get_kv_cache_config_deepseek_v4_main
    old_pool, old_max = core._pool_bytes_per_block, core._max_memory_usage_bytes_from_groups
    old_allocate = NPUModelRunner._allocate_kv_cache_tensors
    old_reshape = NPUModelRunner._reshape_kv_cache_tensors

    def groups_hook(grouped_specs):
        probe = [KVCacheGroupSpec(list(group.kv_cache_specs), group) for group in grouped_specs]
        return group_specs(grouped_specs) if uses_compact(probe) else old_groups(grouped_specs)

    def config_hook(vllm_config, kv_cache_groups, available_memory):
        return (cache_config(vllm_config, kv_cache_groups, available_memory) if uses_compact(kv_cache_groups)
                else old_config(vllm_config, kv_cache_groups, available_memory))

    def pool_hook(kv_cache_groups):
        return pool_bytes_per_block(kv_cache_groups) if uses_compact(kv_cache_groups) else old_pool(kv_cache_groups)

    def max_hook(vllm_config, kv_cache_groups):
        return (max_memory_usage(vllm_config, kv_cache_groups) if uses_compact(kv_cache_groups)
                else old_max(vllm_config, kv_cache_groups))

    def allocate_hook(self, kv_cache_config):
        return (allocate(self, kv_cache_config) if uses_compact(kv_cache_config.kv_cache_groups)
                else old_allocate(self, kv_cache_config))

    def reshape_hook(self, kv_cache_config, kv_cache_raw_tensors, kernel_block_sizes=None):
        if uses_compact(kv_cache_config.kv_cache_groups):
            return reshape(self, kv_cache_config, kv_cache_raw_tensors, kernel_block_sizes)
        return old_reshape(self, kv_cache_config, kv_cache_raw_tensors, kernel_block_sizes)

    upstream._get_kv_cache_groups_uniform_groups = groups_hook
    upstream._get_kv_cache_config_deepseek_v4_main = config_hook
    upstream._ascend_pool_bytes_per_block = core._pool_bytes_per_block = pool_hook
    upstream._ascend_max_memory_usage_bytes_from_groups = core._max_memory_usage_bytes_from_groups = max_hook
    NPUModelRunner._allocate_kv_cache_tensors = allocate_hook
    NPUModelRunner._reshape_kv_cache_tensors = reshape_hook
    _installed = True
