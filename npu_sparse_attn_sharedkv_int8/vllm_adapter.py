"""D=512 DSA integration using frozen C8 scales and the standalone kernels.

Only original/compressed attention cache specs change dtype. By default the
existing page allocation remains intact. SHAREDKV_COMPACT_PAGES=1 additionally
uses the explicit common-stride planner, retaining FP32 compressor states.
"""
from collections import OrderedDict
from dataclasses import replace
from functools import lru_cache
import logging
import math
import os

import torch
import torch_npu  # noqa: F401

from .attention import PreparedAttention
from .c8_attention import enabled as c8_enabled
from .cache import PreparedCacheWrite
from .memcache_manifest import validate_transfer_mode
from .reference import SharedKVCase
from .runtime import native_launcher

_scratch = {}
_reported_c8_routes = set()


def stream_key(device):
    # A dictionary key needs only stream identity. Python current_stream()
    # additionally constructs a Stream and repeats generic device detection;
    # its npu_stream property also drains torch_npu's host task queue.
    return native_launcher().stream_id(device.index if device.index is not None else -1)


def geometry(tensor):
    if tensor is None:
        return None
    return tensor.shape, tensor.stride(), tensor.dtype, tensor.device


def scratch_allocator(name, shape, dtype, device):
    # Attention layers execute sequentially on one stream. Separate streams
    # receive separate storage. Retain old capacities for captured graphs.
    key = device, stream_key(device), name, dtype
    buffers = _scratch.setdefault(key, [])
    count = math.prod(shape)
    if not buffers or buffers[-1].numel() < count:
        buffers.append(torch.empty(1 << (count - 1).bit_length(), dtype=dtype, device=device))
    return buffers[-1][:count].view(shape)


@lru_cache(maxsize=1)
def load_scales():
    path = os.environ.get("SHAREDKV_SCALES")
    if not path:
        raise ValueError("INT8 mode requires SHAREDKV_SCALES containing frozen calibrated scales")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("protocol") != "sharedkv-c8-v1":
        raise ValueError("Wrong shared-KV scale protocol")
    for key, value in payload["scales"].items():
        if value.shape != (512,) or value.dtype != torch.float32 or not bool(torch.isfinite(value).all() & (value > 0).all()):
            raise ValueError(f"Invalid scale vector: {key}")
    return payload["scales"]


@lru_cache(maxsize=16)
def check_device(device):
    if torch.npu.get_device_name(device) != "Ascend910_9382":
        raise ValueError("This adapter is verified only on Ascend910_9382 / dav-2201")
    if native_launcher() is None:
        raise RuntimeError("Build the native host launcher with ./build.sh before serving")


class LayerRuntime:
    def __init__(self, impl):
        self.impl = impl
        self.prefix = impl.swa_cache_layer.prefix.removesuffix(".swa_cache")
        if impl.head_dim != 512 or impl.turboquant is not None:
            raise ValueError("INT8 shared-KV requires ordinary D=512 DSA")
        from vllm_ascend.ascend_config import get_ascend_config
        if getattr(get_ascend_config(), "enable_dsv4_kv_int8", False):
            raise ValueError("Disable the pre-existing enable_dsv4_kv_int8 prototype when using this plugin")
        parallel = impl.vllm_config.parallel_config
        if (getattr(parallel, "decode_context_parallel_size", 1) != 1
                or getattr(parallel, "prefill_context_parallel_size", 1) != 1):
            raise ValueError("Context-parallel cache writers have not been integrated")
        self.scales = {}
        self.writers = {}
        self.attention_ops = OrderedDict()
        self.captured_ops = []
        self.captured_ids = set()
        self.empty_lse = None
        self.plan = None
        self.reported_caches = set()
        self.validator = None
        self.adaptive_graph_splits = os.environ.get("SHAREDKV_ADAPTIVE_KV_SPLITS") == "1"
        self.clean_c8_decode = c8_enabled()
        if os.environ.get("SHAREDKV_VALIDATE_DIR"):
            from .model_validation import ModelValidator
            self.validator = ModelValidator(self.prefix)

    def scale(self, kind, device):
        if kind not in self.scales:
            check_device(device)
            key = f"{self.prefix}.{kind}"
            cpu = load_scales().get(key)
            if cpu is None:
                raise ValueError(f"Missing real-model calibration for {key}")
            self.scales[kind] = (cpu.to(device), cpu.double().reciprocal().float().to(device))
        return self.scales[kind]

    def write(self, cache, updates, slots, kind):
        if updates is None or updates.shape[0] == 0:
            return
        if slots.shape[0] != updates.shape[0]:
            raise ValueError("Cache-write slot/update lengths differ")
        # Layout and cache allocation are stable; number of update rows varies.
        key = (kind, geometry(cache), updates.shape[1:], updates.stride(), updates.dtype,
               slots.shape[1:], slots.stride(), slots.dtype, updates.device, slots.device,
               stream_key(cache.device))
        writer = self.writers.get(key)
        if writer is None:
            writer = PreparedCacheWrite(cache, updates, slots, self.scale(kind, cache.device)[1])
            self.writers[key] = writer
        else:
            writer._rebind_same_geometry(cache, updates, slots)
        writer.run()
        if kind not in self.reported_caches:
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                logging.getLogger("vllm.sharedkv_c8").info(
                    "Standalone INT8 sharedKV cache %s.%s: shape=%s dtype=%s page_stride=%d bytes",
                    self.prefix, kind, tuple(cache.shape), cache.dtype, cache.stride(0))
            self.reported_caches.add(kind)
        if self.validator is not None:
            self.validator.write(cache, updates, slots, kind, self.scale(kind, cache.device)[0])
        # Ordinary same-stream tensor lifetimes suffice after launch. Keeping
        # every layer's last activation would add significant resident HBM.
        writer.updates = writer.slots = None

    def attention(self, q, **kw):
        if kw.get("layout_q") != "TND" or kw.get("layout_kv") != "PA_ND":
            raise ValueError("Expected TND / PA_ND shared-KV layout")
        original_indices = kw.get("ori_sparse_indices")
        if kw.get("ori_mask_mode", 4) != 4 or (kw.get("ori_win_right", 0) != 0 and original_indices is None):
            raise ValueError("Unsupported original window mask")
        ratio = max(1, kw.get("cmp_ratio", 1))
        if ratio > 1 and kw.get("cmp_mask_mode", 3) != 3:
            raise ValueError("Compressed cache must use causal masking")
        case = SharedKVCase(q=q, ori=kw["ori_kv"], ori_block_table=kw["ori_block_table"],
                            cu_q=kw["cu_seqlens_q"], seq_lens=kw["seqused_kv"], sink=kw["sinks"],
                            cmp=kw.get("cmp_kv"), cmp_block_table=kw.get("cmp_block_table"),
                            cmp_indices=kw.get("cmp_sparse_indices"), ratio=ratio,
                            window=kw["ori_win_left"] + 1, softmax_scale=kw["softmax_scale"],
                            ori_indices=original_indices)
        capturing = torch.npu.is_current_stream_capturing()
        # Runtime prefill bounds may specialize eager launches. Graph replay
        # can grow sequence lengths, so no short-context specialization there.
        bound = None if capturing else getattr(kw.get("metadata"), "_sharedkv_max_seq_len", None)
        if bound is not None and 0 < bound <= 1024:
            bound = 1024
        elif bound is not None and ratio == 128 and bound > 32768:
            # Eager mixed/prefill batches can use the wider C128 tile. Keep
            # a valid upper bound, bucketed to avoid a new cached plan at
            # every decode token. Capture still derives choices on device.
            bound = 1 << (int(bound) - 1).bit_length()
        else:
            bound = None
        adaptive = (self.adaptive_graph_splits and capturing and ratio == 4 and original_indices is None
                    and q.shape[1] <= 8 and 24 <= q.shape[0] <= 48
                    and case.cmp_indices.shape[-1] <= 1024)
        return_lse = kw.get("return_softmax_lse", False)
        output = torch.empty_like(q)
        lse = torch.empty(q.shape[:2], dtype=torch.float32, device=q.device) if return_lse else None
        key = (tuple(geometry(t) if isinstance(t, torch.Tensor) else t for t in vars(case).values()),
               bound, return_lse, adaptive, stream_key(q.device))
        op = self.attention_ops.get(key)
        if op is None:
            op = PreparedAttention(case, self.scale("ori", q.device)[0],
                                   self.scale("cmp", q.device)[0] if ratio > 1 else None,
                                   max_seq_len=bound, return_lse=return_lse,
                                   output=output, lse_output=lse,
                                   adaptive_splits=adaptive,
                                   scratch_allocator=scratch_allocator)
            if self.clean_c8_decode:
                route = (getattr(op, 'c8_dynamic', False), ratio, q.shape[0], getattr(op, 'variant', -1))
                if route not in _reported_c8_routes:
                    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                    if rank % 16 == 0:
                        logging.getLogger("vllm.sharedkv_c8").info(
                            "C8 decode route rank=%d clean=%s ratio=%d capacity=%d variant=%d Q/P_residuals=on",
                            rank, *route)
                    _reported_c8_routes.add(route)
            self.attention_ops[key] = op
            if len(self.attention_ops) > 8:
                self.attention_ops.popitem(last=False)
        else:
            self.attention_ops.move_to_end(key)
            op._rebind_same_geometry(case, output=output, lse_output=lse)
        if capturing and id(op) not in self.captured_ids:
            self.captured_ids.add(id(op))
            self.captured_ops.append(op)
        result = op.run()
        if self.validator is not None:
            self.validator.attention(case, result)
        # Captured output allocations belong to torch_npu's graph pool, just
        # like ordinary operator outputs. Cache only launch geometry/scratch.
        op.case = op.out = op.lse = None
        # Match csrc/torch_binding.cpp::construct_output_tensor, including
        # the trailing singleton and the empty FP32 tensor when disabled.
        if lse is not None:
            return result, lse.unsqueeze(-1)
        if self.empty_lse is None:
            self.empty_lse = torch.empty(0, dtype=torch.float32, device=q.device)
        return result, self.empty_lse


def runtime(impl):
    obj = getattr(impl, "_sharedkv_int8_runtime", None)
    if obj is None:
        obj = LayerRuntime(impl)
        impl._sharedkv_int8_runtime = obj
    return obj


def install():
    from vllm_ascend.attention.dsa_v1 import AscendDSAImpl, AscendDSAMetadataBuilder
    from vllm_ascend.models.deepseek_v4.model import AscendDeepseekV4SWACache
    from vllm_ascend.models.layer.attention.layer import DSAAttention

    load_scales()  # Fail before weight loading if the calibration file is absent.
    compact_pages = os.environ.get("SHAREDKV_COMPACT_PAGES") == "1"
    if compact_pages:
        from .cache_layout import install as install_cache_layout
        install_cache_layout()
    if os.environ.get("SHAREDKV_SKIP_SAS_METADATA") == "1":
        from .vllm_metadata import install as install_metadata
        install_metadata()
    original_plan = AscendDSAImpl.get_kv_plan

    def get_plan(self):
        state = runtime(self)
        if state.plan is None:
            state.plan = replace(original_plan(self), sparse_attn_op=state.attention)
        return state.plan

    def write(self, cache, kv, slot_mapping, *, quantize):
        return runtime(self).write(cache, kv, slot_mapping, "cmp" if quantize else "ori")

    AscendDSAImpl.get_kv_plan = get_plan
    AscendDSAImpl._write_kv_cache = write

    for cls in (AscendDeepseekV4SWACache, DSAAttention):
        original_spec = cls.get_kv_cache_spec

        def spec(self, config, _original=original_spec):
            validate_transfer_mode(config, compact_pages)
            old = _original(self, config)
            if old is None:
                return None
            if old.head_size != 512 or old.dtype != torch.bfloat16:
                raise ValueError("INT8 plugin expects the ordinary BF16 D=512 cache specification")
            # The compact planner assigns an explicit common physical stride.
            # The legacy mode retains the BF16 pool's canonical page buckets.
            return replace(old, dtype=torch.int8, page_size_padded=None if compact_pages else old.page_size_bytes,
                           indexes_kv_by_block_stride=True)

        cls.get_kv_cache_spec = spec

    original_metadata = AscendDSAMetadataBuilder._build_sas_metadata

    def metadata(self, *args, **kwargs):
        result = original_metadata(self, *args, **kwargs)
        bound = kwargs.get("max_seqlen_kv", args[5] if len(args) > 5 else None)
        result._sharedkv_max_seq_len = int(bound) if isinstance(bound, int) else None
        return result

    AscendDSAMetadataBuilder._build_sas_metadata = metadata
