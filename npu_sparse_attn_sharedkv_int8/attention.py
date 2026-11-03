"""Default V65 C4/C128 attention with other-ABI compatibility paths."""

import ctypes

import torch
import torch_npu  # noqa: F401

from .runtime import Kernels, native_launcher


class _Args(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in (
        "q ori cmp oriScale cmpScale oriTable cmpTable indices cuQ seqLens sinks "
        "out workspace lse oriStride cmpStride partial oriIndices").split()] + [
        (name, ctypes.c_uint32) for name in (
            "tokens heads batch oriBlock cmpBlock oriTableWidth cmpTableWidth ratio window topk splits oriTopk oriPages").split()
    ] + [("softmaxScale", ctypes.c_float)]


class PreparedAttention:
    """Retains all input/output storage; run() contains no allocation or readback.

    Initial protocol: BF16 Q[T,H,512], INT8 cache[P,B,1,512], frozen FP32
    per-channel scales for original/compressed caches, symmetric zero offsets.
    """

    def __init__(self, case, ori_scale, cmp_scale=None, *, output=None, splits=None, tile_size=None, query_group=None,
                 max_seq_len=None, return_lse=False, lse_output=None, scratch_allocator=None,
                 adaptive_splits=False, trim_cube_tails=None, dynamic_c4=None, dynamic_c128=None,
                 dynamic_grouped_c128=None, implementation="auto"):
        if implementation not in ("auto", "v65", "fallback"):
            raise ValueError("implementation must be auto, v65 or fallback")
        if case.q.device.type != "npu" or case.q.dtype != torch.bfloat16 or case.q.ndim != 3 or case.q.shape[-1] != 512:
            raise ValueError("Expected BF16 Q[T,H,512] on NPU")
        tokens, heads, _ = case.q.shape
        if not heads or heads > 128 or not case.q.is_contiguous():
            raise ValueError("Expected 1..128 heads and contiguous Q")
        if case.ratio not in (1, 4, 128) or case.window <= 0:
            raise ValueError("Expected SWA/C4/C128 and a positive window")
        device = case.q.device

        def check_cache(cache, table, scale):
            if cache is None or cache.dtype != torch.int8 or cache.ndim != 4 or cache.shape[-2:] != (1, 512):
                raise ValueError("Expected INT8 cache[P,B,1,512]")
            if cache.shape[0] == 0 or cache.shape[1] == 0 or cache.shape[0] * cache.shape[1] >= 2**31 or cache.stride(1) != 512 or cache.stride(-1) != 1 or cache.stride(0) < cache.shape[1] * 512:
                raise ValueError("Invalid page stride or empty cache")
            if table is None or table.ndim != 2 or table.shape[0] != case.seq_lens.numel() or table.dtype != torch.int32 or not table.is_contiguous():
                raise ValueError("Expected contiguous INT32 block table[B,num_pages]")
            if scale is None or scale.shape != (512,) or scale.dtype != torch.float32 or not scale.is_contiguous():
                raise ValueError("Expected frozen contiguous FP32 scale[512]")
            if any(x.device != device for x in (cache, table, scale)):
                raise ValueError("All inputs must be on the same NPU")

        check_cache(case.ori, case.ori_block_table, ori_scale)
        if case.ratio != 1:
            check_cache(case.cmp, case.cmp_block_table, cmp_scale)
        if case.ratio == 4:
            ix = case.cmp_indices
            if ix is None or ix.ndim != 3 or ix.shape[:2] != (tokens, 1) or ix.shape[2] == 0 or ix.dtype != torch.int32 or not ix.is_contiguous():
                raise ValueError("C4 requires contiguous INT32 indices[T,1,topk]")
        if case.cu_q.shape != (case.seq_lens.numel() + 1,) or case.seq_lens.ndim != 1 or case.sink.shape != (heads,):
            raise ValueError("Invalid sequence metadata or sink shape")
        for x, dtype in [(case.cu_q, torch.int32), (case.seq_lens, torch.int32), (case.sink, torch.float32)]:
            if x.dtype != dtype or x.device != device or not x.is_contiguous():
                raise ValueError("Invalid metadata dtype, device or contiguity")
        if case.cmp_indices is not None and case.cmp_indices.device != device:
            raise ValueError("Indices must be on the same NPU")
        if case.ori_indices is not None:
            ix = case.ori_indices
            if (ix.ndim != 3 or ix.shape[:2] != (tokens, 1) or ix.shape[-1] == 0
                    or ix.dtype != torch.int32 or ix.device != device or not ix.is_contiguous()):
                raise ValueError("Original physical slots must be contiguous INT32[T,1,K] on the query device")

        self.case = case
        self.scales = ori_scale, cmp_scale
        self.out = output if output is not None else torch.empty_like(case.q)
        if self.out.shape != case.q.shape or self.out.device != device or self.out.dtype != torch.bfloat16 or not self.out.is_contiguous():
            raise ValueError("Invalid output buffer")
        self.lse = lse_output if lse_output is not None else (
            torch.empty((tokens, heads), dtype=torch.float32, device=device) if return_lse else None)
        if self.lse is not None and (self.lse.shape != (tokens, heads) or self.lse.dtype != torch.float32
                                    or self.lse.device != device or not self.lse.is_contiguous()):
            raise ValueError("Invalid FP32 LSE[T,H] output buffer")
        if tokens == 0:
            self.blocks, self.query_group, self.tile_size, self.splits = 0, 1, 128, 1
            self.adaptive_splits = self.trim_cube_tails = self.dynamic_c4 = False
            self.dynamic_c128 = self.dynamic_grouped_c128 = False
            return
        short_c4 = (case.ratio == 4 and max_seq_len is not None and 0 < max_seq_len <= 1024
                    and case.cmp_indices.shape[-1] <= 1024)
        # Grouped plans reserve half their Cube rows for Q/P residuals.
        preferred_group = 4 if case.ratio != 4 and tokens >= 16 * case.seq_lens.numel() else 2
        auto_group = preferred_group if (case.ori_indices is None and tile_size is None and heads == 8
            and (case.ratio != 4 or short_c4) and tokens >= (96 if case.ratio == 4 else 48)
            and tokens >= 2 * case.seq_lens.numel()) else 1
        self.query_group = query_group if query_group is not None else auto_group
        if self.query_group not in (1, 2, 4) or (self.query_group > 1 and (heads != 8 or (case.ratio == 4 and not short_c4))):
            raise ValueError("Query grouping requires eight heads; C4 also requires a max_seq_len bound <= 1024 and top-k <= 1024")
        if self.query_group > 1 and case.ori_indices is not None:
            raise ValueError("Explicit original slots require ungrouped queries")
        tasks = (tokens + self.query_group - 1) // self.query_group if self.query_group > 1 else tokens * ((heads + 15) // 16)
        long_dense = (case.ratio == 128 and max_seq_len is not None and max_seq_len > 32768
                      and tasks >= 24)
        self.tile_size = tile_size if tile_size is not None else (256 if self.query_group > 1 else
            512 if (case.ratio == 4 and tasks >= 24) or long_dense else 128)
        if self.tile_size not in (128, 256, 512) or (self.query_group > 1 and self.tile_size != 256):
            raise ValueError("Expected 256 keys for grouped queries, or 128/512 keys otherwise")
        max_splits = 1 + (case.cmp_indices.shape[-1] + self.tile_size - 1) // self.tile_size if case.ratio == 4 else 2 if case.ratio == 128 else 1
        self.splits = splits if splits is not None else 1 if self.query_group > 1 else min(max_splits, max(1, 24 // tasks))
        if not isinstance(self.splits, int) or not 1 <= self.splits <= 24:
            raise ValueError("Expected 1..24 KV partitions")
        if self.query_group > 1 and self.splits != 1:
            raise ValueError("Grouped-query kernel requires one KV partition")
        self.blocks = min(24, tasks * self.splits)
        self.adaptive_splits = bool(adaptive_splits)
        if self.adaptive_splits and not (case.ratio == 4 and heads <= 8 and self.query_group == 1
                and self.tile_size == 512 and self.splits == 1 and self.blocks == 24
                and case.cmp_indices.shape[-1] <= 1024 and case.ori_indices is None):
            raise ValueError("Adaptive splits require a 24-core, unsplit, wide C4 plan with <=8 heads and top-k<=1024")
        self.trim_cube_tails = (self.query_group == 1 and self.tile_size == 512 and
                                (tokens >= 96 or long_dense)
                                if trim_cube_tails is None else bool(trim_cube_tails))
        if self.trim_cube_tails and (self.query_group != 1 or self.tile_size != 512 or self.adaptive_splits):
            raise ValueError("Cube tail trimming requires a separate wide, ungrouped plan")
        dynamic_eligible = (case.ratio == 4 and heads == 8 and self.query_group == 1
                            and self.tile_size == 512 and self.splits == 1 and self.blocks == 24
                            and case.cmp_indices.shape[-1] <= 1024 and case.ori_indices is None)
        self.dynamic_c4 = (dynamic_eligible and max_seq_len is None and tile_size is None
                           and query_group is None if dynamic_c4 is None else bool(dynamic_c4))
        if self.dynamic_c4 and not dynamic_eligible:
            raise ValueError("Dynamic C4 requires a 24-core wide plan, eight heads and top-k<=1024")
        dense_eligible = (case.ratio == 128 and heads == 8 and self.query_group == 1
                          and self.tile_size == 128 and self.splits == 1 and self.blocks == 24
                          and case.ori_indices is None)
        self.dynamic_c128 = (dense_eligible and max_seq_len is None and tile_size is None
                             and query_group is None and splits is None
                             if dynamic_c128 is None else bool(dynamic_c128))
        if self.dynamic_c128 and not dense_eligible:
            raise ValueError("Dynamic C128 requires a 24-core narrow plan with eight heads")
        if self.dynamic_c128:
            # Reuse the measured wide geometry; only small-batch KV partitioning
            # changes on device. Cube tail trimming keeps short contexts small.
            self.tile_size = 512
            self.trim_cube_tails = True
        grouped_eligible = (case.ratio == 128 and heads == 8 and self.query_group == 2
                            and self.splits == 1 and self.blocks == 24 and case.ori_indices is None)
        self.dynamic_grouped_c128 = (grouped_eligible and query_group is None and tile_size is None
                                    and splits is None and (max_seq_len is None or max_seq_len > 32768)
                                    if dynamic_grouped_c128 is None else bool(dynamic_grouped_c128))
        if self.dynamic_grouped_c128 and not grouped_eligible:
            raise ValueError("Dynamic C128 grouping requires a 24-core paired-query C128 plan")
        # The captured entry can select G4 from live request boundaries. Both
        # implementations keep their own per-core stride inside this allocation.
        m = 64 if self.dynamic_grouped_c128 else 16 * self.query_group
        workspace_tile = 512 if self.dynamic_c128 else self.tile_size
        workspace_bytes = m * 512 + m * workspace_tile * 5 + m * 512 * 4 + workspace_tile * 512 + 512
        allocate = scratch_allocator or (lambda name, shape, dtype, device: torch.empty(shape, dtype=dtype, device=device))
        ptr = lambda t: t.data_ptr() if t is not None else 0
        self.args = _Args(
            *map(ptr, (case.q, case.ori, case.cmp, ori_scale, cmp_scale,
                       case.ori_block_table, case.cmp_block_table, case.cmp_indices,
                       case.cu_q, case.seq_lens, case.sink, self.out, None)),
            ptr(self.lse), case.ori.stride(0), case.cmp.stride(0) if case.cmp is not None else 0, 0, ptr(case.ori_indices),
            tokens, heads, case.seq_lens.numel(), case.ori.shape[1],
            case.cmp.shape[1] if case.cmp is not None else 0, case.ori_block_table.shape[1],
            case.cmp_block_table.shape[1] if case.cmp is not None else 0,
            case.ratio, case.window, case.cmp_indices.shape[-1] if case.cmp_indices is not None else 0, self.splits,
            case.ori_indices.shape[-1] if case.ori_indices is not None else 0, case.ori.shape[0],
            case.softmax_scale,
        )
        from . import c8_attention
        select_c8 = (implementation == "v65" or (implementation == "auto" and c8_attention.enabled()))
        if select_c8 and c8_attention.supported(self):
            # Allocate for the selected implementation before reserving any
            # legacy scratch. Large graph capacities can replay <=12 live
            # queries, but only those queries can ever select split-KV.
            self.workspace = allocate("workspace", (c8_attention.WORKSPACE_BYTES,), torch.uint8, device)
            self.partial = allocate("partial", (min(tokens, c8_attention.CORES // 2),
                heads, c8_attention.MAX_SPLITS, 520), torch.float32, device)
        else:
            self.workspace = allocate("workspace", (self.blocks, workspace_bytes), torch.uint8, device)
            partial_splits = 4 if self.dynamic_c128 else 3 if self.adaptive_splits or self.dynamic_c4 else self.splits
            self.partial = allocate("partial", (tokens, heads, partial_splits, 520), torch.float32, device) if partial_splits > 1 else None
        self.args.workspace, self.args.partial = ptr(self.workspace), ptr(self.partial)
        self.kernels = Kernels()
        entry = ("launch_sharedkv_packed_sparse" if case.ratio == 4 else "launch_sharedkv_packed") if self.query_group == 4 else {
            128: "launch_sharedkv", 256: "launch_sharedkv_medium", 512: "launch_sharedkv_wide"
        }[self.tile_size]
        if self.query_group == 2:
            entry = "launch_sharedkv_paired_sparse" if case.ratio == 4 else "launch_sharedkv_paired"
        if self.adaptive_splits:
            entry = "launch_sharedkv_adaptive"
        elif self.trim_cube_tails:
            entry = "launch_sharedkv_wide_trim"
        if self.dynamic_c4:
            entry = "launch_sharedkv_dynamic"
        if self.dynamic_c128:
            # The wide entry already selects C128 KV partitions from live
            # device metadata. Reuse it: the separate cached-live entry has
            # identical N512 math but measured higher latency on B4/135K.
            entry = "launch_sharedkv_wide_trim"
        if self.dynamic_grouped_c128:
            entry = "launch_sharedkv_dynamic_grouped"
        self.launch = getattr(self.kernels.lib, entry)
        self.launch.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(_Args)]
        self.launch.restype = None
        self.native = native_launcher()
        self.argument_address = ctypes.addressof(self.args)
        self.device_index = device.index
        self.variant = (4 if case.ratio == 4 else 3) if self.query_group == 4 else {128: 0, 256: 1, 512: 2}[self.tile_size]
        if self.query_group == 2:
            self.variant = 8 if case.ratio == 4 else 7
        if self.adaptive_splits:
            self.variant = 5
        elif self.trim_cube_tails:
            self.variant = 6
        if self.dynamic_c4:
            self.variant = 9
        if self.dynamic_c128:
            self.variant = 6
        if self.dynamic_grouped_c128:
            self.variant = 11
        if self.native is not None and self.native.attention_args_size != ctypes.sizeof(_Args):
            raise RuntimeError("Native host ABI changed; rebuild with ./build.sh")
        if self.native is not None and self.variant >= getattr(self.native, "attention_variant_count", 5):
            raise RuntimeError("Native host launcher lacks this kernel variant; rebuild with ./build.sh")
        if select_c8:
            selected = c8_attention.prepare(self, scratch_allocator=scratch_allocator)
            if selected is self and implementation == "v65":
                raise ValueError("This ABI is outside V65 C4/C128; use implementation='auto' for compatibility")
            self.__dict__.update(selected.__dict__)

    def _rebind_same_geometry(self, case, *, output=None, lse_output=None):
        """Internal serving fast path; the adapter checks the geometry key."""
        self.case = case
        if output is not None:
            self.out, self.lse = output, lse_output
        if not self.blocks:
            return
        if output is not None:
            self.args.out = output.data_ptr()
            self.args.lse = lse_output.data_ptr() if lse_output is not None else 0
        for name, tensor in (("q", case.q), ("ori", case.ori), ("cmp", case.cmp),
                             ("oriTable", case.ori_block_table), ("cmpTable", case.cmp_block_table),
                             ("indices", case.cmp_indices), ("cuQ", case.cu_q),
                             ("seqLens", case.seq_lens), ("sinks", case.sink), ("oriIndices", case.ori_indices)):
            setattr(self.args, name, tensor.data_ptr() if tensor is not None else 0)

    def run(self):
        if self.blocks == 0:
            return self.out
        if self.native is not None:
            self.native.attention(self.argument_address, self.blocks, self.variant, self.device_index)
            return self.out
        with torch.npu.device(self.case.q.device):
            self.launch(self.blocks, torch.npu.current_stream().npu_stream, ctypes.byref(self.args))
        return self.out
