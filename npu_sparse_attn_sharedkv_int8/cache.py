"""Frozen-scale INT8 shared-KV cache writes on the caller's NPU stream."""

import ctypes
import torch
import torch_npu  # noqa: F401

from .runtime import Kernels, native_launcher


class _Args(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in (
        "cache updates slots inverseScale pageStride updateStride slotStride slotColumnStride").split()] + [
        (name, ctypes.c_uint32) for name in "tokens pages pageSize int64Slots pairedSlots".split()]


class PreparedCacheWrite:
    """Write unique valid slots; negative/out-of-range destinations are skipped.

    inverse_scale is frozen FP32[512], prepared once from calibrated scales.
    Slots may be INT32/INT64 flat physical IDs or [page, offset] pairs.
    No host reads, tensor allocations or scale changes occur in run().
    """

    def __init__(self, cache, updates, slots, inverse_scale):
        device = cache.device
        if device.type != "npu" or cache.dtype != torch.int8 or cache.ndim != 4 or cache.shape[-2:] != (1, 512):
            raise ValueError("Expected INT8 NPU cache[pages,page_size,1,512]")
        if (not cache.shape[0] or not cache.shape[1] or cache.stride(1) != 512 or cache.stride(-1) != 1
                or cache.stride(0) < cache.shape[1] * 512):
            raise ValueError("Invalid cache strides or empty page storage")
        if (updates.dtype != torch.bfloat16 or updates.ndim not in (2, 3) or updates.shape[-1] != 512
                or (updates.ndim == 3 and updates.shape[1] != 1) or updates.stride(-1) != 1
                or updates.stride(0) < 512):
            raise ValueError("Expected BF16 updates[T,512] or [T,1,512] with contiguous rows")
        if (slots.dtype not in (torch.int32, torch.int64) or slots.ndim not in (1, 2)
                or (slots.ndim == 2 and slots.shape[1] != 2) or slots.shape[0] != updates.shape[0]):
            raise ValueError("Expected INT32/INT64 slots[T] or slots[T,2]")
        if inverse_scale.dtype != torch.float32 or inverse_scale.shape != (512,) or not inverse_scale.is_contiguous():
            raise ValueError("Expected frozen contiguous FP32 inverse_scale[512]")
        if any(t.device != device for t in (updates, slots, inverse_scale)):
            raise ValueError("Cache, updates, slots and scale must be on the same NPU")
        self.cache, self.updates, self.slots, self.inverse_scale = cache, updates, slots, inverse_scale
        self.blocks = min(48, (updates.shape[0] + 7) // 8)
        self.args = _Args(cache.data_ptr(), updates.data_ptr(), slots.data_ptr(), inverse_scale.data_ptr(),
                          cache.stride(0), updates.stride(0), slots.stride(0), slots.stride(1) if slots.ndim == 2 else 0,
                          updates.shape[0], cache.shape[0], cache.shape[1], slots.dtype == torch.int64, slots.ndim == 2)
        self.kernels = Kernels()
        self.launch = self.kernels.lib.launch_cache_write
        self.launch.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(_Args)]
        self.launch.restype = None
        self.native = native_launcher()
        self.argument_address = ctypes.addressof(self.args)
        self.device_index = device.index
        if self.native is not None and self.native.cache_args_size != ctypes.sizeof(_Args):
            raise RuntimeError("Native host ABI changed; rebuild with ./build.sh")

    def _rebind_same_geometry(self, cache, updates, slots):
        """Internal serving fast path; token count may vary, row layout may not."""
        self.cache, self.updates, self.slots = cache, updates, slots
        self.args.cache, self.args.updates, self.args.slots = cache.data_ptr(), updates.data_ptr(), slots.data_ptr()
        self.args.tokens = updates.shape[0]
        self.blocks = min(48, (updates.shape[0] + 7) // 8)

    def run(self):
        if self.blocks:
            if self.native is not None:
                self.native.cache_write(self.argument_address, self.blocks, self.device_index)
                return self.cache
            with torch.npu.device(self.cache.device):
                self.launch(self.blocks, torch.npu.current_stream().npu_stream, ctypes.byref(self.args))
        return self.cache
