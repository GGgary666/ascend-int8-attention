#include <pybind11/pybind11.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include "sharedkv.h"
#include "cache_write.h"

extern "C" {
void launch_sharedkv(uint32_t, void *, const SharedKVArgs *);
void launch_sharedkv_medium(uint32_t, void *, const SharedKVArgs *);
void launch_sharedkv_wide(uint32_t, void *, const SharedKVArgs *);
void launch_sharedkv_packed(uint32_t, void *, const SharedKVArgs *);
void launch_sharedkv_packed_sparse(uint32_t, void *, const SharedKVArgs *);
void launch_sharedkv_adaptive(uint32_t, void *, const SharedKVArgs *);
void launch_sharedkv_wide_trim(uint32_t, void *, const SharedKVArgs *);
void launch_sharedkv_paired(uint32_t, void *, const SharedKVArgs *);
void launch_sharedkv_paired_sparse(uint32_t, void *, const SharedKVArgs *);
void launch_sharedkv_dynamic(uint32_t, void *, const SharedKVArgs *);
void launch_sharedkv_dynamic_dense(uint32_t, void *, const SharedKVArgs *);
void launch_sharedkv_dynamic_grouped(uint32_t, void *, const SharedKVArgs *);
void launch_cache_write(uint32_t, void *, const CacheWriteArgs *);
}

// Internal prepared-launch API. Python owners retain the argument structs
// and all tensors until queued work completes. Current-stream lookup follows
// vllm-ascend's torch_binding.cpp, so graph/auxiliary streams remain correct.
void attention(uintptr_t argument, uint32_t blocks, uint32_t variant, int32_t device)
{
    c10_npu::NPUGuard guard(static_cast<c10::DeviceIndex>(device));
    auto stream = c10_npu::getCurrentNPUStream().stream();
    auto args = reinterpret_cast<const SharedKVArgs *>(argument);
    switch (variant) {
        case 0: launch_sharedkv(blocks, stream, args); break;
        case 1: launch_sharedkv_medium(blocks, stream, args); break;
        case 2: launch_sharedkv_wide(blocks, stream, args); break;
        case 3: launch_sharedkv_packed(blocks, stream, args); break;
        case 4: launch_sharedkv_packed_sparse(blocks, stream, args); break;
        case 5: launch_sharedkv_adaptive(blocks, stream, args); break;
        case 6: launch_sharedkv_wide_trim(blocks, stream, args); break;
        case 7: launch_sharedkv_paired(blocks, stream, args); break;
        case 8: launch_sharedkv_paired_sparse(blocks, stream, args); break;
        case 9: launch_sharedkv_dynamic(blocks, stream, args); break;
        case 10: launch_sharedkv_dynamic_dense(blocks, stream, args); break;
        case 11: launch_sharedkv_dynamic_grouped(blocks, stream, args); break;
        default: throw std::invalid_argument("Unknown shared-KV kernel variant");
    }
}

void cache_write(uintptr_t argument, uint32_t blocks, int32_t device)
{
    c10_npu::NPUGuard guard(static_cast<c10::DeviceIndex>(device));
    launch_cache_write(blocks, c10_npu::getCurrentNPUStream().stream(),
                       reinterpret_cast<const CacheWriteArgs *>(argument));
}

// Use torch_npu's ordered host queue, as in vllm-ascend's
// grouped_matmul_situ_quant/utils/torch_kernel_helper.h::EXEC_KERNEL_CMD.
// stream() drains that queue; stream(false) is safe here because RunOpApi
// submits the launch through the same queue as preceding torch operators.
void attention_queued(uintptr_t argument, uint32_t blocks, uint32_t variant, int32_t device)
{
    if (variant > 11) throw std::invalid_argument("Unknown shared-KV kernel variant");
    c10_npu::NPUGuard guard(static_cast<c10::DeviceIndex>(device));
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    // Prepared Python arguments may be rebound before the queue runs. Copy
    // their VALUES into the callback, never retain the ctypes host pointer.
    SharedKVArgs args = *reinterpret_cast<const SharedKVArgs *>(argument);
    at_npu::native::OpCommand::RunOpApi("sharedkv_i8_attention", [args, blocks, variant, stream]() -> int {
        switch (variant) {
            case 0: launch_sharedkv(blocks, stream, &args); break;
            case 1: launch_sharedkv_medium(blocks, stream, &args); break;
            case 2: launch_sharedkv_wide(blocks, stream, &args); break;
            case 3: launch_sharedkv_packed(blocks, stream, &args); break;
            case 4: launch_sharedkv_packed_sparse(blocks, stream, &args); break;
            case 5: launch_sharedkv_adaptive(blocks, stream, &args); break;
            case 6: launch_sharedkv_wide_trim(blocks, stream, &args); break;
            case 7: launch_sharedkv_paired(blocks, stream, &args); break;
            case 8: launch_sharedkv_paired_sparse(blocks, stream, &args); break;
            case 9: launch_sharedkv_dynamic(blocks, stream, &args); break;
            case 10: launch_sharedkv_dynamic_dense(blocks, stream, &args); break;
            case 11: launch_sharedkv_dynamic_grouped(blocks, stream, &args); break;
        }
        return 0;
    });
}

void cache_write_queued(uintptr_t argument, uint32_t blocks, int32_t device)
{
    c10_npu::NPUGuard guard(static_cast<c10::DeviceIndex>(device));
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    CacheWriteArgs args = *reinterpret_cast<const CacheWriteArgs *>(argument);
    at_npu::native::OpCommand::RunOpApi("sharedkv_i8_cache_write", [args, blocks, stream]() -> int {
        launch_cache_write(blocks, stream, &args);
        return 0;
    });
}

int64_t stream_id(int32_t device)
{
    // Identity lookup only. Do not call stream(), which drains the task queue,
    // or construct a Python Stream, which repeats generic device detection.
    return c10_npu::getCurrentNPUStream(static_cast<c10::DeviceIndex>(device)).id();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("attention", &attention, pybind11::call_guard<pybind11::gil_scoped_release>());
    m.def("cache_write", &cache_write, pybind11::call_guard<pybind11::gil_scoped_release>());
    m.def("attention_queued", &attention_queued, pybind11::call_guard<pybind11::gil_scoped_release>());
    m.def("cache_write_queued", &cache_write_queued, pybind11::call_guard<pybind11::gil_scoped_release>());
    m.def("stream_id", &stream_id);
    m.attr("attention_args_size") = sizeof(SharedKVArgs);
    m.attr("attention_variant_count") = 12;
    m.attr("cache_args_size") = sizeof(CacheWriteArgs);
}
