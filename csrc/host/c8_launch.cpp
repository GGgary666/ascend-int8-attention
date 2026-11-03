#include <pybind11/pybind11.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include "args.h"

extern "C" void launch_sharedkv_c8_sparse(uint32_t, void *, const SharedKVArgs *);
extern "C" void launch_sharedkv_c8_dense(uint32_t, void *, const SharedKVArgs *);

extern "C" void launch_sharedkv_c8_sparse_small(uint32_t, void *, const SharedKVArgs *);

static void launch(uint32_t blocks, void *stream, uint32_t variant, const SharedKVArgs &args)
{
    if (variant == 20) launch_sharedkv_c8_sparse(blocks, stream, &args);
    else if (variant == 21) launch_sharedkv_c8_dense(blocks, stream, &args);
    else if (variant == 22) launch_sharedkv_c8_sparse_small(blocks, stream, &args);
    else throw std::invalid_argument("Invalid C8 variant");
}

void attention(uintptr_t address, uint32_t blocks, uint32_t variant, int32_t device)
{
    c10_npu::NPUGuard guard(static_cast<c10::DeviceIndex>(device));
    launch(blocks, c10_npu::getCurrentNPUStream().stream(), variant,
           *reinterpret_cast<const SharedKVArgs *>(address));
}

void attention_queued(uintptr_t address, uint32_t blocks, uint32_t variant, int32_t device)
{
    c10_npu::NPUGuard guard(static_cast<c10::DeviceIndex>(device));
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    // Capture argument values before Python can rebind the prepared object.
    const SharedKVArgs args = *reinterpret_cast<const SharedKVArgs *>(address);
    at_npu::native::OpCommand::RunOpApi("sharedkv_i8_c8", [args, blocks, variant, stream]() -> int {
        launch(blocks, stream, variant, args); return 0;
    });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("attention", &attention);
    m.def("attention_queued", &attention_queued);
    m.attr("attention_args_size") = sizeof(SharedKVArgs);
}
