#include <pybind11/pybind11.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include "args.h"

// Bind a validated library per prepared operation, while retaining the same
// ordered queue and by-value argument capture as the maintained host bridge.
void attention_queued(uintptr_t address, uint32_t blocks, uint32_t variant,
                      int32_t device, uintptr_t function_address)
{
    c10_npu::NPUGuard guard(static_cast<c10::DeviceIndex>(device));
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    const SharedKVArgs args = *reinterpret_cast<const SharedKVArgs *>(address);
    using Launch = void (*)(uint32_t, void *, const SharedKVArgs *);
    const auto function = reinterpret_cast<Launch>(function_address);
    if (!function) throw std::invalid_argument("Missing validated attention launcher");
    at_npu::native::OpCommand::RunOpApi("sharedkv_i8_c8", [args, blocks, stream, function]() -> int {
        function(blocks, stream, &args);
        return 0;
    });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("attention_queued", &attention_queued);
    m.attr("attention_args_size") = sizeof(SharedKVArgs);
}
