// One captured entry supports changing live lengths. Both implementations
// use the same by-value ABI and fit the wide plan's workspace allocation.
#define SHARED_IMPLEMENTATION_ONLY 1
#define SHARED_KERNEL_NAMESPACE sharedkv_dynamic_short
#define SHARED_TILE 256
#define SHARED_CUBE_TAILS 1
#define SHARED_QUERY_GROUP 2
#define SHARED_PACKED_C4 1
#include "sharedkv.cpp"

#undef SHARED_KERNEL_NAMESPACE
#undef SHARED_TILE
#undef SHARED_QUERY_GROUP
#undef SHARED_PACKED_C4
#undef SHARED_ADAPTIVE_SPLITS
#undef SHARED_CUBE_TAILS
#define SHARED_KERNEL_NAMESPACE sharedkv_dynamic_long
#define SHARED_TILE 512
#define SHARED_QUERY_GROUP 1
#define SHARED_ADAPTIVE_SPLITS 1
#define SHARED_CUBE_TAILS 1
#include "sharedkv.cpp"

static_assert(sharedkv_dynamic_short::WORKSPACE_BYTES <= sharedkv_dynamic_long::WORKSPACE_BYTES);

// Keep the two implementations as separate device functions. On dav_c220
// the combined entry measured 7-10 us slower even with a precomputed branch
// flag; these wrappers restore the fixed-entry latency. The compiler emits
// actual device calls (verified in the ELF), with the same math and buffers.
__aicore__ __attribute__((noinline)) void RunShort(const SharedKVArgs &p)
{ sharedkv_dynamic_short::Run(p); }
__aicore__ __attribute__((noinline)) void RunLong(const SharedKVArgs &p)
{ sharedkv_dynamic_long::Run(p); }

extern "C" __global__ __aicore__ void sharedkv_i8_fused_dynamic(SharedKVArgs p)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    const uint32_t batch = sharedkv_dynamic_long::ActualBatch(p);
    bool shortContext = sharedkv_dynamic_long::ActualTokens(p) >= 24;
    auto lengths = sharedkv_dynamic_long::Gm<int32_t>(p.seqLens);
    for (uint32_t b = 0; shortContext && b < batch; ++b)
        shortContext = lengths.GetValue(b) <= 1024;
    if (shortContext) RunShort(p);
    else RunLong(p);
}

extern "C" void launch_sharedkv_dynamic(uint32_t blocks, void *stream, const SharedKVArgs *args)
{
    sharedkv_i8_fused_dynamic<<<blocks, nullptr, stream>>>(*args);
}
