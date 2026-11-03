// C128 decode: avoid a second work wave when four-query groups fit one wave.
// Device metadata is read on every launch, including captured graph replays.
#define SHARED_IMPLEMENTATION_ONLY 1
#define SHARED_KERNEL_NAMESPACE sharedkv_grouped_two
#define SHARED_TILE 256
#define SHARED_CUBE_TAILS 1
#define SHARED_QUERY_GROUP 2
#include "sharedkv.cpp"

#undef SHARED_KERNEL_NAMESPACE
#undef SHARED_QUERY_GROUP
#define SHARED_KERNEL_NAMESPACE sharedkv_grouped_four
#define SHARED_QUERY_GROUP 4
#include "sharedkv.cpp"

static_assert(sharedkv_grouped_two::WORKSPACE_BYTES <= sharedkv_grouped_four::WORKSPACE_BYTES);

// The C4 same-input ablation verified that separate device functions avoid
// the latency introduced by merging both large implementations into one body.
__aicore__ __attribute__((noinline)) void RunGroupedTwo(const SharedKVArgs &p)
{ sharedkv_grouped_two::Run(p); }
__aicore__ __attribute__((noinline)) void RunGroupedFour(const SharedKVArgs &p)
{ sharedkv_grouped_four::Run(p); }

extern "C" __global__ __aicore__ void sharedkv_i8_fused_dynamic_grouped(SharedKVArgs p)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    const uint32_t batch = sharedkv_grouped_two::ActualBatch(p);
    auto cu = sharedkv_grouped_two::Gm<int32_t>(p.cuQ);
    auto lengths = sharedkv_grouped_two::Gm<int32_t>(p.seqLens);
    uint32_t tasks2 = 0, tasks4 = 0, begin = 0, maximum = 0;
    for (uint32_t b = 0; b < batch; ++b) {
        const uint32_t end = cu.GetValue(b + 1);
        const uint32_t count = end - begin;
        tasks2 += (count + 1) / 2;
        tasks4 += (count + 3) / 4;
        const uint32_t length = lengths.GetValue(b);
        maximum = length > maximum ? length : maximum;
        begin = end;
    }
    if (maximum > 32768 && tasks2 > GetBlockNum() && tasks4 <= GetBlockNum())
        RunGroupedFour(p);
    else
        RunGroupedTwo(p);
}

extern "C" void launch_sharedkv_dynamic_grouped(uint32_t blocks, void *stream, const SharedKVArgs *args)
{
    sharedkv_i8_fused_dynamic_grouped<<<blocks, nullptr, stream>>>(*args);
}
