#include "schedule.h"
#define c8_sparse_wide c8_sparse_small
#define C8SparseWide C8SparseSmall
#define C8_SMALL_CAPACITY 1
#include "c4_wide.h"
#define c8_sparse_narrow c8_sparse_small_narrow
#define C8SparseNarrow C8SparseSmallNarrow
#include "c4_narrow.h"

extern "C" __global__ __aicore__ void sharedkv_i8_c8_sparse_small(SharedKVArgs p)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    const uint32_t batch = c8_sparse_small::ActualBatch(p);
    auto lengths = c8_sparse_small::Gm<int32_t>(p.seqLens);
    bool narrow = true;
    for (uint32_t b = 0; narrow && b < batch; ++b) narrow = lengths.GetValue(b) / 4 <= 128;
    if (narrow) C8SparseNarrow(p, 0);
    else C8SparseWide(p);
}

extern "C" void launch_sharedkv_c8_sparse_small(uint32_t blocks, void *stream, const SharedKVArgs *p)
{ sharedkv_i8_c8_sparse_small<<<blocks, nullptr, stream>>>(*p); }
