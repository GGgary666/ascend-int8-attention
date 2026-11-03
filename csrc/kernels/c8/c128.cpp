#include "schedule.h"
#define C8_RATIO 128u
#define SHARED_KERNEL_NAMESPACE c8_dense_narrow
#define SHARED_TILE 128
#define SHARED_QUERY_GROUP 1
#include "attention_impl.h"

#undef SHARED_KERNEL_NAMESPACE
#undef SHARED_TILE
#define SHARED_KERNEL_NAMESPACE c8_dense_medium
#define SHARED_TILE 256
#include "attention_impl.h"

#undef SHARED_KERNEL_NAMESPACE
#undef SHARED_QUERY_GROUP
#undef SHARED_CUBE_TAILS
#define SHARED_KERNEL_NAMESPACE c8_dense_paired
#define SHARED_QUERY_GROUP 2
#define SHARED_CUBE_TAILS 1
#include "attention_impl.h"

#undef SHARED_KERNEL_NAMESPACE
#undef SHARED_QUERY_GROUP
#define SHARED_KERNEL_NAMESPACE c8_dense_packed
#define SHARED_QUERY_GROUP 4
#include "attention_impl.h"

#undef SHARED_KERNEL_NAMESPACE
#undef SHARED_ALTERNATING_WAVES
#define SHARED_KERNEL_NAMESPACE c8_dense_packed_linear
#define SHARED_ALTERNATING_WAVES 0
#include "attention_impl.h"

#undef SHARED_KERNEL_NAMESPACE
#undef SHARED_TILE
#undef SHARED_QUERY_GROUP
#undef SHARED_DENSE_L0_PIPELINE
#define SHARED_KERNEL_NAMESPACE c8_dense_wide
#define SHARED_TILE 512
#define SHARED_QUERY_GROUP 1
#define SHARED_DENSE_L0_PIPELINE 1
#include "attention_impl.h"

// Keep the added phases after existing arithmetic bodies. The same task
// batch is used when both QK operands fit one L0 load. Wider tiles retain
// the per-task plan pending a broadly beneficial resource schedule.
namespace c8_dense_narrow { namespace {
#include "task_batch.h"
} }

// Keep optional batch setup out of the shared dispatcher. Other schedules
// do not execute it, and retain their small common-entry instruction span.
__aicore__ __attribute__((noinline)) void C8DenseNarrow(const SharedKVArgs &p,
    uint32_t uniformQueries, uint32_t queries, uint32_t maximumLength, uint32_t taskOrder)
{
    if (queries > sharedkv_c8::kCores)
        c8_dense_narrow::RunTaskBatch(p, queries, uniformQueries, maximumLength / 128, maximumLength, taskOrder);
    // Pair ownership is only produced above one core wave. This branch
    // has <=24 queries, so keep its ordinary Cube path compile-time unordered.
    else c8_dense_narrow::RunWithSplits(p, 1, 0, uniformQueries);
}

__aicore__ __attribute__((noinline)) void C8DenseMedium(const SharedKVArgs &p,
    uint32_t uniformQueries, uint32_t queries, uint32_t maximumLength, uint32_t taskOrder)
{
    if (queries > sharedkv_c8::kCores && maximumLength >= 128)
        C8DenseNarrow(p, uniformQueries, queries, maximumLength, taskOrder);
    else c8_dense_medium::RunWithSplits(p, 1, 0, uniformQueries);
}
__aicore__ __attribute__((noinline)) void C8DensePaired(const SharedKVArgs &p, uint32_t uniformQueries)
{ c8_dense_paired::RunWithSplits(p, 1, 0, uniformQueries); }
__aicore__ __attribute__((noinline)) void C8DensePacked(const SharedKVArgs &p, uint32_t uniformQueries)
{ c8_dense_packed::RunWithSplits(p, 1, 0, uniformQueries); }
__aicore__ __attribute__((noinline)) void C8DensePackedLinear(const SharedKVArgs &p, uint32_t uniformQueries, uint32_t taskOrder)
{ c8_dense_packed_linear::RunWithSplits(p, 1, taskOrder, uniformQueries); }
__aicore__ __attribute__((noinline)) void C8DenseWide(const SharedKVArgs &p, uint32_t splits, uint32_t taskOrder, uint32_t uniformQueries)
{ c8_dense_wide::RunWithSplits(p, splits, taskOrder, uniformQueries); }

extern "C" __global__ __aicore__ void sharedkv_i8_c8_dense(SharedKVArgs p)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    const auto live = c8_dense_narrow::ReadLiveShape(p);
    sharedkv_c8::Workload shape;
    shape.batch = live.batch; shape.queries = live.tokens;
    const auto w = sharedkv_c8::PlanWorkload(p, shape);
    // Direct lookup amortizes its division when request metadata spans more
    // than one core wave. Small metadata sets retain the short linear scan;
    // this selects an indexing method, never an admitted batch/query limit.
    const uint32_t uniformQueries = w.batch > sharedkv_c8::kCores ? w.uniformQueries : 0;
    using sharedkv_c8::DensePlan;
    switch (sharedkv_c8::DenseSchedule(w)) {
        case DensePlan::Narrow:
            C8DenseNarrow(p, uniformQueries, w.queries, w.maximum, w.taskOrder); break;
        case DensePlan::Medium: C8DenseMedium(p, uniformQueries, w.queries, w.maximum, w.taskOrder); break;
        case DensePlan::Paired: C8DensePaired(p, uniformQueries); break;
        case DensePlan::Packed:
            if (w.packedOrder & sharedkv_c8::kPairedOrder)
                C8DensePackedLinear(p, uniformQueries, w.packedOrder);
            else if (w.packedOrder) C8DensePacked(p, uniformQueries);
            else C8DensePackedLinear(p, uniformQueries, 0);
            break;
        case DensePlan::Wide:
            C8DenseWide(p, sharedkv_c8::SplitCount(w, 1 + (w.maximum / 128 + 511) / 512), w.taskOrder, uniformQueries); break;
    }
}

extern "C" void launch_sharedkv_c8_dense(uint32_t blocks, void *stream, const SharedKVArgs *p)
{ sharedkv_i8_c8_dense<<<blocks, nullptr, stream>>>(*p); }

#include "metadata_impl.h"
