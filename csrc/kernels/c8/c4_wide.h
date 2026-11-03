// Shared single-query implementation. Small-capacity launches exclude the
// paired implementation from their instruction image (see msopprof evidence).
#undef SHARED_KERNEL_NAMESPACE
#undef SHARED_TILE
#undef SHARED_QUERY_GROUP
#undef SHARED_PACKED_C4
#undef SHARED_CUBE_TAILS
#define C8_RATIO 4u
#define SHARED_KERNEL_NAMESPACE c8_sparse_wide
#define SHARED_TILE 512
#define SHARED_QUERY_GROUP 1
#define SHARED_PACKED_C4 0
#define SHARED_CUBE_TAILS 1
#include "attention_impl.h"

__aicore__ __attribute__((noinline)) void C8SparseWide(const SharedKVArgs &p)
{
    const uint32_t actual = c8_sparse_wide::ActualTokens(p);
#ifndef C8_SMALL_CAPACITY
    if (actual > sharedkv_c8::kCores && actual <= 2 * sharedkv_c8::kCores) {
        const auto workload = sharedkv_c8::ReadWorkload<4>(p);
        c8_sparse_wide::RunWithSplits(p, 1, workload.taskOrder);
        return;
    }
#endif
    uint32_t splits = 1;
    if (actual && actual <= sharedkv_c8::kCores / 2) {
        const uint32_t batch = c8_sparse_wide::ActualBatch(p);
        auto lengths = c8_sparse_wide::Gm<int32_t>(p.seqLens);
        uint32_t keys = 0;
        for (uint32_t b = 0; b < batch && keys < 1024; ++b) {
            const uint32_t count = lengths.GetValue(b) / 4;
            keys = count > keys ? count : keys;
        }
        keys = keys < 1024 ? keys : 1024;
        const uint32_t tiles = 1 + (keys + 511) / 512;
        const uint32_t budget = sharedkv_c8::kCores / actual;
        splits = budget < tiles ? budget : tiles;
    }
    c8_sparse_wide::RunWithSplits(p, splits);
}
