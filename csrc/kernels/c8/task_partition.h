#pragma once
#include "kernel_operator.h"

namespace sharedkv_c8 {
// DAV2201 A1/B1 share L1. Budget Q, compact P and KV by actual task width;
// reserve128B for TPipe. These are resource limits, never admission limits.
constexpr uint32_t kRetainedL1Bytes = 512 * 1024 - 128;
constexpr uint32_t kRetainedMatrixRows = 16, kRetainedChannels = 512;
constexpr uint32_t kRetainedTasks = 5;
constexpr uint32_t kRetainedKeys =
    (kRetainedL1Bytes - kRetainedMatrixRows * kRetainedChannels) /
    (kRetainedChannels + kRetainedMatrixRows) / 32 * 32;
struct RetainedResources {
    uint32_t width, scoreColumns, residentRows;
    bool reuseKvSlot;
};
// Callers prove aligned rows fit kRetainedKeys before choosing this plan.
// Metadata publication and both consumers use this same width predicate.
__aicore__ inline RetainedResources PlanRetainedRows(uint32_t originalRows, uint32_t compressedRows)
{
    const uint32_t score = compressedRows <= 128 ? 128 : compressedRows <= 256 ? 256 :
        compressedRows <= 512 ? 512 : compressedRows <= 768 ? 768 :
        compressedRows <= 896 ? 896 : kRetainedKeys;
    const uint32_t sum = originalRows + compressedRows;
    // Every region is a multiple of512B. Count feasible widths in the fixed
    // five-task UB envelope instead of performing a runtime integer divide.
    constexpr uint32_t units = kRetainedL1Bytes / kRetainedChannels;
    const uint32_t perTask = kRetainedMatrixRows + score / 32 + sum;
    const bool reuse = perTask > units;
    const uint32_t resident = reuse ? (originalRows > compressedRows ? originalRows : compressedRows) : sum;
    static_assert(kRetainedTasks == 5);
    const uint32_t width = 1 + (perTask <= units / 2) + (perTask <= units / 3) +
        (perTask <= units / 4) + (perTask <= units / 5);
    return {width, score, resident, reuse};
}
// Aligned work estimate for a single retained query (width=1). Wider
// retained batches share column maxima and must not use this additive model.
__aicore__ inline uint32_t RetainedQueryCost(uint32_t length, uint32_t ratio)
{
    const uint32_t original = length < 128 ? length : 128;
    const uint32_t keys = ratio == 4 ? length >> 2 : length >> 7;
    const uint32_t compressed = keys < kRetainedKeys ? keys : kRetainedKeys;
    return 4 + (original + 63) / 64 + (compressed + 63) / 64;
}
__aicore__ inline bool PreferContiguous(uint64_t cyclic, uint64_t contiguous)
{
    // Require headroom for planning and address setup; this is a cost
    // comparison, independent of any request count or known test geometry.
    return contiguous + 4 <= cyclic && 10 * contiguous <= 9 * cyclic;
}
struct TaskRange { uint32_t first, count, stride; };
// Decode the planner's two independent task IDs once, outside hot loops.
// Sorting within a pair keeps stride nonnegative without changing ownership
// or peak cost. 63 denotes a core with only one task (IDs are below48).
__aicore__ inline TaskRange PairedTaskRange(uint32_t order)
{
    const uint32_t first = order & 63u, second = (order >> 6) & 63u;
    if (second == 63u) return {first, 1, 1};
    return first < second ? TaskRange{first, 2, second - first}
                          : TaskRange{second, 2, first - second};
}
__aicore__ inline TaskRange PartitionTasks(uint32_t tasks, uint32_t core, uint32_t cores, bool contiguous)
{
    if (contiguous) {
        const uint32_t first = uint64_t(tasks) * core / cores;
        const uint32_t end = uint64_t(tasks) * (core + 1) / cores;
        return {first, end - first, 1};
    }
    return {core, core < tasks ? 1 + (tasks - 1 - core) / cores : 0, cores};
}
}
