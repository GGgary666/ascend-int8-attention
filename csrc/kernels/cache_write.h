#pragma once
#include <cstdint>

struct CacheWriteArgs {
    uint64_t cache, updates, slots, inverseScale;
    uint64_t pageStride, updateStride, slotStride, slotColumnStride;
    uint32_t tokens, pages, pageSize, int64Slots, pairedSlots;
};
