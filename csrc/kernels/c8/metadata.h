#pragma once
#include "args.h"
#include "kernel_operator.h"

namespace sharedkv_c8 {
struct MetadataSummary { uint32_t maximum, minimumQueries, maximumQueries, paired; };
// Scalar scans are cheaper below one64-record tile. This is a cost choice,
// never an admission limit: larger arrays stream through fixed512-row UB.
constexpr uint32_t kMetadataVectorMinimum = 64;
// Device implementation is included after the translation unit's entries.
// Keeping this interface small also lets the CPU planner use an independent
// summary provider without pretending to emulate device pipelines.
static __aicore__ __attribute__((noinline, section(".text.sharedkv_metadata")))
MetadataSummary ReadVectorMetadata(const SharedKVArgs &p, uint32_t batch);
}
