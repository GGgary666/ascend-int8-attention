#include "schedule.h"
#define C8_RATIO 4u
#define SHARED_KERNEL_NAMESPACE c8_sparse_paired
#define SHARED_TILE 256
#define SHARED_QUERY_GROUP 2
#define SHARED_PACKED_C4 1
#define SHARED_CUBE_TAILS 1
#include "attention_impl.h"

__aicore__ __attribute__((noinline)) void C8SparsePaired(const SharedKVArgs &p, uint32_t uniformQueries)
{ c8_sparse_paired::RunWithSplits(p, 1, 0, uniformQueries); }

#include "c4_wide.h"
#include "c4_narrow.h"

namespace c8_sparse_narrow { namespace {
#include "task_batch.h"
} }

extern "C" __global__ __aicore__ void sharedkv_i8_c8_sparse(SharedKVArgs p)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    const auto live = c8_sparse_wide::ReadLiveShape(p);
    const uint32_t actual = live.tokens;
    // Choose from the active key extent before doing optional query-stride
    // validation. Wide sparse tasks never consume uniform-query metadata.
    const uint32_t batch = live.batch;
    sharedkv_c8::MetadataSummary summary{};
    const bool vectorMetadata = batch >= sharedkv_c8::kMetadataVectorMinimum && actual > 2 * sharedkv_c8::kCores;
    if (vectorMetadata) summary = sharedkv_c8::ReadVectorMetadata(p, batch);
    auto lengths = c8_sparse_wide::Gm<int32_t>(p.seqLens);
    uint32_t maximumLength = summary.maximum, keys = maximumLength / 4;
    if (!vectorMetadata) {
        for (uint32_t b = 0; b < batch && keys <= sharedkv_c8::kRetainedKeys; ++b) {
            const int32_t length = lengths.GetValue(b);
            if (length > 0 && static_cast<uint32_t>(length) > maximumLength) maximumLength = length;
            keys = maximumLength / 4;
        }
    }
    if (keys <= sharedkv_c8::kRetainedKeys && actual >= sharedkv_c8::kCores) {
        // Once a second core wave exists, independent queries can share
        // Vector setup without sacrificing adjacent-query KV reuse.
        uint32_t uniformQueries = actual > sharedkv_c8::kCores && batch > sharedkv_c8::kCores && actual % batch == 0 ? actual / batch : 0;
        auto cu = c8_sparse_wide::Gm<int32_t>(p.cuQ);
        uint32_t pairedTasks = summary.paired, previous = 0;
        if (vectorMetadata) {
            if (summary.minimumQueries != uniformQueries || summary.maximumQueries != uniformQueries)
                uniformQueries = 0;
        } else {
            for (uint32_t b = 0; b < batch; ++b) {
                const uint32_t end = cu.GetValue(b + 1);
                pairedTasks += (end - previous + 1) / 2;
                if (end != (b + 1) * uniformQueries) uniformQueries = 0;
                previous = end;
            }
        }
        // No adjacent-query KV reuse: batch independent tasks in Vector,
        // retaining independent INT8 Cube products and sparse multiplicities.
        // Wider keys or profitable G2 reuse keep the existing resource plan.
        if (pairedTasks == actual) {
            c8_sparse_narrow::RunTaskBatch(p, actual, uniformQueries, keys, maximumLength);
            return;
        }
        if (keys <= 256) C8SparsePaired(p, uniformQueries);
        else C8SparseWide(p);
    } else if (keys <= 128) C8SparseNarrow(p, 0);
    else C8SparseWide(p);
}

extern "C" void launch_sharedkv_c8_sparse(uint32_t blocks, void *stream, const SharedKVArgs *p)
{ sharedkv_i8_c8_sparse<<<blocks, nullptr, stream>>>(*p); }

#include "metadata_impl.h"
