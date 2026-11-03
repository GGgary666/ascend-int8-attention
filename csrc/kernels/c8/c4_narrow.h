// Short sparse selections need no 512-column softmax/quantization tile.
// Duplicate indices and arbitrary ordering still use exact compact/gather.
#undef SHARED_KERNEL_NAMESPACE
#undef SHARED_TILE
#undef SHARED_QUERY_GROUP
#undef SHARED_PACKED_C4
#undef SHARED_CUBE_TAILS
#define SHARED_KERNEL_NAMESPACE c8_sparse_narrow
#define SHARED_TILE 128
#define SHARED_QUERY_GROUP 1
#define SHARED_PACKED_C4 0
#define SHARED_CUBE_TAILS 1
#include "attention_impl.h"

__aicore__ __attribute__((noinline)) void C8SparseNarrow(const SharedKVArgs &p, uint32_t uniformQueries)
{
    // Two short tiles do not amortize SyncAll plus the partial-output pass.
    // Keep the complete online state on one AIC/two-AIV group per query.
    c8_sparse_narrow::RunWithSplits(p, 1);
}
