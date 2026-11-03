// Short-context C4: four queries share a dense candidate tile; membership
// masks preserve each query's actual selected multiset, including duplicates.
#define SHARED_TILE 256
#define SHARED_CUBE_TAILS 1
#define SHARED_QUERY_GROUP 4
#define SHARED_PACKED_C4 1
#define sharedkv_i8_fused sharedkv_i8_fused_packed_sparse
#define launch_sharedkv launch_sharedkv_packed_sparse
#include "sharedkv.cpp"
