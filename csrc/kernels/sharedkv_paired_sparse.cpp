// Short C4 dense candidate membership, with exact multiplicity and residuals.
#define SHARED_TILE 256
#define SHARED_CUBE_TAILS 1
#define SHARED_QUERY_GROUP 2
#define SHARED_PACKED_C4 1
#define sharedkv_i8_fused sharedkv_i8_fused_paired_sparse
#define launch_sharedkv launch_sharedkv_paired_sparse
#include "sharedkv.cpp"
