// Share the dense KV window between two queries; keep both Q/P rounding
// residuals in the other 16 physical Cube rows.
#define SHARED_TILE 256
#define SHARED_CUBE_TAILS 1
#define SHARED_QUERY_GROUP 2
#define sharedkv_i8_fused sharedkv_i8_fused_paired
#define launch_sharedkv launch_sharedkv_paired
#include "sharedkv.cpp"
