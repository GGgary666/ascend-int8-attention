// Small C128 decode uses the measured fixed512/tail-trim geometry. Cache the
// live shape once across dispatch, Cube/vector tasks, reduction and padding.
#define SHARED_TILE 512
#define SHARED_CUBE_TAILS 1
#define SHARED_ADAPTIVE_DENSE 1
#define SHARED_CACHE_LIVE 1
#define sharedkv_i8_fused sharedkv_i8_fused_dynamic_dense
#define launch_sharedkv launch_sharedkv_dynamic_dense
#include "sharedkv.cpp"
