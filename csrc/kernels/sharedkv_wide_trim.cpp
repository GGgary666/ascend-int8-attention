// Prefill: omit padded Cube work while retaining existing softmax tiling.
#define SHARED_TILE 512
#define SHARED_ADAPTIVE_DENSE 1
#define SHARED_CUBE_TAILS 1
#define SHARED_DENSE_L0_PIPELINE 1
#define sharedkv_i8_fused sharedkv_i8_fused_wide_trim
#define launch_sharedkv launch_sharedkv_wide_trim
#include "sharedkv.cpp"
