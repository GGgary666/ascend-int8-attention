// Captured C4 decode: select partitions from live query/sequence metadata.
#define SHARED_TILE 512
#define SHARED_ADAPTIVE_SPLITS 1
#define sharedkv_i8_fused sharedkv_i8_fused_adaptive
#define launch_sharedkv launch_sharedkv_adaptive
#include "sharedkv.cpp"
