// SWA/C128 prefill: four adjacent queries (8 local heads each) share a KV
// window union, while every query keeps its own causal/window mask and sink.
#define SHARED_TILE 256
#define SHARED_CUBE_TAILS 1
#define SHARED_QUERY_GROUP 4
#define sharedkv_i8_fused sharedkv_i8_fused_packed
#define launch_sharedkv launch_sharedkv_packed
#include "sharedkv.cpp"
