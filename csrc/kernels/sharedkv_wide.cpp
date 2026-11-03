// Same protocol with 512 selected keys per online-softmax update. K-split
// Cube operations stay within dav_c220's 64 KiB L0A/L0B capacity.
#define SHARED_TILE 512
#define sharedkv_i8_fused sharedkv_i8_fused_wide
#define launch_sharedkv launch_sharedkv_wide
#include "sharedkv.cpp"
