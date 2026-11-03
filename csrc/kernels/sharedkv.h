#pragma once
#include <cstdint>

// Host/device ABI. Pointer fields contain device virtual addresses. Strides
// are in INT8 elements (bytes); block tables address pages, not raw offsets.
struct SharedKVArgs {
    uint64_t q, ori, cmp, oriScale, cmpScale, oriTable, cmpTable, indices;
    uint64_t cuQ, seqLens, sinks, out, workspace, lse;
    uint64_t oriStride, cmpStride, partial, oriIndices;
    uint32_t tokens, heads, batch, oriBlock, cmpBlock, oriTableWidth, cmpTableWidth;
    uint32_t ratio, window, topk, splits, oriTopk, oriPages;
    float softmaxScale;
};

// Device-local derived metadata; not part of the host launch ABI.
struct SharedKVLiveShape {
    uint32_t batch, tokens;
};

