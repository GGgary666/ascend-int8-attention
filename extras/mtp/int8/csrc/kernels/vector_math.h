#pragma once
#include "kernel_operator.h"

namespace sharedkv {
using namespace AscendC;

// dav_c220 supports FP32->FP32 RINT but not FP32->INT8. Round BEFORE the
// FP16 conversion: integers in [-127,127] are exact in FP16, avoiding double
// rounding around n+0.5. Caller clamps/scales to that interval first.
__aicore__ inline void RoundInt8(LocalTensor<int8_t> out, LocalTensor<half> halfTmp,
                               LocalTensor<float> x, uint32_t count)
{
    Cast(x, x, RoundMode::CAST_RINT, count);
    PipeBarrier<PIPE_V>();
    Cast(halfTmp, x, RoundMode::CAST_RINT, count);
    PipeBarrier<PIPE_V>();
    Cast(out, halfTmp, RoundMode::CAST_RINT, count);
}

__aicore__ inline void VectorToScalar()
{
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
}
}
