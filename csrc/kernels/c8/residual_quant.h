// C8 production core: frozen-channel KV, Q/P double residuals.
#pragma once
#include "vector_math.h"

namespace sharedkv {
// The normalized input is in [-127,127]. Retain its rounding residual in
// another INT8 row: x ~= high + low/127. Both products remain INT8 Cube.
// Callers may pack the residual into unused rows of the same M=16 operation.
__aicore__ inline void RoundResidualInt8(LocalTensor<int8_t> high, LocalTensor<int8_t> low,
                                        LocalTensor<half> halfTmp, LocalTensor<float> x,
                                        LocalTensor<float> residual, uint32_t count)
{
    // Keep x until the residual subtraction. Rounding directly into the
    // scratch buffer avoids copying x solely to preserve its unrounded value.
    // All rounding boundaries and the high/low INT8 protocol stay unchanged.
    Cast(residual, x, RoundMode::CAST_RINT, count);
    PipeBarrier<PIPE_V>();
    Cast(halfTmp, residual, RoundMode::CAST_RINT, count);
    PipeBarrier<PIPE_V>();
    Cast(high, halfTmp, RoundMode::CAST_RINT, count);
    Sub(residual, x, residual, count);
    PipeBarrier<PIPE_V>();
    Muls(residual, residual, 127.0f, count);
    PipeBarrier<PIPE_V>();
    RoundInt8(low, halfTmp, residual, count);
}

__aicore__ inline void AddResidualProduct(LocalTensor<float> high,
                                          LocalTensor<int32_t> low, uint32_t count)
{
    auto f = low.ReinterpretCast<float>();
    Cast(f, low, RoundMode::CAST_RINT, count);
    PipeBarrier<PIPE_V>();
    // dav_c220 FP32 Axpy lowers to vaxpy. Combine residual rescaling and
    // accumulation without an extra full-buffer vector pass.
    Axpy(high, f, 1.0f / 127.0f, count);
    PipeBarrier<PIPE_V>();
}

// A caller-owned contiguous [high, low] INT32 region permits one Cast setup
// for both products. The result occupies its first count FP32 elements;
// the second half retires after Axpy. Rounding and residual scale match
// AddResidualProduct exactly. count-mode Cast is supported on dav_c220.
__aicore__ inline void DecodeResidualProducts(LocalTensor<float> products, uint32_t count)
{
    Cast(products, products.ReinterpretCast<int32_t>(), RoundMode::CAST_RINT, 2 * count);
    PipeBarrier<PIPE_V>();
    Axpy(products, products[count], 1.0f / 127.0f, count);
    PipeBarrier<PIPE_V>();
}
}
