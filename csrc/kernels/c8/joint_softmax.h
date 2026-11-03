// Fold 64-lane chunks, then reduce rows: vLLM Ascend arch22
// attn_res_fwd/op_kernel/arch22/reduce_common.h uses the same fold principle.
// A joint tile has only its sink as history; ordinary tiles retain an
// explicit online max/sum and the factor needed to rescale prior PV output.
#pragma once
#include "kernel_operator.h"

namespace sharedkv {
using namespace AscendC;
struct SoftmaxBuffers {
    uint32_t x, sum, maximum, exponent, scratch;
};

template<uint32_t RowCapacity, uint32_t ScalarStride, bool Online = false>
__aicore__ __attribute__((always_inline)) inline
void CompactSoftmax(uint32_t rows, uint32_t width, const SoftmaxBuffers &buf, uint32_t state = 0)
{
    static_assert(ScalarStride >= RowCapacity && ScalarStride % 8 == 0);
    // Callers reserve at least this many bytes of dead softmax scratch.
    constexpr uint32_t scratchElements = RowCapacity * 64 + ScalarStride * 8;
    LocalTensor<float> x(TPosition::VECCALC, buf.x, RowCapacity * 512);
    LocalTensor<float> sum(TPosition::VECCALC, buf.sum, 2 * ScalarStride);
    LocalTensor<float> maximum(TPosition::VECCALC, buf.maximum, 2 * ScalarStride);
    LocalTensor<float> exponent(TPosition::VECCALC, buf.exponent, ScalarStride);
    LocalTensor<float> scratch(TPosition::VECCALC, buf.scratch, scratchElements);
    auto brc = scratch[RowCapacity * 64];
    const uint32_t next = Online ? state ^ ScalarStride : ScalarStride;
    auto oldMax = maximum[Online ? state : 0];
    auto rowMax = maximum[next], rowSum = sum[next];
    BinaryRepeatParams fold{1, 1, 1, 8, static_cast<uint8_t>(width / 8), static_cast<uint8_t>(width / 8)};
    if (width > 64) {
        Max(scratch, x, x[64], 64, rows, fold); PipeBarrier<PIPE_V>();
        fold.src0RepStride = 8;
        for (uint32_t col = 128; col < width; col += 64) {
            Max(scratch, scratch, x[col], 64, rows, fold); PipeBarrier<PIPE_V>();
        }
    }
    ReduceRepeat<ReduceType::MAX>(rowMax, width == 64 ? x : scratch, 64, rows, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
    PipeBarrier<PIPE_V>();
    Max(rowMax, rowMax, oldMax, rows); PipeBarrier<PIPE_V>();
    Sub(exponent, oldMax, rowMax, rows); PipeBarrier<PIPE_V>();
    if constexpr (Online) {
        Exp(exponent, exponent, rows); PipeBarrier<PIPE_V>();
        Mul(rowSum, exponent, sum[state], rows);
    } else {
        Exp(rowSum, exponent, rows);
    }
    // Broadcast only live rows; the padded scalar stride still owns storage.
    Brcb(brc, rowMax, (rows + 7) / 8, {1, 8}); PipeBarrier<PIPE_V>();
    BinaryRepeatParams sub{1, 1, 0, 8, 8, 0};
    if (width / 64 <= rows) {
        sub.dstRepStride = sub.src0RepStride = width / 8; sub.src1RepStride = 1;
        for (uint32_t col = 0; col < width; col += 64)
            Sub(x[col], x[col], brc, 64, rows, sub);
    } else {
        for (uint32_t row = 0; row < rows; ++row)
            Sub(x[row * width], x[row * width], brc[row * 8], 64, width / 64, sub);
    }
    PipeBarrier<PIPE_V>(); Exp(x, x, rows * width); PipeBarrier<PIPE_V>();
    fold.src0RepStride = width / 8;
    if (width > 64) {
        Add(scratch, x, x[64], 64, rows, fold); PipeBarrier<PIPE_V>();
        fold.src0RepStride = 8;
        for (uint32_t col = 128; col < width; col += 64) {
            Add(scratch, scratch, x[col], 64, rows, fold); PipeBarrier<PIPE_V>();
        }
    }
    // The broadcast maximum is dead after all score subtraction finishes.
    // Reuse its aligned scratch for the new tile sum, preserving the online
    // rescale factor in exponent until the caller updates the PV accumulator.
    auto tileSum = Online ? brc : exponent;
    ReduceRepeat<ReduceType::SUM>(tileSum, width == 64 ? x : scratch, 64, rows, 1, 1, 8);
    PipeBarrier<PIPE_V>(); Add(rowSum, rowSum, tileSum, rows);
}
}
