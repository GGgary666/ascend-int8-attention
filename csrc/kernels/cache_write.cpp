#include "vector_math.h"
#include "cache_write.h"

using namespace AscendC;

// Follow upstream scatter_nd_update_sk/arch22's bounds check and explicit
// physical-page stride. This specialized writer fuses static C8 quantization
// with the scatter and accepts unique valid destination slots per invocation.
template<class Index, bool Paired>
__global__ __aicore__ void sharedkv_cache_write(CacheWriteArgs p)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV {
        constexpr uint32_t D = 512, ROWS = 8;
        TPipe pipe;
        TBuf<TPosition::VECCALC> qb, xb, sb, ob;
        pipe.InitBuffer(qb, ROWS * D * 2); pipe.InitBuffer(xb, ROWS * D * 4);
        pipe.InitBuffer(sb, D * 4); pipe.InitBuffer(ob, ROWS * D);
        auto q = qb.Get<bfloat16_t>(); auto halfX = qb.Get<half>();
        auto x = xb.Get<float>(); auto scale = sb.Get<float>(); auto out = ob.Get<int8_t>();
        GlobalTensor<bfloat16_t> updates;
        GlobalTensor<Index> slots;
        GlobalTensor<float> inverseScale;
        GlobalTensor<int8_t> cache;
        updates.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(p.updates));
        slots.SetGlobalBuffer(reinterpret_cast<__gm__ Index *>(p.slots));
        inverseScale.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(p.inverseScale));
        cache.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(p.cache));
        DataCopy(scale, inverseScale, D);
        for (uint32_t firstRow = GetBlockIdx() * ROWS; firstRow < p.tokens; firstRow += GetBlockNum() * ROWS) {
            uint32_t rows = p.tokens - firstRow < ROWS ? p.tokens - firstRow : ROWS;
            DataCopyPad(q, updates[static_cast<uint64_t>(firstRow) * p.updateStride],
                        DataCopyExtParams{static_cast<uint16_t>(rows), D * 2,
                                          static_cast<uint32_t>((p.updateStride - D) * 2), 0, 0},
                        DataCopyPadExtParams<bfloat16_t>{});
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Cast(x, q, RoundMode::CAST_NONE, rows * D); PipeBarrier<PIPE_V>();
            // Upstream sharedkv ScaleChannels uses row repeats to broadcast
            // a channel scale without one instruction loop per token.
            if (rows == ROWS) {
                BinaryRepeatParams rp{};
                rp.dstBlkStride = rp.src0BlkStride = rp.src1BlkStride = 1;
                rp.dstRepStride = rp.src0RepStride = D / 8; rp.src1RepStride = 0;
                for (uint32_t col = 0; col < D; col += 64)
                    Mul(x[col], x[col], scale[col], 64, rows, rp);
            } else for (uint32_t r = 0; r < rows; ++r) Mul(x[r * D], x[r * D], scale, D);
            PipeBarrier<PIPE_V>();
            Mins(x, x, 127.0f, rows * D); Maxs(x, x, -127.0f, rows * D);
            sharedkv::RoundInt8(out, halfX, x, rows * D);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0); WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            for (uint32_t r = 0; r < rows; ++r) {
            uint32_t row = firstRow + r;
            int64_t first = slots.GetValue(static_cast<uint64_t>(row) * p.slotStride);
            if (first < 0) continue;
            int64_t page, offset;
            if constexpr (Paired) {
                page = first;
                offset = slots.GetValue(static_cast<uint64_t>(row) * p.slotStride + p.slotColumnStride);
            } else {
                page = first / p.pageSize;
                offset = first % p.pageSize;
            }
            if (page >= p.pages || offset < 0 || offset >= p.pageSize) continue;
            DataCopy(cache[static_cast<uint64_t>(page) * p.pageStride + offset * D], out[r * D], D);
            }
            PipeBarrier<PIPE_ALL>();
        }
    }
}

extern "C" void launch_cache_write(uint32_t blocks, void *stream, const CacheWriteArgs *args)
{
    if (args->int64Slots) {
        if (args->pairedSlots) sharedkv_cache_write<int64_t, true><<<blocks, nullptr, stream>>>(*args);
        else sharedkv_cache_write<int64_t, false><<<blocks, nullptr, stream>>>(*args);
    } else {
        if (args->pairedSlots) sharedkv_cache_write<int32_t, true><<<blocks, nullptr, stream>>>(*args);
        else sharedkv_cache_write<int32_t, false><<<blocks, nullptr, stream>>>(*args);
    }
}
