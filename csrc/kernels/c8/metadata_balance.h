#pragma once
#include "task_partition.h"

namespace sharedkv_c8 {
// These helpers run inside the existing metadata TPipe, before its single
// publication. The full raw length tile survives the query-count summary.
// One cached tile favors Scalar reads; longer streams use bounded Vector
// reductions/scan. All global loads and sums remain exact integers.
static __aicore__ __attribute__((noinline, section(".text.sharedkv_metadata_balance")))
uint32_t BalanceCachedMetadata(AscendC::LocalTensor<int32_t> lengths, uint32_t batch, uint32_t ratio)
{
    uint64_t loads[24] = {}, blockLoad = 0, blockedPeak = 0, cyclicPeak = 0;
    uint32_t rank = 1, cut = batch / 24, owner = 0;
    for (uint32_t i = 0; i < batch; ++i) {
        if (i == cut && rank < 24) {
            blockedPeak = blockLoad > blockedPeak ? blockLoad : blockedPeak;
            blockLoad = 0; ++rank; cut = uint64_t(batch) * rank / 24;
        }
        const int32_t raw = lengths.GetValue(i);
        const uint32_t cost = RetainedQueryCost(raw > 0 ? uint32_t(raw) : 0, ratio);
        loads[owner] += cost; blockLoad += cost;
        if (++owner == 24) owner = 0;
    }
    blockedPeak = blockLoad > blockedPeak ? blockLoad : blockedPeak;
    for (uint32_t i = 0; i < 24; ++i) cyclicPeak = loads[i] > cyclicPeak ? loads[i] : cyclicPeak;
    return PreferContiguous(cyclicPeak, blockedPeak);
}

static __aicore__ __attribute__((noinline, section(".text.sharedkv_metadata_balance")))
uint32_t BalanceStreamedMetadata(const SharedKVArgs &p, uint32_t batch, AscendC::LocalTensor<int32_t> all)
{
    using namespace AscendC;
    constexpr uint32_t chunk = 384;
    // [256 zero guard][512 costs][512 tmp][512 byte offsets]. All views
    // are disjoint and32B aligned; Gather's relative offsets are4B aligned.
    auto costs = all[256], tmp = all[768], indices = all[1280];
    SetFlag<HardEvent::S_V>(EVENT_ID0); WaitFlag<HardEvent::S_V>(EVENT_ID0);
    Duplicate(all, int32_t(0), 256);
    ArithProgression<int32_t>(indices, int32_t(0), int32_t(4), chunk); PipeBarrier<PIPE_V>();
    auto source = AscendC::GlobalTensor<int32_t>();
    source.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(p.seqLens));
    uint64_t loads[24] = {}, prefix = 0, boundaryCost = 0, blockedPeak = 0, cyclicPeak = 0;
    uint32_t rank = 1;
    for (uint32_t base = 0; base < batch;) {
        const uint32_t count = batch - base < chunk ? batch - base : chunk;
        SetFlag<HardEvent::S_MTE2>(EVENT_ID0); WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0); WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        DataCopyPad(costs, source[base], DataCopyExtParams{1, count * 4, 0, 0, 0},
                    DataCopyPadExtParams<int32_t>{true, 0, uint8_t((8-count%8)%8), 0});
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Maxs(costs, costs, int32_t(0), count); PipeBarrier<PIPE_V>();
        Mins(tmp, costs, int32_t(128), count);
        ShiftRight(costs.ReinterpretCast<uint32_t>(), costs.ReinterpretCast<uint32_t>(), p.ratio == 4 ? 2u : 7u, count);
        PipeBarrier<PIPE_V>(); Mins(costs, costs, int32_t(kRetainedKeys), count);
        Adds(tmp, tmp, int32_t(63), count); PipeBarrier<PIPE_V>();
        Adds(costs, costs, int32_t(63), count);
        ShiftRight(tmp.ReinterpretCast<uint32_t>(), tmp.ReinterpretCast<uint32_t>(), 6u, count);
        PipeBarrier<PIPE_V>();
        ShiftRight(costs.ReinterpretCast<uint32_t>(), costs.ReinterpretCast<uint32_t>(), 6u, count);
        PipeBarrier<PIPE_V>(); Add(costs, costs, tmp, count); PipeBarrier<PIPE_V>();
        Adds(costs, costs, int32_t(4), count); PipeBarrier<PIPE_V>();
        const uint32_t aligned = (count + 7) / 8 * 8;
        if (aligned < chunk) Duplicate(costs[aligned], int32_t(0), chunk-aligned);
        PipeBarrier<PIPE_V>(); DataCopy(tmp, costs, chunk); PipeBarrier<PIPE_V>();
        for (uint32_t half = chunk/2; half >= 24; half /= 2) {
            Add(tmp, tmp, tmp[half], half); PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
        for (uint32_t i = 0; i < 24; ++i) loads[i] += uint32_t(tmp.GetValue(i));
        SetFlag<HardEvent::S_V>(EVENT_ID0); WaitFlag<HardEvent::S_V>(EVENT_ID0);
        for (uint32_t distance = 1; distance < chunk; distance *= 2) {
            Gather(tmp, all, indices.ReinterpretCast<uint32_t>(), 4 * (256-distance), count);
            PipeBarrier<PIPE_V>(); Add(costs, costs, tmp, count); PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
        while (rank <= 24) {
            const uint32_t cut = uint64_t(batch) * rank / 24;
            if (cut > base + count) break;
            const uint64_t sum = cut == base ? prefix : prefix + uint32_t(costs.GetValue(cut-base-1));
            const uint64_t load = sum - boundaryCost;
            blockedPeak = load > blockedPeak ? load : blockedPeak;
            boundaryCost = sum; ++rank;
        }
        prefix += uint32_t(costs.GetValue(count-1)); base += count;
    }
    for (uint32_t i = 0; i < 24; ++i) cyclicPeak = loads[i] > cyclicPeak ? loads[i] : cyclicPeak;
    return PreferContiguous(cyclicPeak, blockedPeak);
}
}
