#pragma once
#include "metadata.h"
#include "metadata_balance.h"

namespace sharedkv_c8 {
constexpr uint32_t kMetadataChunk = 512;
constexpr uint16_t kMetadataReady = 0;

template<class T> __aicore__ inline AscendC::GlobalTensor<T> MetadataGm(uint64_t address)
{
    AscendC::GlobalTensor<T> value;
    value.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(address)); return value;
}

// Called only above48 live queries, where every attention plan has one
// split and never touches partial. Each AIC group owns an independent512B
// slot there. Both AIVs compute locally; sub0 transfers four exact summary
// words to AIC. Eligible retained plans also publish a mode in word4, with
// a separate record for each AIV. The20KiB arena retires before attention.
static __aicore__ __attribute__((noinline, section(".text.sharedkv_metadata")))
MetadataSummary ReadVectorMetadata(const SharedKVArgs &p, uint32_t batch)
{
    using namespace AscendC;
    MetadataSummary s{0, 0x7fffffffu, 0, 0};
    uint32_t core = GetBlockIdx();
    if ASCEND_IS_AIV { core /= 2; }
    const uint64_t transfer = p.partial + static_cast<uint64_t>(core) * 512;
    if ASCEND_IS_AIV {
        TPipe pipe;
        TBuf<TPosition::VECCALC> arena;
        pipe.InitBuffer(arena, 10 * kMetadataChunk * sizeof(int32_t));
        auto all = arena.Get<int32_t>();
        // Keep raw seq separate from counts so a single cached tile can
        // be reused by the optional cost comparison. cu retires after Sub;
        // its520 aligned words cannot overlap the byte-offset indices.
        auto lengths = all[8*kMetadataChunk], ends = all[0], starts = all[kMetadataChunk];
        auto tmp = all[kMetadataChunk], indices = all[2*kMetadataChunk + 8];
        auto maxLength = all[4*kMetadataChunk];
        auto minCount = all[5*kMetadataChunk], maxCount = all[6*kMetadataChunk], pairs = all[7*kMetadataChunk];
        auto minLength = all[9*kMetadataChunk];
        // Bound once for the whole stream, not from the final chunk: a
        // shorter last chunk must retain every prior accumulator lane.
        uint32_t span = 8;
        while (span < batch && span < kMetadataChunk) span *= 2;
        ArithProgression<int32_t>(indices, int32_t(0), int32_t(4), span);
        Duplicate(maxLength, int32_t(0), span);
        Duplicate(minCount, int32_t(0x7fffffff), span);
        Duplicate(maxCount, int32_t(0), span);
        if constexpr (C8_RATIO == 4) Duplicate(pairs, int32_t(0), span);
        Duplicate(minLength, int32_t(0x7fffffff), span);
        for (uint32_t base = 0; base < batch;) {
            const uint32_t count = batch - base < kMetadataChunk ? batch - base : kMetadataChunk;
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0); WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            DataCopyExtParams copy{1, count * uint32_t(sizeof(int32_t)), 0, 0, 0};
            DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
            DataCopyPad(lengths, MetadataGm<int32_t>(p.seqLens)[base], copy, pad);
            copy.blockLen = (count + 1) * uint32_t(sizeof(int32_t));
            DataCopyPad(starts, MetadataGm<int32_t>(p.cuQ)[base], copy, pad);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Max(maxLength, maxLength, lengths, count);
            Min(minLength, minLength, lengths, count); PipeBarrier<PIPE_V>();
            Gather(ends, starts, indices.ReinterpretCast<uint32_t>(), uint32_t(4), count);
            PipeBarrier<PIPE_V>();
            Sub(ends, ends, starts, count); PipeBarrier<PIPE_V>();
            Min(minCount, minCount, ends, count); Max(maxCount, maxCount, ends, count);
            // Only C4 consumes paired counts. C128 calls this summary to
            // validate q1; on failure its general planner recounts from cuQ.
            // Keep field3 initialized to0 and preserve the publication ABI.
            if constexpr (C8_RATIO == 4) {
                // ceil(nonnegative INT32/2), without count+1 overflow or
                // a metadata cast through FP32 (which loses integers>2^24).
                ShiftLeft(tmp.ReinterpretCast<uint32_t>(), ends.ReinterpretCast<uint32_t>(), uint32_t(31), count);
                PipeBarrier<PIPE_V>();
                ShiftRight(tmp.ReinterpretCast<uint32_t>(), tmp.ReinterpretCast<uint32_t>(), uint32_t(31), count);
                ShiftRight(ends.ReinterpretCast<uint32_t>(), ends.ReinterpretCast<uint32_t>(), uint32_t(1), count);
                PipeBarrier<PIPE_V>(); Add(tmp, tmp, ends, count); PipeBarrier<PIPE_V>();
                Add(pairs, pairs, tmp, count); PipeBarrier<PIPE_V>();
            } else PipeBarrier<PIPE_V>(); // retire min/max before the next fold
            base += count;
        }
        for (uint32_t half = span / 2; half >= 8; half /= 2) {
            Max(maxLength, maxLength, maxLength[half], half);
            Min(minCount, minCount, minCount[half], half);
            Min(minLength, minLength, minLength[half], half);
            Max(maxCount, maxCount, maxCount[half], half);
            if constexpr (C8_RATIO == 4) Add(pairs, pairs, pairs[half], half);
            PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
        uint32_t minimumLength = 0x7fffffffu;
        for (uint32_t i = 0; i < 8; ++i) {
            const uint32_t length = maxLength.GetValue(i), lo = minCount.GetValue(i), hi = maxCount.GetValue(i);
            s.maximum = length > s.maximum ? length : s.maximum;
            s.minimumQueries = lo < s.minimumQueries ? lo : s.minimumQueries;
            s.maximumQueries = hi > s.maximumQueries ? hi : s.maximumQueries;
            if constexpr (C8_RATIO == 4) s.paired += pairs.GetValue(i);
            const int32_t rawMinimum = minLength.GetValue(i);
            const uint32_t safeMinimum = rawMinimum > 0 ? uint32_t(rawMinimum) : 0;
            minimumLength = safeMinimum < minimumLength ? safeMinimum : minimumLength;
        }
        uint32_t order = 0;
        const uint32_t keys = s.maximum / p.ratio;
        const uint32_t originalRows = ((s.maximum < 128 ? s.maximum : 128) + 31) / 32 * 32;
        const uint32_t compressedRows = (keys + 31) / 32 * 32;
        const bool canPlan = s.minimumQueries == 1 && s.maximumQueries == 1 &&
            keys <= kRetainedKeys && PlanRetainedRows(originalRows, compressedRows).width == 1;
        if (canPlan && RetainedQueryCost(minimumLength, p.ratio) != RetainedQueryCost(s.maximum, p.ratio))
            order = batch <= kMetadataChunk ? BalanceCachedMetadata(lengths, batch, p.ratio)
                                          : BalanceStreamedMetadata(p, batch, all);
        // Each partner owns a distinct32B record. No second publication can
        // overwrite an unread record; inactive plans explicitly publish0.
        if (GetBlockIdx() % 2 == 0 || canPlan) {
            ends.SetValue(0, s.maximum); ends.SetValue(1, s.minimumQueries);
            ends.SetValue(2, s.maximumQueries); ends.SetValue(3, s.paired);
            ends.SetValue(4, order);
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0); WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            DataCopy(MetadataGm<int32_t>(transfer + (GetBlockIdx() % 2) * 64), ends, 8);
        }
        // DAV2201 mode2 requires one notification from each AIV partner.
        CrossCoreSetFlag<2, PIPE_MTE3>(kMetadataReady);
        PipeBarrier<PIPE_ALL>();
    }
    if ASCEND_IS_AIC {
        CrossCoreWaitFlag(kMetadataReady);
        auto record = MetadataGm<uint32_t>(transfer);
        DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(record);
        s = {record.GetValue(0), record.GetValue(1), record.GetValue(2),
             C8_RATIO == 4 ? record.GetValue(3) : 0};
    }
    return s;
}

}
