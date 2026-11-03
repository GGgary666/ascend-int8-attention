// 135000 tokens / C4 / 32 tokens per page needs 1055 entries. Round up the
// local cache to 64 entries (only 256 B more UB than the old 1024 limit).
// Addressing still follows upstream SCFA GetKeyGmOffset/CopyInKv. Cache the
// active prefix, not the serving table's maximum-capacity row: short requests
// must not fall back to scalar GM reads just because max_model_len is large.
constexpr uint32_t C4_PAGE_CACHE_ENTRIES = 1088;

__aicore__ inline uint32_t CachedC4PageCount(const SharedKVArgs &p, const Selection &sel)
{
    // Avoid runtime division for the model's 32-token physical pages. Read
    // whole 32 B blocks when capacity allows; only the final capacity edge
    // needs DataCopyPad's partial-block handling. Extra IDs are never indexed.
    const uint32_t pages = sel.cmpCount ? (p.cmpBlock == 32 ?
        ((sel.cmpCount - 1) >> 5) + 1 : (sel.cmpCount - 1) / p.cmpBlock + 1) : 0;
    if (pages > C4_PAGE_CACHE_ENTRIES || pages > p.cmpTableWidth) return 0;
    const uint32_t aligned = (pages + 7) / 8 * 8;
    return aligned < p.cmpTableWidth ? aligned : p.cmpTableWidth;
}

// Original sparse indices follow upstream sharedkv DataCopyPABySlots: they
// are physical slot IDs. The page table is not consulted for this input.
__aicore__ inline void GatherOriginalSlots(const SharedKVArgs &p, const Selection &sel,
                                           uint32_t start, uint32_t sub,
                                           LocalTensor<int8_t> kv, LocalTensor<int32_t> ix,
                                           LocalTensor<float> mask, LocalTensor<float> tmp,
                                           LocalTensor<uint8_t> bits, GlobalTensor<int8_t> dst)
{
    Duplicate(ix, -1, N);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0); WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    uint32_t valid = p.oriTopk - start < N ? p.oriTopk - start : N;
    DataCopyPad(ix, Gm<int32_t>(p.oriIndices)[static_cast<uint64_t>(sel.token) * p.oriTopk + start],
                DataCopyExtParams{1, valid * 4, 0, 0, 0}, DataCopyPadExtParams<int32_t>{});
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0); WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    for (uint32_t n = valid; n < (valid + 7) / 8 * 8; ++n) ix.SetValue(n, -1);
    SetFlag<HardEvent::S_V>(EVENT_ID0); WaitFlag<HardEvent::S_V>(EVENT_ID0);
    uint32_t capacity = p.oriPages * p.oriBlock;
    auto clipped = tmp.ReinterpretCast<int32_t>();
    Maxs(clipped, ix, 0, N); PipeBarrier<PIPE_V>();
    Mins(clipped, clipped, static_cast<int32_t>(capacity - 1), N); PipeBarrier<PIPE_V>();
    Sub(clipped, ix, clipped, N); PipeBarrier<PIPE_V>();
    // Equality to zero preserves exact INT32 bounds even above 2**24.
    Cast(tmp, clipped, RoundMode::CAST_RINT, N); PipeBarrier<PIPE_V>();
    CompareScalar(bits, tmp, 0.0f, CMPMODE::EQ, N); PipeBarrier<PIPE_V>();
    Duplicate(mask, 0.0f, N); PipeBarrier<PIPE_V>();
    Select(mask, bits, mask, -1e30f, SELMODE::VSEL_TENSOR_SCALAR_MODE, N);
    VectorToScalar();
    auto cache = Gm<int8_t>(p.ori);
    for (uint32_t chunk = 0; chunk < N / 2; chunk += 64) {
        Duplicate(kv.ReinterpretCast<int32_t>(), 0, 64 * D / 4);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0); WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        for (uint32_t n = 0; n < 64;) {
            uint32_t slot = sub * (N / 2) + chunk + n;
            int32_t idx = ix.GetValue(slot);
            if (idx < 0 || static_cast<uint32_t>(idx) >= capacity) { ++n; continue; }
            uint32_t run = 1, limit = p.oriBlock - static_cast<uint32_t>(idx) % p.oriBlock;
            limit = limit < 64 - n ? limit : 64 - n;
            while (run < limit && ix.GetValue(slot + run) == idx + static_cast<int32_t>(run)) ++run;
            uint64_t offset = static_cast<uint64_t>(static_cast<uint32_t>(idx) / p.oriBlock) * p.oriStride +
                              static_cast<uint64_t>(static_cast<uint32_t>(idx) % p.oriBlock) * D;
            DataCopyPad(kv[n * D], cache[offset], DataCopyExtParams{1, run * D, 0, 0, 0}, DataCopyPadExtParams<int8_t>{});
            n += run;
        }
        SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0); WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        DataCopy(dst[(sub * (N / 2) + chunk) * D], kv, 64 * D);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0); WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
    CrossCoreSetFlag<2, PIPE_MTE3>(KV_READY);
}

// AIV sparse gather, following upstream sharedkv SCFA CopyInKv/ProcessVec0L:
// vllm-ascend 86add7adfd1194c225fd8880699d5be7a248c3ce,
// csrc/attention/sparse_attn_sharedkv/op_kernel/arch32/
// sparse_attn_sharedkv_scfa_block_vector.h:585-716.
// Two valid keys can be reordered within a tile: their QK columns and V rows
// move together. Invalid holes stay at the original slot, preserving masking.
// Short-context compaction follows upstream arch22 fused_sparse_attention_overlap
// GatherValidSelectionTopk: validity is tested before compacting, so holes and
// repeated indices are supported. It removes empty work without assuming that
// a negative entry ends the valid prefix.
__aicore__ inline void CompactSelection(const SharedKVArgs &p, Selection &sel,
                                       LocalTensor<int32_t> full, LocalTensor<float> tmp,
                                       uint64_t scratch)
{
    Duplicate(full, -1, 1024);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0); WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    DataCopyPad(full, Gm<int32_t>(p.indices)[static_cast<uint64_t>(sel.token) * p.topk],
                DataCopyExtParams{1, p.topk * 4, 0, 0, 0}, DataCopyPadExtParams<int32_t>{});
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0); WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    for (uint32_t n = p.topk; n < (p.topk + 7) / 8 * 8; ++n) full.SetValue(n, -1);
    SetFlag<HardEvent::S_V>(EVENT_ID0); WaitFlag<HardEvent::S_V>(EVENT_ID0);
    uint32_t aligned = (p.topk + 63) / 64 * 64;
    Cast(tmp, full, RoundMode::CAST_RINT, aligned); PipeBarrier<PIPE_V>();
    auto lower = tmp[1024].ReinterpretCast<uint8_t>();
    auto upper = tmp[1088].ReinterpretCast<uint8_t>();
    CompareScalar(lower, tmp, -1.0f, CMPMODE::GT, aligned); PipeBarrier<PIPE_V>();
    CompareScalar(upper, tmp, static_cast<float>(static_cast<int32_t>(sel.cmpCount)), CMPMODE::LT, aligned); PipeBarrier<PIPE_V>();
    And(lower.ReinterpretCast<uint16_t>(), lower.ReinterpretCast<uint16_t>(),
        upper.ReinterpretCast<uint16_t>(), aligned / 16); PipeBarrier<PIPE_V>();
    Duplicate(full.ReinterpretCast<float>(), -1.0f, 1024); PipeBarrier<PIPE_V>();
    GatherMaskParams gp{};
    gp.repeatTimes = 1; gp.src0BlockStride = 1; gp.src0RepeatStride = 8; gp.src1RepeatStride = 0;
    uint64_t count = 0;
    GatherMask(full.ReinterpretCast<float>(), tmp, lower.ReinterpretCast<uint32_t>(), true, aligned, gp, count);
    PipeBarrier<PIPE_V>();
    Cast(full, full.ReinterpretCast<float>(), RoundMode::CAST_RINT, 1024); PipeBarrier<PIPE_V>();
    sel.sparseCount = static_cast<uint32_t>(count);
    auto stat = tmp[1152].ReinterpretCast<int32_t>();
    Duplicate(stat, static_cast<int32_t>(count), 8);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0); WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    if (GetBlockIdx() % 2 == 0) DataCopy(Gm<int32_t>(scratch + SLOTS_OFFSET), stat, 8);
    CrossCoreSetFlag<2, PIPE_MTE3>(COMPACT_READY);
    CrossCoreWaitFlag(COMPACT_CONSUMED);
}

__aicore__ inline void CopyRowPair(LocalTensor<int8_t> dst, GlobalTensor<int8_t> cache,
                                  int32_t row0, int32_t row1)
{
    const uint64_t off0 = row0 >= 0 ? static_cast<uint64_t>(row0) * D : 0;
    const uint64_t off1 = row1 >= 0 ? static_cast<uint64_t>(row1) * D : 0;
    const uint64_t distance = off0 > off1 ? off0 - off1 : off1 - off0;
    DataCopyExtParams cp{1, D, 0, 0, 0};
    if (row0 >= 0 && row1 >= 0 && distance >= D && distance - D < 0x7fffffffULL) {
        cp.blockCount = 2; cp.srcStride = distance - D;
        DataCopyPad(dst, cache[off0 < off1 ? off0 : off1], cp, DataCopyPadExtParams<int8_t>{});
    } else {
        if (row0 >= 0) DataCopyPad(dst, cache[off0], cp, DataCopyPadExtParams<int8_t>{});
        if (row1 >= 0) DataCopyPad(dst[D], cache[off1], cp, DataCopyPadExtParams<int8_t>{});
    }
}

__aicore__ inline void OrderRows(int32_t &a, int32_t &b)
{
    const int32_t low = a < b ? a : b;
    b = a < b ? b : a; a = low;
}

__aicore__ inline void GatherSparse(const SharedKVArgs &p, const Selection &sel,
                                   uint32_t start, uint32_t sub, bool pagesCached,
                                   LocalTensor<int8_t> kv, LocalTensor<int32_t> ix,
                                   LocalTensor<int32_t> compact,
                                   LocalTensor<float> mask, LocalTensor<float> tmp,
                                   LocalTensor<uint8_t> bits, LocalTensor<int32_t> pages,
                                   GlobalTensor<int8_t> dst)
{
    Duplicate(ix, -1, N);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0); WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    uint32_t valid = sel.sparseCount - start < N ? sel.sparseCount - start : N;
    if (sel.NeedsCompact(p)) {
        DataCopy(ix, compact[start], N);
        VectorToScalar();
    } else {
        DataCopyExtParams ip{1, valid * 4, 0, 0, 0};
        DataCopyPad(ix, Gm<int32_t>(p.indices)[static_cast<uint64_t>(sel.token) * p.topk + start],
                    ip, DataCopyPadExtParams<int32_t>{});
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0); WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    }
    // MTE2 pads the last 32 B block; restore only those padded elements.
    for (uint32_t n = valid; n < (valid + 7) / 8 * 8; ++n) ix.SetValue(n, -1);
    SetFlag<HardEvent::S_V>(EVENT_ID0); WaitFlag<HardEvent::S_V>(EVENT_ID0);
    Cast(tmp, ix, RoundMode::CAST_RINT, N); PipeBarrier<PIPE_V>();
    CompareScalar(bits, tmp, -1.0f, CMPMODE::GT, N); PipeBarrier<PIPE_V>();
    CompareScalar(bits[64], tmp, static_cast<float>(static_cast<int32_t>(sel.cmpCount)), CMPMODE::LT, N); PipeBarrier<PIPE_V>();
    And(bits.ReinterpretCast<uint16_t>(), bits.ReinterpretCast<uint16_t>(),
        bits[64].ReinterpretCast<uint16_t>(), N / 16); PipeBarrier<PIPE_V>();
    Duplicate(mask, 0.0f, N); PipeBarrier<PIPE_V>();
    Select(mask, bits, mask, -1e30f, SELMODE::VSEL_TENSOR_SCALAR_MODE, N);
    auto table = Gm<int32_t>(p.cmpTable);
    auto cache = Gm<int8_t>(p.cmp);
    // Page lookup is a UB Gather, following upstream arch22 fused sparse
    // attention's GatherValidSelectionTopk/membership-byte-offset pattern.
    // Keep row addresses in integer arithmetic; float conversion would lose
    // page/slot precision for larger caches.
    bool vectorOffsets = pagesCached && p.cmpStride % D == 0 &&
                         (p.cmpBlock & (p.cmpBlock - 1)) == 0;
    auto rows = bits[256].ReinterpretCast<int32_t>();
    if (vectorOffsets) {
        auto clipped = tmp.ReinterpretCast<int32_t>();
        auto offsets = bits[256 + N * 4].ReinterpretCast<int32_t>();
        int32_t shift = 0;
        for (uint32_t block = p.cmpBlock; block > 1; block >>= 1) ++shift;
        Maxs(clipped, ix, 0, N); PipeBarrier<PIPE_V>();
        Mins(clipped, clipped, static_cast<int32_t>(sel.cmpCount ? sel.cmpCount - 1 : 0), N); PipeBarrier<PIPE_V>();
        ShiftRight(rows, clipped, shift, N); PipeBarrier<PIPE_V>();
        ShiftLeft(offsets, rows, shift, N); PipeBarrier<PIPE_V>();
        Sub(clipped, clipped, offsets, N); PipeBarrier<PIPE_V>();
        ShiftLeft(rows, rows, 2, N); PipeBarrier<PIPE_V>();
        Gather(offsets, pages, rows.ReinterpretCast<uint32_t>(), 0, N); PipeBarrier<PIPE_V>();
        Muls(offsets, offsets, static_cast<int32_t>(p.cmpStride / D), N); PipeBarrier<PIPE_V>();
        Add(rows, offsets, clipped, N); PipeBarrier<PIPE_V>();
        // dav_c220 Select supports float/half only. vsel is a bitwise choice:
        // reinterpret preserves all 32 address bits, and -1.0f supplies a
        // negative INT32 sentinel for invalid slots, with no FP arithmetic.
        Select(rows.ReinterpretCast<float>(), bits, rows.ReinterpretCast<float>(),
               -1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, N);
        VectorToScalar();
    }
    // Retain the 64-row DMA batch while alternating two buffers. The earlier
    // 32-row experiment increased launch overhead and was rejected. This
    // fits the wide plan's 184 KiB user UB after aliasing the dead raw-score
    // input buffer with x. A compile-time byte budget enforces that limit.
    constexpr uint32_t buffers = N == 512 ? 2 : 1;
    for (uint32_t b = 0; b < buffers; ++b) SetFlag<HardEvent::MTE3_V>(b);
    for (uint32_t chunk = 0; chunk < N / 2; chunk += 64) {
        uint32_t buffer = (chunk / 64) % buffers;
        auto current = kv[buffer * 64 * D];
        WaitFlag<HardEvent::MTE3_V>(buffer);
        Duplicate(current.ReinterpretCast<int32_t>(), 0, 64 * D / 4);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0); WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        if (vectorOffsets) {
            for (uint32_t n = 0; n < 64; n += 4) {
                const uint32_t slot = sub * (N / 2) + chunk + n;
                int32_t a = rows.GetValue(slot), b = rows.GetValue(slot + 1);
                int32_t c = rows.GetValue(slot + 2), d = rows.GetValue(slot + 3);
                // Upstream SCFA retains a signed 2 GiB DMA-stride limit.
                // Re-pair only four all-valid keys when a pair crosses it;
                // QK columns and V rows move together within this softmax tile.
                // Masked holes and duplicate keys keep their multiplicity.
                if (a >= 0 && b >= 0 && c >= 0 && d >= 0 &&
                    ((a > b ? a - b : b - a) > 4194304 ||
                     (c > d ? c - d : d - c) > 4194304)) {
                    OrderRows(a, b); OrderRows(c, d);
                    OrderRows(a, c); OrderRows(b, d); OrderRows(b, c);
                }
                CopyRowPair(current[n * D], cache, a, b);
                CopyRowPair(current[(n + 2) * D], cache, c, d);
            }
        } else for (uint32_t n = 0; n < 64; n += 2) {
            uint32_t slot = sub * (N / 2) + chunk + n;
            int32_t idx0 = vectorOffsets ? rows.GetValue(slot) : ix.GetValue(slot);
            int32_t idx1 = vectorOffsets ? rows.GetValue(slot + 1) : ix.GetValue(slot + 1);
            bool ok0 = idx0 >= 0 && (vectorOffsets || static_cast<uint32_t>(idx0) < sel.cmpCount);
            bool ok1 = idx1 >= 0 && (vectorOffsets || static_cast<uint32_t>(idx1) < sel.cmpCount);
            uint64_t off0 = 0, off1 = 0;
            if (vectorOffsets) {
                if (ok0) off0 = static_cast<uint64_t>(idx0) * D;
                if (ok1) off1 = static_cast<uint64_t>(idx1) * D;
            } else {
            if (ok0) off0 = static_cast<uint64_t>(pagesCached ? pages.GetValue(idx0 / p.cmpBlock)
                                : table.GetValue(sel.batch * p.cmpTableWidth + idx0 / p.cmpBlock)) * p.cmpStride
                            + static_cast<uint64_t>(idx0 % p.cmpBlock) * D;
            if (ok1) off1 = static_cast<uint64_t>(pagesCached ? pages.GetValue(idx1 / p.cmpBlock)
                                : table.GetValue(sel.batch * p.cmpTableWidth + idx1 / p.cmpBlock)) * p.cmpStride
                            + static_cast<uint64_t>(idx1 % p.cmpBlock) * D;
            }
            uint64_t distance = off0 > off1 ? off0 - off1 : off1 - off0;
            DataCopyExtParams cp{1, D, 0, 0, 0};
            if (ok0 && ok1 && distance >= D && distance - D < 0x7fffffffULL) {
                cp.blockCount = 2; cp.srcStride = distance - D;
                DataCopyPad(current[n * D], cache[off0 < off1 ? off0 : off1], cp, DataCopyPadExtParams<int8_t>{});
            } else {
                if (ok0) DataCopyPad(current[n * D], cache[off0], cp, DataCopyPadExtParams<int8_t>{});
                if (ok1) DataCopyPad(current[(n + 1) * D], cache[off1], cp, DataCopyPadExtParams<int8_t>{});
            }
        }
        SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0); WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        DataCopy(dst[(sub * (N / 2) + chunk) * D], current, 64 * D);
        SetFlag<HardEvent::MTE3_V>(buffer);
    }
    for (uint32_t b = 0; b < buffers; ++b) WaitFlag<HardEvent::MTE3_V>(b);
    CrossCoreSetFlag<2, PIPE_MTE3>(KV_READY);
    SetFlag<HardEvent::S_V>(EVENT_ID0); WaitFlag<HardEvent::S_V>(EVENT_ID0);
}
