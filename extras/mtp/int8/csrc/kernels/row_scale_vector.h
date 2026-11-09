// Private helper, included only in the optional row-scale implementations.
#ifdef SHARED_COMP_ROW_SCALE
__aicore__ inline void LoadCompressedFactors(const SharedKVArgs &p, const Selection &sel,
                                             uint32_t start, LocalTensor<float> factor,
                                             LocalTensor<float> pvFactor)
{
    // Dense short-C4 candidates are in logical index order. A future sparse
    // implementation must reorder factors together with the gathered KV.
    static_assert(SHARED_PACKED_C4 || SHARED_SMALL_C4);
    const uint32_t count = sel.cmpCount - start < N ? sel.cmpCount - start : N;
    Duplicate(factor, 1.0f, N);
#ifdef SHARED_COMP_FACTOR_PREFETCH
    SetFlag<HardEvent::V_MTE2>(EVENT_ID1); WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
#else
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0); WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
#endif
    auto table = Gm<int32_t>(p.cmpTable);
    auto source = Gm<float>(p.cmpFactors);
    // The explicit entry requires block32 and tile-aligned start. Copy only
    // initialized slots; padding in a partially written page may be NaN.
    for (uint32_t n = 0; n < count; n += 32) {
        const uint32_t page = table.GetValue(sel.batch * p.cmpTableWidth + (start + n) / 32);
        const uint32_t run = count - n < 32 ? count - n : 32;
        DataCopyPad(factor[n],source[static_cast<uint64_t>(page) * p.cmpFactorStride],
                    DataCopyExtParams{1,run * 4,0,0,0},DataCopyPadExtParams<float>{});
    }
#ifdef SHARED_COMP_FACTOR_PREFETCH
    // A separate UB pair and event1 keep this data live across original KV
    // and membership processing. The vector consumer waits before first use.
    SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
#else
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    Muls(pvFactor,factor,127.0f,N); PipeBarrier<PIPE_V>();
#endif
}
#endif

__aicore__ inline void LoadOriginalFactors(const SharedKVArgs &p, const Selection &sel,
                                           uint32_t start, LocalTensor<float> factor,
                                           LocalTensor<float> pages, LocalTensor<uint32_t> offsets)
{
    const uint32_t count = sel.oriCount - start < N ? sel.oriCount - start : N;
    Duplicate(factor, 1.0f, N);
    if (SHARED_SMALL_C4 || (!p.oriIndices && p.oriBlock == 32)) {
        // Copy complete factor pages into aligned UB, then gather the shifted
        // logical window. This avoids unaligned UB DMA destinations and does
        // not assume consecutive physical pages. Padding factors stay finite.
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0); WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        const uint32_t first = sel.firstOri + start, offset = first % 32;
        const uint32_t pageCount = (offset + count + 31) / 32;
        auto table = Gm<int32_t>(p.oriTable);
        auto source = Gm<float>(p.oriFactors);
        for (uint32_t page = 0; page < pageCount; ++page) {
            const uint32_t physical = table.GetValue(sel.batch * p.oriTableWidth + first / 32 + page);
            DataCopyPad(pages[page * 32],source[static_cast<uint64_t>(physical) * p.oriFactorStride],
                        DataCopyExtParams{1,128,0,0,0},DataCopyPadExtParams<float>{});
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        // Read only valid slots: uninitialized factors in a partial physical
        // page may be NaN. The initial Duplicate preserves finite padding.
        Gather(factor,pages,offsets,offset * sizeof(float),count);
        PipeBarrier<PIPE_V>();
    } else {
        // Explicit physical slots (DSpark) and non-power-of-two page sizes.
        // Use exact integer addressing; invalid slots retain a finite factor.
        VectorToScalar();
        auto table = Gm<int32_t>(p.oriTable);
        auto source = Gm<float>(p.oriFactors);
        for (uint32_t n = 0; n < count; ++n) {
            int64_t page, slot;
            if (p.oriIndices) {
                const int32_t index = Gm<int32_t>(p.oriIndices).GetValue(
                    static_cast<uint64_t>(sel.token) * p.oriTopk + start + n);
                if (index < 0 || static_cast<uint32_t>(index) >= p.oriPages * p.oriBlock) continue;
                page = index / p.oriBlock; slot = index % p.oriBlock;
            } else {
                const uint32_t index = sel.firstOri + start + n;
                page = table.GetValue(sel.batch * p.oriTableWidth + index / p.oriBlock);
                slot = index % p.oriBlock;
            }
            factor.SetValue(n,source.GetValue(static_cast<uint64_t>(page) * p.oriFactorStride + slot));
        }
        SetFlag<HardEvent::S_V>(EVENT_ID0); WaitFlag<HardEvent::S_V>(EVENT_ID0);
    }
    // The page DMA/Gather has finished. Reuse its storage for the PV scale;
    // QK continues to use the original unmodified factor vector.
    Muls(pages,factor,127.0f,N); PipeBarrier<PIPE_V>();
    // Offsets are immutable and disjoint from the residual-score DMA buffer.
    // G1's factor/page storage remains live until both factor multiplications
    // finish; P's PIPE_ALL completes those reads before sparse gather reuses it.
}

__aicore__ inline void ScaleKeyColumns(LocalTensor<float> x, LocalTensor<float> factor, uint32_t rows)
{
    if (N / 64 <= rows) {
        BinaryRepeatParams rp{};
        rp.dstBlkStride = rp.src0BlkStride = rp.src1BlkStride = 1;
        rp.dstRepStride = rp.src0RepStride = N / 8; rp.src1RepStride = 0;
        for (uint32_t col = 0; col < N; col += 64) Mul(x[col],x[col],factor[col],64,rows,rp);
    } else {
        for (uint32_t row = 0; row < rows; ++row) Mul(x[row * N],x[row * N],factor,N);
    }
    PipeBarrier<PIPE_V>();
}

#ifdef SHARED_VECTOR_TAILS
__aicore__ inline void ScaleKeyColumnsWidth(LocalTensor<float> x, LocalTensor<float> factor,
                                           uint32_t rows, uint32_t width)
{
    if (width / 64 <= rows) {
        BinaryRepeatParams rp{};
        rp.dstBlkStride = rp.src0BlkStride = rp.src1BlkStride = 1;
        rp.dstRepStride = rp.src0RepStride = width / 8; rp.src1RepStride = 0;
        for (uint32_t col = 0; col < width; col += 64) Mul(x[col],x[col],factor[col],64,rows,rp);
    } else {
        for (uint32_t row = 0; row < rows; ++row) Mul(x[row * width],x[row * width],factor,width);
    }
    PipeBarrier<PIPE_V>();
}
#endif
