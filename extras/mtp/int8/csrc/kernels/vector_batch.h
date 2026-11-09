// Private implementation, included in sharedkv.cpp's kernel namespace.
// Reference: vllm-ascend 86add7adfd1194c225fd8880699d5be7a248c3ce,
// sparse_attn_sharedkv_swa_block_vector.h:331 (SoftmaxFlashV2Compute),
// :687 (Brcb), :769 (row-wise scaling via vector repeat strides).
constexpr SoftmaxConfig SHARED_SOFTMAX_CFG{false, 0, 0, SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC};

#undef SHARED_VEC_N
#undef SHARED_SCALE_KEYS
#ifdef SHARED_VECTOR_TAILS
#ifdef SHARED_DUAL_VECTOR_TAILS
static_assert(N <= 256 && SHARED_QUERY_GROUP == 4 && SHARED_COMP_ROW_SCALE && SHARED_PACKED_C4);
#else
static_assert(N == 512 && SHARED_QUERY_GROUP == 1 && SHARED_CUBE_TAILS && SHARED_ROW_SCALE);
#endif
#define SHARED_VEC_N vecN
#define SHARED_SCALE_KEYS(x,f,r) ScaleKeyColumnsWidth(x,f,r,vecN)
#else
#define SHARED_VEC_N N
#define SHARED_SCALE_KEYS(x,f,r) ScaleKeyColumns(x,f,r)
#endif

__aicore__ inline void ScaleRows(LocalTensor<float> dst, LocalTensor<float> src,
                               LocalTensor<float> brc, uint32_t rows, uint32_t width)
{
    BinaryRepeatParams rp{};
    rp.dstBlkStride = rp.src0BlkStride = 1; rp.src1BlkStride = 0;
    if (width / 64 <= rows) {
        rp.dstRepStride = rp.src0RepStride = width / 8; rp.src1RepStride = 1;
        for (uint32_t col = 0; col < width; col += 64)
            Mul(dst[col], src[col], brc, 64, rows, rp);
    } else {
        rp.dstRepStride = rp.src0RepStride = 8; rp.src1RepStride = 0;
        for (uint32_t row = 0; row < rows; ++row)
            Mul(dst[row * width], src[row * width], brc[row * 8], 64, width / 64, rp);
    }
}

__aicore__ inline void ScaleChannels(LocalTensor<float> dst, LocalTensor<float> src,
                                   LocalTensor<float> scale, uint32_t rows)
{
    if (rows >= 8) {
        BinaryRepeatParams rp{};
        rp.dstBlkStride = rp.src0BlkStride = rp.src1BlkStride = 1;
        rp.dstRepStride = rp.src0RepStride = D / 8; rp.src1RepStride = 0;
        for (uint32_t col = 0; col < D; col += 64)
            Mul(dst[col], src[col], scale[col], 64, rows, rp);
    } else {
        for (uint32_t row = 0; row < rows; ++row) Mul(dst[row * D], src[row * D], scale, D);
    }
}

__aicore__ inline void AccumulateChannels(LocalTensor<float> dst, LocalTensor<float> src,
                                         LocalTensor<float> scale, uint32_t rows)
{
    // CANN 9.2 dav_c220 explicitly supports FP32 vmla/MulAddDst. Some
    // upstream uses are arch35 MicroAPI; this uses the verified c220 API.
    if (rows >= 8) {
        BinaryRepeatParams rp{};
        rp.dstBlkStride = rp.src0BlkStride = rp.src1BlkStride = 1;
        rp.dstRepStride = rp.src0RepStride = D / 8; rp.src1RepStride = 0;
        for (uint32_t col = 0; col < D; col += 64)
            MulAddDst(dst[col], src[col], scale[col], 64, rows, rp);
    } else {
        for (uint32_t row = 0; row < rows; ++row) MulAddDst(dst[row * D], src[row * D], scale, D);
    }
}

// The upstream arch22 sparse-attention kernel builds selection membership
// maps and compacts valid indices with GatherMask. Here the short-context
// candidate set fits one 256-key tile, shared by four adjacent queries.
// Multiplicity is retained as log(count): no assumption of unique, ordered
// indices or of a negative entry terminating the selection is needed.
__aicore__ inline void BuildC4Membership(const SharedKVArgs &p, const Selection &sel,
                                        uint32_t begin, uint32_t nr, LocalTensor<float> weights,
                                        LocalTensor<float> source, LocalTensor<float> tmp)
{
    uint32_t firstQ = begin / SHARED_HEADS(p);
    uint32_t endQ = (begin + nr + SHARED_HEADS(p) - 1) / SHARED_HEADS(p);
    for (uint32_t q = firstQ; q < endQ; ++q) {
        uint32_t limit = (sel.position + q + 1) / 4;
        auto counts = weights[(q - firstQ) * N].ReinterpretCast<int32_t>();
        Duplicate(counts, 0, N);
        Duplicate(source.ReinterpretCast<int32_t>(), -1, 1024);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0); WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        DataCopyPad(source.ReinterpretCast<int32_t>(), Gm<int32_t>(p.indices)[static_cast<uint64_t>(sel.token + q) * p.topk],
                    DataCopyExtParams{1, p.topk * 4, 0, 0, 0}, DataCopyPadExtParams<int32_t>{});
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0); WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        for (uint32_t n = p.topk; n < (p.topk + 7) / 8 * 8; ++n)
            source.ReinterpretCast<int32_t>().SetValue(n, -1);
        SetFlag<HardEvent::S_V>(EVENT_ID0); WaitFlag<HardEvent::S_V>(EVENT_ID0);
        uint32_t aligned = (p.topk + 63) / 64 * 64;
        Cast(tmp, source.ReinterpretCast<int32_t>(), RoundMode::CAST_RINT, aligned); PipeBarrier<PIPE_V>();
        auto lower = tmp[1024].ReinterpretCast<uint8_t>();
        auto upper = tmp[1088].ReinterpretCast<uint8_t>();
        CompareScalar(lower, tmp, -1.0f, CMPMODE::GT, aligned); PipeBarrier<PIPE_V>();
        CompareScalar(upper, tmp, static_cast<float>(static_cast<int32_t>(limit)), CMPMODE::LT, aligned); PipeBarrier<PIPE_V>();
        And(lower.ReinterpretCast<uint16_t>(), lower.ReinterpretCast<uint16_t>(),
            upper.ReinterpretCast<uint16_t>(), aligned / 16); PipeBarrier<PIPE_V>();
        GatherMaskParams gp{};
        gp.repeatTimes = 1; gp.src0BlockStride = 1; gp.src0RepeatStride = 8; gp.src1RepeatStride = 0;
        uint64_t count = 0;
        GatherMask(source, tmp, lower.ReinterpretCast<uint32_t>(), true, aligned, gp, count);
        PipeBarrier<PIPE_V>();
        Cast(source.ReinterpretCast<int32_t>(), source, RoundMode::CAST_RINT, aligned);
        VectorToScalar();
        if (count == limit) {
            // Common short-context selection contains every causal key once.
            // Confirm that property exactly: equal valid count alone cannot
            // exclude duplicates. Mark presence, then count unique entries.
            for (uint32_t n = 0; n < count; ++n)
                counts.SetValue(source.ReinterpretCast<int32_t>().GetValue(n), 1);
            SetFlag<HardEvent::S_V>(EVENT_ID0); WaitFlag<HardEvent::S_V>(EVENT_ID0);
            Cast(tmp, counts, RoundMode::CAST_RINT, N); PipeBarrier<PIPE_V>();
            ReduceSum(tmp[1152], tmp, tmp[1216], N);
            VectorToScalar();
            if (tmp.GetValue(1152) == static_cast<float>(static_cast<int32_t>(limit))) {
                // All selected keys have multiplicity one. This gives the
                // same zero/log-count mask without integer UB read-modify-write
                // for each key or a vector Ln. Scatter is unsupported on c220.
                CompareScalar(lower, tmp, 0.0f, CMPMODE::GT, N); PipeBarrier<PIPE_V>();
                auto w = counts.ReinterpretCast<float>();
                Duplicate(w, 0.0f, N); PipeBarrier<PIPE_V>();
                Select(w, lower, w, -1e30f, SELMODE::VSEL_TENSOR_SCALAR_MODE, N);
                PipeBarrier<PIPE_V>();
                continue;
            }
            // Equal count with duplicate/hole: rebuild true multiplicities.
            Duplicate(counts, 0, N);
            VectorToScalar();
        }
        for (uint32_t n = 0; n < count; ++n) {
            uint32_t idx = source.ReinterpretCast<int32_t>().GetValue(n);
            counts.SetValue(idx, counts.GetValue(idx) + 1);
        }
        SetFlag<HardEvent::S_V>(EVENT_ID0); WaitFlag<HardEvent::S_V>(EVENT_ID0);
        auto w = counts.ReinterpretCast<float>();
        Cast(w, counts, RoundMode::CAST_RINT, N); PipeBarrier<PIPE_V>();
        CompareScalar(lower, w, 0.0f, CMPMODE::GT, N); PipeBarrier<PIPE_V>();
        Maxs(tmp, w, 1.0f, N); PipeBarrier<PIPE_V>();
        Ln(w, tmp, N); PipeBarrier<PIPE_V>();
        Select(w, lower, w, -1e30f, SELMODE::VSEL_TENSOR_SCALAR_MODE, N); PipeBarrier<PIPE_V>();
    }
}

#if SHARED_ROW_SCALE
#include "row_scale_vector.h"
#endif

__aicore__ void VectorBatched(const SharedKVArgs &p, uint32_t splits SHARED_LIVE_PARAM SHARED_TASK_PARAM)
{
    constexpr uint32_t gatherRows = N == 512 ? 128 : 64;
    constexpr uint32_t ubBytes = R * D * 16 + D * 4 * KV_L1_SLOTS + 128 + 84 * R + 32768 + N * 8
        + (SHARED_QUERY_GROUP > 1 ? 64 : 0)
        + (SHARED_PACKED_C4 || SHARED_SMALL_C4 ? N * 8 : 0)
        + (gatherRows * D + C4_PAGE_CACHE_ENTRIES * 4 + 4096)
        + (SHARED_ROW_SCALE && SHARED_QUERY_GROUP > 1 ? (2 * N + 32) * 4 : 0)
        + (SHARED_ROW_SCALE ? N * 4 : 0)
#ifdef SHARED_COMP_FACTOR_PREFETCH
        + 2 * N * 4
#endif
        ;
    // CANN 9.2 kernel_utils_constants.h: TOTAL_VEC_LOCAL_SIZE / TMP_UB_OFFSET
    // are 184 KiB on __NPU_ARCH__=2201; the top 8 KiB of 192 KiB is reserved.
    static_assert(ubBytes <= 184 * 1024, "Vector buffers exceed dav_c220 user UB (184 KiB)");
    TPipe pipe;
#ifdef SHARED_COMP_FACTOR_PREFETCH
    static_assert(SHARED_QUERY_GROUP > 1 && SHARED_COMP_ROW_SCALE);
    TBuf<TPosition::VECCALC> cmpFactorsB, cmpPvFactorsB;
    pipe.InitBuffer(cmpFactorsB,N * 4); pipe.InitBuffer(cmpPvFactorsB,N * 4);
    auto cmpFactors=cmpFactorsB.Get<float>(), cmpPvFactors=cmpPvFactorsB.Get<float>();
#endif
#if SHARED_ROW_SCALE
    TBuf<TPosition::VECCALC> factorB, factorPagesB, factorOffsetsB;
    pipe.InitBuffer(factorOffsetsB, N * 4);
    if constexpr (SHARED_QUERY_GROUP > 1) {
        pipe.InitBuffer(factorB, N * 4); pipe.InitBuffer(factorPagesB, (N + 32) * 4);
    }
#endif
    TBuf<TPosition::VECCALC> qb, scaleB, xb, absB, ib, rb, accB, maskB, sinksB;
    TBuf<TPosition::VECCALC> maxB, sumB, expB, softTmpB, brcB, gatherB, indexB, pagesB, compactB, membershipB;
    pipe.InitBuffer(qb, R * D * 2); pipe.InitBuffer(scaleB, D * 4 * KV_L1_SLOTS);
    pipe.InitBuffer(xb, R * D * 4); pipe.InitBuffer(absB, R * D * 4);
    pipe.InitBuffer(ib, R * D * 2); pipe.InitBuffer(rb, (32 + R * 8) * 4);
    pipe.InitBuffer(accB, R * D * 4);
    pipe.InitBuffer(maskB, N * 4);
    pipe.InitBuffer(maxB, 2 * R * 4); pipe.InitBuffer(sumB, 2 * R * 4); pipe.InitBuffer(expB, R * 4);
    pipe.InitBuffer(softTmpB, 32768); pipe.InitBuffer(brcB, R * 8 * 4);
    pipe.InitBuffer(indexB, N * 4);
    if constexpr (SHARED_PACKED_C4 || SHARED_SMALL_C4) pipe.InitBuffer(membershipB, 2 * N * 4);
    if constexpr (true) {
        pipe.InitBuffer(gatherB, gatherRows * D); pipe.InitBuffer(pagesB, C4_PAGE_CACHE_ENTRIES * 4);
        pipe.InitBuffer(compactB, 1024 * 4);
    }
    auto indices = indexB.Get<int32_t>();
    auto q = qb.Get<bfloat16_t>(); auto scale = scaleB.Get<float>();
    auto x = xb.Get<float>(); auto absX = absB.Get<float>();
    // Q is dead after Cast; reuse its bytes for the intermediate half and
    // final BF16 output. Score is dead before the PV read into the same UB.
    auto h = qb.Get<half>(); auto i8 = ib.Get<int8_t>(); auto r = rb.Get<float>();
    // Raw score/PV input can reuse x: its old contents
    // have already been consumed. The in-place INT32->FP32 cast is supported
    // on dav_c220 (also used for residual products). Sharing this buffer
    // saves 16 KiB for ungrouped queries and 32 KiB for four-query groups.
    auto sc = xb.Get<int32_t>();
    auto pv = sc;
    auto acc = accB.Get<float>(); auto out = qb.Get<bfloat16_t>(); auto mask = maskB.Get<float>();
    auto maxBuf = maxB.Get<float>(); auto sumBuf = sumB.Get<float>(); auto expBuf = expB.Get<float>();
    auto softTmp = softTmpB.Get<uint8_t>(); auto brc = brcB.Get<float>();
#if SHARED_ROW_SCALE
    LocalTensor<float> factors, factorPages;
    auto factorOffsets = factorOffsetsB.Get<int32_t>();
    if (SHARED_QUERY_GROUP == 1 && !p.oriIndices && p.oriBlock == 32) {
        // A dedicated immutable offset vector avoids both repeated index
        // generation and aliasing the residual-score DMA destination.
        CreateVecIndex(factorOffsets, int32_t{0}, N); PipeBarrier<PIPE_V>();
        ShiftLeft(factorOffsets, factorOffsets, 2, N); PipeBarrier<PIPE_V>();
    }
    if constexpr (SHARED_QUERY_GROUP == 1) {
        // Original dense KV goes directly to Cube L1. For explicit slots,
        // GatherOriginalSlots drains its UB->GM stores before returning.
        // Factors are consumed before P_READY and compressed-KV prefetch,
        // so this staging UB is dead throughout both factor uses.
        static_assert((2 * N + 32) * 4 <= gatherRows * D);
        factors = gatherB.Get<float>(); factorPages = factors[N];
    } else {
        factors = factorB.Get<float>(); factorPages = factorPagesB.Get<float>();
    }
#endif
    if constexpr (SHARED_QUERY_GROUP > 1) {
        // SCFA InitSinks keeps/broadcasts sinks in UB. All grouped entries
        // require H8; cache the normal and head-four rotation once. A q3
        // tail gives 12 rows per vector partner, and a one-query tail gives
        // four. Both used to reread GM scalarly for the second partner.
        pipe.InitBuffer(sinksB, 64);
        auto cachedSinks = sinksB.Get<float>();
        DataCopy(cachedSinks, Gm<float>(p.sinks), 8);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0); WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        for (uint32_t head = 0; head < 8; ++head)
            cachedSinks.SetValue(8 + head, cachedSinks.GetValue((head + 4) % 8));
        SetFlag<HardEvent::S_V>(EVENT_ID0); WaitFlag<HardEvent::S_V>(EVENT_ID0);
        CreateVecIndex(indices.ReinterpretCast<float>(), 0.0f, N);
        PipeBarrier<PIPE_V>();
#if SHARED_ROW_SCALE
        // Grouped causal masks already need 0..N-1. Reuse that sequence
        // instead of executing CreateVecIndex's scalar/vector expansion twice.
        Cast(factorOffsets, indices.ReinterpretCast<float>(), RoundMode::CAST_RINT, N);
        PipeBarrier<PIPE_V>();
        ShiftLeft(factorOffsets, factorOffsets, 2, N); PipeBarrier<PIPE_V>();
#endif
    }
    const uint32_t core = GetBlockIdx() / 2, sub = GetBlockIdx() % 2;
    const bool residual = SHARED_HEADS(p) <= 8;
    uint64_t scratch = p.workspace + static_cast<uint64_t>(core) * WORKSPACE_BYTES;
    auto qg = Gm<int8_t>(scratch + Q_OFFSET), pg = Gm<int8_t>(scratch + P_OFFSET);
    auto sg = Gm<int32_t>(scratch + SCORE_OFFSET), vg = Gm<int32_t>(scratch + PV_OFFSET);
    auto query = Gm<bfloat16_t>(p.q); auto output = Gm<bfloat16_t>(p.out);
#if (defined(SHARED_TAIL_BALANCE) && SHARED_TAIL_BALANCE == 2) || defined(SHARED_SHORT_TASK_MAP)
    const uint32_t tasks = taskPlan.total;
#else
    const uint32_t tasks = TaskCount(p, splits SHARED_LIVE_ARG);
#endif
    for (uint32_t task = core; task < tasks; task += GetBlockNum()) {
        Selection sel(p, task, splits SHARED_TASK_ARG);
#ifdef SHARED_SHORT_TASK_MAP
        if (!sel.rows) continue;
#endif
        const bool pairPipeline = PipelineTwoKinds(p, sel, splits);
        if (SHARED_QUERY_GROUP == 1 && sel.NeedsCompact(p))
            CompactSelection(p, sel, compactB.Get<int32_t>(), absX, scratch);
        const uint32_t perVec = (sel.rows + 1) / 2, begin = sub * perVec;
        const uint32_t nr = begin < sel.rows ? (sel.rows - begin < perVec ? sel.rows - begin : perVec) : 0;
        const uint64_t rowBase = static_cast<uint64_t>(sel.token) * SHARED_HEADS(p) + sel.head + begin;
#ifdef SHARED_COMP_FACTOR_PREFETCH
        if (p.cmpFactors && nr && sel.cmpCount)
            LoadCompressedFactors(p,sel,0,cmpFactors,cmpPvFactors);
#endif
        if constexpr (SHARED_PACKED_C4 || SHARED_SMALL_C4)
            BuildC4Membership(p, sel, begin, nr, membershipB.Get<float>(), x, absX);
        uint32_t state = 0;
        bool hasAccumulated = false;
        uint32_t stagedStart = 0xffffffffU;
        uint32_t cachedPages = 0;
#ifdef SHARED_CACHE_QUERY
        if (!pairPipeline) Duplicate(acc, 0.0f, R * D);
#else
        Duplicate(acc, 0.0f, R * D);
#endif
        Duplicate(sumBuf, sel.part == 0 ? 1.0f : 0.0f, 2 * R);
        PipeBarrier<PIPE_ALL>();
        if constexpr (SHARED_QUERY_GROUP > 1) {
            for (uint32_t row = 0; row < nr; row += 8)
                DataCopy(maxBuf[row], sinksB.Get<float>()[(begin & 4) ? 8 : 0], 8);
            // Rounded UB copies stay inside maxBuf[0:R]; nr controls softmax.
            PipeBarrier<PIPE_V>();
        } else {
          for (uint32_t row = 0; row < nr;) {
            uint32_t head = (sel.head + begin + row) % SHARED_HEADS(p);
            uint32_t run = SHARED_HEADS(p) - head < nr - row ? SHARED_HEADS(p) - head : nr - row;
            // Every UB destination in this branch is 32 B aligned.
            DataCopyPad(maxBuf[row], Gm<float>(p.sinks)[head],
                        DataCopyExtParams{1, run * 4, 0, 0, 0}, DataCopyPadExtParams<float>{});
            row += run;
          }
          SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        }
        SoftMaxShapeInfo shape{nr, 128, nr, 128};
        // nr=0 is the unused vector partner of a one-head tail; it still
        // participates in cross-core flags, without invoking a zero-row API.
        SoftMaxTiling smTiling = SoftMaxFlashV2TilingFunc(SoftMaxShapeInfo{nr ? nr : 1, 128, nr ? nr : 1, 128},
                                                        4, 4, 32768, true, false);
        if (SHARED_QUERY_GROUP == 1 && !SHARED_SMALL_C4 && SHARED_RATIO(p) == 4 && sel.Start(p, 1) < sel.Count(p, 1)) {
            cachedPages = CachedC4PageCount(p, sel);
            if (cachedPages) {
                DataCopyExtParams cp{1, cachedPages * 4, 0, 0, 0};
                DataCopyPad(pagesB.Get<int32_t>(), Gm<int32_t>(p.cmpTable)[sel.batch * p.cmpTableWidth], cp,
                            DataCopyPadExtParams<int32_t>{});
                SetFlag<HardEvent::MTE2_S>(EVENT_ID0); WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            }
        }
        for (uint32_t kind = 0; kind < (SHARED_RATIO(p) == 1 ? 1u : 2u); ++kind) {
            const uint32_t count = sel.Count(p, kind);
            if (sel.Start(p, kind) >= count) continue;
#ifdef SHARED_VECTOR_TAILS
            // Upstream SCFA uses aligned actual columns for its vector work.
            // Keep the N512 GM row pitch and Cube math unchanged. On dav_c220,
            // DataCopyParams uses32-byte lengths/gaps in both directions.
#ifdef SHARED_DUAL_VECTOR_TAILS
            // Plans8/9 compact to live64/128 columns; plans10/11 retain full
            // width in the same compiled body. Cube zero-fills all absent KV
            // columns, so untouched P padding never contributes to PV.
            const uint32_t vecN = p.plan < 10 ? (count <= 64 ? 64 : count <= 128 ? 128 : N) : N;
#else
            const uint32_t vecN = p.plan == 28 && kind == 0 && !p.oriIndices && count <= 128 ? 128 : N;
#endif
            shape = SoftMaxShapeInfo{nr, vecN, nr, vecN};
            smTiling = SoftMaxFlashV2TilingFunc(SoftMaxShapeInfo{nr ? nr : 1, vecN, nr ? nr : 1, vecN},
                                              4, 4, 32768, true, false);
#endif
            if (!pairPipeline || kind == 0) {
              for (uint32_t qkind = kind; qkind < (pairPipeline ? 2u : kind + 1); ++qkind) {
                auto scale = scaleB.Get<float>()[pairPipeline ? qkind * D : 0];
                auto qdst = Gm<int8_t>(scratch + (pairPipeline && qkind ? Q_SECOND_OFFSET : Q_OFFSET));
                if (nr) {
                    DataCopy(scale, Gm<float>(qkind == 0 ? p.oriScale : p.cmpScale), D);
#ifdef SHARED_CACHE_QUERY
                    if (!pairPipeline || qkind == 0) DataCopy(q, query[rowBase * D], nr * D);
                    SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                    // acc is dead before PV0. Keep unscaled FP32 Q there for
                    // the second cache kind, avoiding its GM read and cast.
                    if (pairPipeline) {
                        if (qkind == 0) { Cast(acc,q,RoundMode::CAST_NONE,nr * D); PipeBarrier<PIPE_V>(); }
                        ScaleChannels(x,acc,scale,nr);
                    } else {
                        Cast(x,q,RoundMode::CAST_NONE,nr * D); PipeBarrier<PIPE_V>();
                        ScaleChannels(x,x,scale,nr);
                    }
#else
                    DataCopy(q, query[rowBase * D], nr * D);
                    SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                    Cast(x, q, RoundMode::CAST_NONE, nr * D); PipeBarrier<PIPE_V>();
                    ScaleChannels(x, x, scale, nr);
#endif
                    PipeBarrier<PIPE_V>();
                    Abs(absX, x, nr * D); PipeBarrier<PIPE_V>();
                    ReduceRepeat<ReduceType::MAX>(r[32], absX, 64, nr * 8, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
                    PipeBarrier<PIPE_V>();
                    ReduceRepeat<ReduceType::MAX>(r, r[32], 8, nr, 1, 1, 1, ReduceOrder::ORDER_ONLY_VALUE);
                    PipeBarrier<PIPE_V>();
                    Muls(r, r, 1.0f / 127.0f, nr); PipeBarrier<PIPE_V>();
                    Maxs(r, r, 1e-12f, nr); PipeBarrier<PIPE_V>();
                    if (SHARED_QUERY_GROUP > 1 || SHARED_HEADS(p) == 8) {
                        // Experimental H8 path: dav_c220 vdiv has at most
                        // one-ULP reciprocal error in the dedicated probe. Q/P
                        // residuals remain enabled; attention is checked against
                        // the unchanged scalar-reciprocal oracle and real data.
                        Duplicate(r[R], 1.0f, R); PipeBarrier<PIPE_V>();
                        Div(r[R], r[R], r, nr); PipeBarrier<PIPE_V>();
                    } else {
                        VectorToScalar();
                        for (uint32_t row = 0; row < nr; ++row)
                            r.SetValue(R + row, 1.0f / r.GetValue(row));
                        SetFlag<HardEvent::S_V>(EVENT_ID0); WaitFlag<HardEvent::S_V>(EVENT_ID0);
                    }
                    Brcb(brc, r[R], R / 8, {1, 8}); PipeBarrier<PIPE_V>();
                    ScaleRows(x, x, brc, nr, D);
                    PipeBarrier<PIPE_V>();
                    Mins(x, x, 127.0f, nr * D); Maxs(x, x, -127.0f, nr * D);
                    if (residual) RoundResidualInt8(i8, i8[R * D], h, x, absX, nr * D);
                    else RoundInt8(i8, h, x, nr * D);
                    SetFlag<HardEvent::V_MTE3>(EVENT_ID0); WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                    DataCopy(qdst[begin * D], i8, nr * D);
                    if (residual) DataCopy(qdst[(begin + RESIDUAL_ROW_OFFSET) * D], i8[R * D], nr * D);
                    PipeBarrier<PIPE_ALL>();
                }
                CrossCoreSetFlag<2, PIPE_MTE3>(pairPipeline && qkind ? Q_READY_SECOND : Q_READY);
                // Q folding used the original scale. PV always divides by 127;
                // precompute that per-channel factor once for this cache kind.
                if (nr) { Muls(scale, scale, 1.0f / 127.0f, D); PipeBarrier<PIPE_V>(); }
                if (pairPipeline && nr) {
                    // maxBuf[R] is dead until softmax0. Keep Q0 alpha there while
                    // the second Q reuses reduce scratch, then keep Q1 at r[32].
                    DataCopy(qkind == 0 ? maxBuf[R] : r[32], r, (nr + 7) / 8 * 8);
                    PipeBarrier<PIPE_V>();
                }
              }
#ifdef SHARED_CACHE_QUERY
              if (pairPipeline && nr) { Duplicate(acc,0.0f,R * D); PipeBarrier<PIPE_V>(); }
#endif
            }
            auto scale = scaleB.Get<float>()[pairPipeline ? kind * D : 0];
            if (pairPipeline && nr) {
                DataCopy(r, kind == 0 ? maxBuf[R] : r[32], (nr + 7) / 8 * 8);
                PipeBarrier<PIPE_V>();
            }
            for (uint32_t start = sel.Start(p, kind); start < count; start += N * splits) {
                if (kind == 0 && p.oriIndices)
                    GatherOriginalSlots(p, sel, start, sub, gatherB.Get<int8_t>(), indices, mask, x,
                                        absX.ReinterpretCast<uint8_t>(), Gm<int8_t>(scratch + KV_OFFSET));
                if (SHARED_QUERY_GROUP == 1 && !SHARED_SMALL_C4 && kind == 1 && SHARED_RATIO(p) == 4 && start != stagedStart) {
                    GatherSparse(p, sel, start, sub, cachedPages != 0, gatherB.Get<int8_t>(), indices, compactB.Get<int32_t>(), mask, x,
                                 absX.ReinterpretCast<uint8_t>(), pagesB.Get<int32_t>(), Gm<int8_t>(scratch + KV_OFFSET));
                    stagedStart = start;
                }
#if SHARED_ROW_SCALE
                if (kind == 0 && nr)
                    LoadOriginalFactors(p,sel,start,factors,factorPages,factorOffsets.ReinterpretCast<uint32_t>());
#ifdef SHARED_COMP_ROW_SCALE
#ifdef SHARED_COMP_FACTOR_PREFETCH
                if (kind == 1 && nr && p.cmpFactors) {
                    WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
                    Muls(cmpPvFactors,cmpFactors,127.0f,N); PipeBarrier<PIPE_V>();
                }
#else
                if (kind == 1 && nr && p.cmpFactors)
                    LoadCompressedFactors(p,sel,start,factors,factorPages);
#endif
#endif
#endif
                CrossCoreWaitFlag(pairPipeline && kind ? SCORE_READY_SECOND : SCORE_READY);
                auto score = Gm<int32_t>(scratch + (pairPipeline && kind ? SCORE_SECOND_OFFSET : SCORE_OFFSET));
                if (nr) {
                    if (SHARED_QUERY_GROUP == 1 && ((kind == 0 && (SHARED_SMALL_C4 || !p.oriIndices)) || (kind == 1 && SHARED_RATIO(p) == 128))) {
                        uint32_t valid = count - start < N ? count - start : N;
                        Duplicate(mask, 0.0f, N);
                        if (valid < N) {
                            // Only the first partial 32 B block requires
                            // scalar writes. Fill the aligned remainder with
                            // one vector operation, including wide-tile padding.
                            uint32_t aligned = (valid + 7) / 8 * 8;
                            if (aligned != valid) {
                                VectorToScalar();
                                for (uint32_t n = valid; n < aligned; ++n) mask.SetValue(n, -1e30f);
                                SetFlag<HardEvent::S_V>(EVENT_ID0); WaitFlag<HardEvent::S_V>(EVENT_ID0);
                            }
                            if (aligned < N) {
                                PipeBarrier<PIPE_V>();
                                Duplicate(mask[aligned], -1e30f, N - aligned);
                            }
                        }
                    }
#if SHARED_SMALL_C4
                    if (SHARED_QUERY_GROUP == 1 && kind == 1) {
                        DataCopy(mask,membershipB.Get<float>(),N); PipeBarrier<PIPE_V>();
                    }
#endif
#ifdef SHARED_VECTOR_TAILS
                    const DataCopyParams scoreCopy{static_cast<uint16_t>(nr), static_cast<uint16_t>(vecN / 8),
                                                   static_cast<uint16_t>((N - vecN) / 8), 0};
                    DataCopy(sc, score[begin * N], scoreCopy);
                    if (residual) DataCopy(absX.ReinterpretCast<int32_t>(), score[(begin + RESIDUAL_ROW_OFFSET) * N], scoreCopy);
#else
                    DataCopy(sc, score[begin * N], nr * N);
                    if (residual) DataCopy(absX.ReinterpretCast<int32_t>(), score[(begin + RESIDUAL_ROW_OFFSET) * N], nr * N);
#endif
                    SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                    Cast(x, sc, RoundMode::CAST_RINT, nr * SHARED_VEC_N); PipeBarrier<PIPE_V>();
                    if (residual) AddResidualProduct(x, absX.ReinterpretCast<int32_t>(), nr * SHARED_VEC_N);
                    Muls(r[R], r, p.softmaxScale, nr); PipeBarrier<PIPE_V>();
                    Brcb(brc, r[R], R / 8, {1, 8}); PipeBarrier<PIPE_V>();
                    ScaleRows(x, x, brc, nr, SHARED_VEC_N);
                    PipeBarrier<PIPE_V>();
#if SHARED_ROW_SCALE
#ifdef SHARED_COMP_ROW_SCALE
#ifdef SHARED_COMP_FACTOR_PREFETCH
                    if (kind == 0) SHARED_SCALE_KEYS(x,factors,nr);
                    else if (p.cmpFactors) SHARED_SCALE_KEYS(x,cmpFactors,nr);
#else
                    if (kind == 0 || p.cmpFactors) SHARED_SCALE_KEYS(x,factors,nr);
#endif
#else
                    if (kind == 0) SHARED_SCALE_KEYS(x,factors,nr);
#endif
#endif
                    for (uint32_t row = 0; row < nr; ++row) {
                        if constexpr (SHARED_QUERY_GROUP > 1) {
                            if (!p.oriIndices && (row == 0 || (begin + row) % SHARED_HEADS(p) == 0)) {
                                if (SHARED_PACKED_C4 && kind == 1) {
                                    uint32_t localQ = (begin + row) / SHARED_HEADS(p) - begin / SHARED_HEADS(p);
                                    DataCopy(mask, membershipB.Get<float>()[localQ * N], N);
                                    PipeBarrier<PIPE_V>();
                                } else {
                                int32_t pos = sel.position + (begin + row) / SHARED_HEADS(p);
                                int32_t low = kind == 0 ? pos + 1 - static_cast<int32_t>(p.window) -
                                    static_cast<int32_t>(sel.firstOri + start) : 0;
                                int32_t high = kind == 0 ? pos + 1 - static_cast<int32_t>(sel.firstOri + start)
                                    : (pos + 1) / static_cast<int32_t>(SHARED_RATIO(p)) - static_cast<int32_t>(start);
                                auto bits = absX.ReinterpretCast<uint8_t>();
                                CompareScalar(bits, indices.ReinterpretCast<float>(), static_cast<float>(low), CMPMODE::GE, N);
                                PipeBarrier<PIPE_V>();
                                CompareScalar(bits[64], indices.ReinterpretCast<float>(), static_cast<float>(high), CMPMODE::LT, N);
                                PipeBarrier<PIPE_V>();
                                And(bits.ReinterpretCast<uint16_t>(), bits.ReinterpretCast<uint16_t>(),
                                    bits[64].ReinterpretCast<uint16_t>(), N / 16); PipeBarrier<PIPE_V>();
                                Duplicate(mask, 0.0f, N); PipeBarrier<PIPE_V>();
                                Select(mask, bits, mask, -1e30f, SELMODE::VSEL_TENSOR_SCALAR_MODE, N); PipeBarrier<PIPE_V>();
                                }
                            }
                        }
                        Add(x[row * SHARED_VEC_N], x[row * SHARED_VEC_N], mask, SHARED_VEC_N);
                    }
                    PipeBarrier<PIPE_V>();
                    // Gather/QK use N256; preserve both original N128
                    // online-softmax, probability rounding and PV accumulation steps.
                    const DataCopyParams splitScore{static_cast<uint16_t>(nr),16,16,0};
                    DataCopy(absX,x,splitScore);
                    DataCopy(gatherB.Get<float>(),x[128],splitScore);
                    PipeBarrier<PIPE_V>();
                }
                for (uint32_t pvPart = 0; pvPart < 2; ++pvPart) {
                  if (nr) {
                    DataCopy(x,pvPart == 0 ? absX : gatherB.Get<float>(),nr * 128);
                    PipeBarrier<PIPE_V>();
                    const uint32_t next = state ^ R;
                    SoftmaxFlashV2<float, true, true, false, false, SHARED_SOFTMAX_CFG>(
                        x, sumBuf[next], maxBuf[next], x, expBuf, sumBuf[state], maxBuf[state],
                        softTmp, smTiling, shape);
                    state = next;
                    PipeBarrier<PIPE_V>();
                    if (hasAccumulated) {
                        Brcb(brc, expBuf, R / 8, {1, 8}); PipeBarrier<PIPE_V>();
                        ScaleRows(acc, acc, brc, nr, D);
                    }
#if SHARED_ROW_SCALE
                    // Softmax sum uses unscaled probability. factorPages now
                    // holds 127*a, combining the PV factor and P quantization
                    // scale in one broadcast over the probability matrix.
#ifdef SHARED_COMP_ROW_SCALE
#ifdef SHARED_COMP_FACTOR_PREFETCH
                    if (kind == 0) SHARED_SCALE_KEYS(x,factorPages,nr);
                    else if (p.cmpFactors) SHARED_SCALE_KEYS(x,cmpPvFactors,nr);
#else
                    if (kind == 0 || p.cmpFactors) SHARED_SCALE_KEYS(x,factorPages,nr);
#endif
#else
                    if (kind == 0) SHARED_SCALE_KEYS(x,factorPages,nr);
#endif
                    else { Muls(x, x, 127.0f, nr * 128); PipeBarrier<PIPE_V>(); }
#else
                    Muls(x, x, 127.0f, nr * 128); PipeBarrier<PIPE_V>();
#endif
                    if (residual) RoundProbabilityPacked(i8,h,x,absX,nr * 128);
                    else RoundInt8(i8, h, x, nr * 128);
                    SetFlag<HardEvent::V_MTE3>(EVENT_ID0); WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                    const DataCopyParams pc{static_cast<uint16_t>(nr),4,0,4};
                    DataCopy(pg[begin * N + pvPart * 128],i8,pc);
                    if (residual) DataCopy(pg[(begin + RESIDUAL_ROW_OFFSET) * N + pvPart * 128],i8[nr * 128],pc);
                }
                // V->MTE3 above protects P source; the PIPE_MTE3 flag
                // publishes both strided stores without a duplicate ALL drain.
                CrossCoreSetFlag<2, PIPE_MTE3>(P_READY);
                // Once SCORE_READY was observed, the current KV is retained
                // in L1. Its GM staging tile can hold the next selection while
                // Cube computes PV. Mirrors upstream SCFA's staggered V0/C2
                // schedule, without reusing live L1 or score/P buffers.
                if (SHARED_QUERY_GROUP == 1 && !SHARED_SMALL_C4 && SHARED_RATIO(p) == 4 && (kind == 1 || start + N * splits >= count)) {
                    uint32_t next = kind == 1 ? start + N * splits : sel.Start(p, 1);
                    if (next < sel.Count(p, 1) && next != stagedStart) {
                        GatherSparse(p, sel, next, sub, cachedPages != 0, gatherB.Get<int8_t>(), indices, compactB.Get<int32_t>(), mask, x,
                                     absX.ReinterpretCast<uint8_t>(), pagesB.Get<int32_t>(), Gm<int8_t>(scratch + KV_OFFSET));
                        stagedStart = next;
                    }
                }
                CrossCoreWaitFlag(PV_READY);
                if (nr) {
                    DataCopy(pv, vg[begin * D], nr * D);
                    if (residual) DataCopy(absX.ReinterpretCast<int32_t>(), vg[(begin + RESIDUAL_ROW_OFFSET) * D], nr * D);
                    SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                    Cast(x, pv, RoundMode::CAST_RINT, nr * D); PipeBarrier<PIPE_V>();
                    if (residual) AddResidualProduct(x, absX.ReinterpretCast<int32_t>(), nr * D);
                    AccumulateChannels(acc, x, scale, nr);
                    PipeBarrier<PIPE_ALL>();
                    hasAccumulated = true;
                }
                }
            }
        }
        if (nr) {
            VectorToScalar();
            if (splits > 1) {
                for (uint32_t row = 0; row < nr; ++row) {
                    auto part = Gm<float>(p.partial)[((rowBase + row) * splits + sel.part) * (D + 8)];
                    SetFlag<HardEvent::V_MTE3>(EVENT_ID0); WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                    DataCopy(part, acc[row * D], D);
                    r.SetValue(0, maxBuf.GetValue(state + row)); r.SetValue(1, sumBuf.GetValue(state + row));
                    SetFlag<HardEvent::S_MTE3>(EVENT_ID0); WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
                    DataCopy(part[D], r, 8);
                    PipeBarrier<PIPE_ALL>();
                }
            } else {
                Duplicate(r[R], 1.0f, R); PipeBarrier<PIPE_V>();
                Div(r[R], r[R], sumBuf[state], nr); PipeBarrier<PIPE_V>();
                Brcb(brc, r[R], R / 8, {1, 8}); PipeBarrier<PIPE_V>();
                ScaleRows(x, acc, brc, nr, D);
                PipeBarrier<PIPE_V>();
                Cast(out, x, RoundMode::CAST_RINT, nr * D);
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0); WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                DataCopy(output[rowBase * D], out, nr * D);
                if (p.lse) {
                    // Same online-softmax LSE formula as upstream ProcessLse.
                    Ln(r, sumBuf[state], nr); PipeBarrier<PIPE_V>();
                    Add(r, r, maxBuf[state], nr);
                    SetFlag<HardEvent::V_MTE3>(EVENT_ID0); WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                    DataCopyPad(Gm<float>(p.lse)[rowBase], r, DataCopyExtParams{1, nr * 4, 0, 0, 0});
                }
                PipeBarrier<PIPE_ALL>();
            }
        }
    }
}
