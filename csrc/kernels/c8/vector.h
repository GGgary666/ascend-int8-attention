// C8 production core: frozen-channel KV, Q/P double residuals.
// Private implementation, included in sharedkv.cpp's kernel namespace.
// Reference: vllm-ascend 86add7adfd1194c225fd8880699d5be7a248c3ce,
// sparse_attn_sharedkv_swa_block_vector.h:331 (SoftmaxFlashV2Compute),
// :687 (Brcb), :769 (row-wise scaling via vector repeat strides).
constexpr SoftmaxConfig SHARED_SOFTMAX_CFG{false, 0, 0, SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC};

#undef SHARED_SCALE_KEYS
#define SHARED_SCALE_KEYS(x,f,r) ScaleKeyColumns(x,f,r)

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
// candidate set fits one tile, shared by adjacent query/head rows.
// Multiplicity is retained as log(count): no assumption of unique, ordered
// indices or of a negative entry terminating the selection is needed.
template<uint32_t MembershipCapacity = 512, uint32_t ScratchRows = R>
__aicore__ inline void BuildC4Membership(const SharedKVArgs &p, const Selection &sel,
                                        uint32_t begin, uint32_t nr, LocalTensor<float> weights,
                                        LocalTensor<float> source, LocalTensor<float> tmp, uint32_t columns = N)
{
    uint32_t firstQ = begin / SHARED_HEADS(p);
    uint32_t endQ = (begin + nr + SHARED_HEADS(p) - 1) / SHARED_HEADS(p);
    for (uint32_t q = firstQ; q < endQ; ++q) {
        uint32_t limit = (sel.position + q + 1) / 4;
        auto counts = weights[(q - firstQ) * columns].ReinterpretCast<int32_t>();
        // An empty causal key domain has zero membership regardless of the
        // supplied selection list. No index DMA, filtering or sorting is due.
        if (!limit) {
            Duplicate(counts.ReinterpretCast<float>(), -1e30f, columns);
            PipeBarrier<PIPE_V>();
            continue;
        }
        Duplicate(counts, 0, columns);
        // C8's fixed 1024-entry ABI overwrites this entire input region.
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0); WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        DataCopyPad(source.ReinterpretCast<int32_t>(), Gm<int32_t>(p.indices)[static_cast<uint64_t>(sel.token + q) * C4_TOPK],
                    DataCopyExtParams{1, C4_TOPK * 4, 0, 0, 0}, DataCopyPadExtParams<int32_t>{});
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0); WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        for (uint32_t n = C4_TOPK; n < (C4_TOPK + 7) / 8 * 8; ++n)
            source.ReinterpretCast<int32_t>().SetValue(n, -1);
        SetFlag<HardEvent::S_V>(EVENT_ID0); WaitFlag<HardEvent::S_V>(EVENT_ID0);
        uint32_t aligned = (C4_TOPK + 63) / 64 * 64;
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
        // Empty/singleton selections need neither a permutation check nor
        // a histogram/logarithm. This is selection cardinality, independent
        // of the key-domain size, query capacity or index position/order.
        if (count <= 1) {
            auto w = counts.ReinterpretCast<float>();
            Duplicate(w, -1e30f, columns);
            PipeBarrier<PIPE_V>();
            if (count) {
                const uint32_t index = source.ReinterpretCast<int32_t>().GetValue(0);
                VectorToScalar();
                w.SetValue(index, 0.0f); // log(multiplicity=1)
                SetFlag<HardEvent::S_V>(EVENT_ID0); WaitFlag<HardEvent::S_V>(EVENT_ID0);
            }
            continue;
        }
        if (count == limit && limit) {
            // Verify a complete permutation exactly, in Vector. Equal count
            // alone cannot exclude duplicate/hole pairs. Like upstream arch22
            // SortSelectionTopk, use c220 Sort32/merge via the legacy Sort API.
            // All values and differences are small exact FP32 integers.
            // These disjoint slices precede Q encoding; neither x nor its
            // residual scratch contains live attention data yet.
            const uint32_t k = (limit + 31) / 32 * 32;
            static_assert(1024 + 3 * MembershipCapacity <= ScratchRows * D,
                          "Sort values and2K scratch must fit Vector storage");
            static_assert(1536 + (MembershipCapacity > 512 ? 3 : 4) * MembershipCapacity <= ScratchRows * D,
                          "Sort pairs and labels must fit Vector storage");
            auto values = source[1024], sortScratch = source[1024 + MembershipCapacity];
            auto sorted = source[1024 + 2 * MembershipCapacity];
            auto pairs = tmp[1536];
            auto labels = tmp[1536 + 2 * MembershipCapacity].ReinterpretCast<uint32_t>();
            // Sort consumes input labels before the PIPE_V below. Extract
            // may then reuse those bytes as its output; pairs stay disjoint.
            auto sortedLabels = tmp[MembershipCapacity > 512 ?
                1536 + 2 * MembershipCapacity : 1536 + 3 * MembershipCapacity].ReinterpretCast<uint32_t>();
            Duplicate(values, -1.0f, k); Duplicate(labels, 0u, k); PipeBarrier<PIPE_V>();
            Cast(values, source.ReinterpretCast<int32_t>(), RoundMode::CAST_RINT, limit);
            PipeBarrier<PIPE_V>();
            Sort<float, true>(pairs, values, labels, sortScratch, k / 32); PipeBarrier<PIPE_V>();
            Extract(sorted, sortedLabels, pairs, k / 32); PipeBarrier<PIPE_V>();
            CreateVecIndex(values, 0.0f, k); PipeBarrier<PIPE_V>();
            Muls(values, values, -1.0f, k); PipeBarrier<PIPE_V>();
            Adds(values, values, static_cast<float>(static_cast<int32_t>(limit - 1)), k);
            PipeBarrier<PIPE_V>(); Maxs(values, values, -1.0f, k); PipeBarrier<PIPE_V>();
            Sub(values, sorted, values, k); PipeBarrier<PIPE_V>();
            Abs(values, values, k); PipeBarrier<PIPE_V>();
            ReduceSum(tmp[1152], values, tmp[1216], k);
            VectorToScalar();
            if (tmp.GetValue(1152) == 0.0f) {
                CreateVecIndex(tmp, 0.0f, columns); PipeBarrier<PIPE_V>();
                CompareScalar(lower, tmp, static_cast<float>(static_cast<int32_t>(limit)), CMPMODE::LT, columns);
                PipeBarrier<PIPE_V>();
                auto w = counts.ReinterpretCast<float>();
                Duplicate(w, 0.0f, columns); PipeBarrier<PIPE_V>();
                Select(w, lower, w, -1e30f, SELMODE::VSEL_TENSOR_SCALAR_MODE, columns);
                PipeBarrier<PIPE_V>();
                continue;
            }
        }
        for (uint32_t n = 0; n < count; ++n) {
            uint32_t idx = source.ReinterpretCast<int32_t>().GetValue(n);
            counts.SetValue(idx, counts.GetValue(idx) + 1);
        }
        SetFlag<HardEvent::S_V>(EVENT_ID0); WaitFlag<HardEvent::S_V>(EVENT_ID0);
        auto w = counts.ReinterpretCast<float>();
        Cast(w, counts, RoundMode::CAST_RINT, columns); PipeBarrier<PIPE_V>();
        CompareScalar(lower, w, 0.0f, CMPMODE::GT, columns); PipeBarrier<PIPE_V>();
        Maxs(tmp, w, 1.0f, columns); PipeBarrier<PIPE_V>();
        Ln(w, tmp, columns); PipeBarrier<PIPE_V>();
        Select(w, lower, w, -1e30f, SELMODE::VSEL_TENSOR_SCALAR_MODE, columns); PipeBarrier<PIPE_V>();
    }
}


// Normalize and encode independent rows. G1 has eight reserved UB rows but
// only four live heads per AIV, so both cache-specific Q encodings fit together.
template<uint32_t RowStride = R>
__aicore__ inline void EncodeQueryRows(LocalTensor<float> x, LocalTensor<float> absX,
    LocalTensor<float> r, LocalTensor<float> brc, LocalTensor<half> h,
    LocalTensor<int8_t> i8, uint32_t nr, bool residual)
{
    constexpr uint32_t scalarStride = (RowStride + 7) / 8 * 8;
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
        Duplicate(r[scalarStride], 1.0f, scalarStride); PipeBarrier<PIPE_V>();
        Div(r[scalarStride], r[scalarStride], r, nr); PipeBarrier<PIPE_V>();
    } else {
        VectorToScalar();
        for (uint32_t row = 0; row < nr; ++row)
            r.SetValue(scalarStride + row, 1.0f / r.GetValue(row));
        SetFlag<HardEvent::S_V>(EVENT_ID0); WaitFlag<HardEvent::S_V>(EVENT_ID0);
    }
    Brcb(brc, r[scalarStride], (nr + 7) / 8, {1, 8}); PipeBarrier<PIPE_V>();
    ScaleRows(x, x, brc, nr, D);
    PipeBarrier<PIPE_V>();
    // Absmax normalization already bounds every finite row by127. A few
    // FP32 ulps of endpoint overshoot still encode high=+/-127, low=0:
    // the low residual's rounding interval is1/(2*127), much wider.
    if (residual) RoundResidualInt8(i8, i8[RowStride * D], h, x, absX, nr * D);
    else RoundInt8(i8, h, x, nr * D);
}

// Only byte offsets of already-owned TPipe buffers cross the helper call.
// CANN9.2 LocalTensor(VECCALC,addr,size) recreates a view, not an allocation;
// no layout offsets are duplicated here. Keep lookahead out of the hot loop.
struct QueryStageBuffers {
    uint32_t q, x, residual, scales, i8, reduce, broadcast, alpha, floatQ;
};

__aicore__ __attribute__((noinline, section(".text.sharedkv_schedule")))
bool StageGroupedQuery(const SharedKVArgs &p, uint32_t nextTask, uint32_t tasks,
    uint32_t taskOrder, uint32_t uniformQueries, uint64_t scratch, const QueryStageBuffers &buf)
{
    const uint32_t nextIndex = TaskIndex(nextTask, tasks, taskOrder);
    if (nextIndex >= tasks) return false;
    Selection future(p, nextIndex, 1, uniformQueries);
    if (!PipelineTwoKinds(p, future, 1)) return false;
    const uint32_t halfRows = (future.rows + 1) / 2;
    const uint32_t first = (GetBlockIdx() % 2) * halfRows;
    const uint32_t live = first < future.rows ?
        (future.rows - first < halfRows ? future.rows - first : halfRows) : 0;
    const uint64_t futureRow = static_cast<uint64_t>(future.token) * SHARED_HEADS(p) + future.head + first;
    LocalTensor<bfloat16_t> q(TPosition::VECCALC, buf.q, R * D);
    LocalTensor<float> x(TPosition::VECCALC, buf.x, R * D);
    LocalTensor<float> absX(TPosition::VECCALC, buf.residual, R * D);
    LocalTensor<float> scales(TPosition::VECCALC, buf.scales, 4 * D);
    LocalTensor<int8_t> i8(TPosition::VECCALC, buf.i8, 2 * R * D);
    LocalTensor<float> r(TPosition::VECCALC, buf.reduce, 32 + R * 8);
    LocalTensor<float> brc(TPosition::VECCALC, buf.broadcast, R * 8);
    LocalTensor<float> alpha(TPosition::VECCALC, buf.alpha, 2 * R);
    // Current softmax has retired before lookahead; on initial staging it
    // has not started. Keep one exact FP32 Q in that dead scratch through
    // both channel-scale encodings. Q's BF16 bytes then become half scratch.
    static_assert(R * D * sizeof(float) <= 32768);
    LocalTensor<float> floatQ(TPosition::VECCALC, buf.floatQ, R * D);
    if (live) {
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0); WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        DataCopy(q, Gm<bfloat16_t>(p.q)[futureRow * D], live * D);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(floatQ, q, RoundMode::CAST_NONE, live * D); PipeBarrier<PIPE_V>();
    }
    for (uint32_t kind = 0; kind < 2; ++kind) {
        if (live) {
            ScaleChannels(x, floatQ, scales[kind * D], live); PipeBarrier<PIPE_V>();
            EncodeQueryRows(x, absX, r, brc, q.ReinterpretCast<half>(), i8, live, true);
            DataCopy(alpha[kind * R], r, (live + 7) / 8 * 8); PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0); WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            auto dst = Gm<int8_t>(scratch + (kind ? Q_SECOND_OFFSET : Q_OFFSET));
            DataCopy(dst[first * D], i8, live * D);
            DataCopy(dst[(first + RESIDUAL_ROW_OFFSET) * D], i8[R * D], live * D);
            PipeBarrier<PIPE_ALL>();
        }
        CrossCoreSetFlag<2, PIPE_MTE3>(kind ? Q_READY_SECOND : Q_READY);
    }
    return true;
}

#include "joint_tile.h"

// Skew the two FP32 operands by half a 512-byte bank-group period on c220.
// This is a physical-layout choice, independent of request/task geometry.
constexpr uint32_t VECTOR_OPERAND_SKEW = 256;
// One TPipe allocation reserves the complete arena; views need no individual
// dynamic buffer handles. The supplied ND softmax scratch remains explicit.
template<uint32_t Bytes, uint32_t Offset = 0>
struct VectorUbRegion {
    static constexpr uint32_t end = Offset + Bytes;
    static_assert(Offset % 32 == 0 && Bytes % 32 == 0 && end <= 184 * 1024);
    template<class T> __aicore__ inline LocalTensor<T> Get() const
    { return LocalTensor<T>(TPosition::VECCALC, Offset, Bytes / sizeof(T)); }
};

// Dense phases benefit from inlining like upstream SCFA. Sparse gather has a
// larger instruction footprint; retain its separate Vector phase at an
// instruction-fetch block boundary (2 KiB on DAV2201).
#if C8_RATIO == 128
__aicore__ __attribute__((always_inline)) inline void VectorBatched(const SharedKVArgs &p, uint32_t splits, uint32_t taskOrder, uint32_t uniformQueries)
#else
__aicore__ __attribute__((aligned(2048))) void VectorBatched(const SharedKVArgs &p, uint32_t splits, uint32_t taskOrder, uint32_t uniformQueries)
#endif
{
    constexpr uint32_t gatherRows = N == 512 ? 128 : 64;
    constexpr uint32_t ubBytes = R * D * 16 + D * 4 * 4 + 128 + 84 * R + 32768 + N * 8
        + 64 + VECTOR_OPERAND_SKEW
        + (SHARED_PACKED_C4 ? N * 8 : 0)
        + (SHARED_QUERY_GROUP == 1 ? gatherRows * D + C4_PAGE_CACHE_ENTRIES * 4 + 4096 : 0)
        + (SHARED_QUERY_GROUP > 1 ? 2 * R * 4 : 0)
        ;
    // CANN 9.2 kernel_utils_constants.h: TOTAL_VEC_LOCAL_SIZE / TMP_UB_OFFSET
    // are 184 KiB on __NPU_ARCH__=2201; the top 8 KiB of 192 KiB is reserved.
    static_assert(ubBytes <= 184 * 1024, "Vector buffers exceed dav_c220 user UB (184 KiB)");
    TPipe pipe;
    VectorUbRegion<R * D * 2> qb;
    VectorUbRegion<D * 4 * 4, decltype(qb)::end> scaleB;
    VectorUbRegion<R * D * 4, decltype(scaleB)::end> xb;
    VectorUbRegion<R * D * 4, decltype(xb)::end + VECTOR_OPERAND_SKEW> absB;
    VectorUbRegion<R * D * 2, decltype(absB)::end> ib;
    VectorUbRegion<(32 + R * 8) * 4, decltype(ib)::end> rb;
    VectorUbRegion<R * D * 4, decltype(rb)::end> accB;
    VectorUbRegion<N * 4, decltype(accB)::end> maskB;
    VectorUbRegion<2 * R * 4, decltype(maskB)::end> maxB;
    VectorUbRegion<2 * R * 4, decltype(maxB)::end> sumB;
    VectorUbRegion<R * 4, decltype(sumB)::end> expB;
    VectorUbRegion<32768, decltype(expB)::end> softTmpB;
    VectorUbRegion<R * 8 * 4, decltype(softTmpB)::end> brcB;
    VectorUbRegion<N * 4, decltype(brcB)::end> indexB;
    VectorUbRegion<(SHARED_QUERY_GROUP > 1 ? 2 * R * 4 : 0), decltype(indexB)::end> nextQScaleB;
    VectorUbRegion<SHARED_PACKED_C4 ? 2 * N * 4 : 0, decltype(nextQScaleB)::end> membershipB;
    VectorUbRegion<SHARED_QUERY_GROUP == 1 ? gatherRows * D : 0, decltype(membershipB)::end> gatherB;
    VectorUbRegion<SHARED_QUERY_GROUP == 1 ? C4_PAGE_CACHE_ENTRIES * 4 : 0, decltype(gatherB)::end> pagesB;
    VectorUbRegion<SHARED_QUERY_GROUP == 1 ? 1024 * 4 : 0, decltype(pagesB)::end> compactB;
    VectorUbRegion<64, decltype(compactB)::end> sinksB;
    static_assert(decltype(sinksB)::end == ubBytes, "Fixed views must match the UB arena");
    TBuf<TPosition::VECCALC> arena;
    pipe.InitBuffer(arena, ubBytes);
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
    static_assert((R * 64 + R * 8) * sizeof(float) <= 32768);
    const SoftmaxBuffers softmaxBuffers{
        static_cast<uint32_t>(x.GetPhyAddr()), static_cast<uint32_t>(sumBuf.GetPhyAddr()),
        static_cast<uint32_t>(maxBuf.GetPhyAddr()), static_cast<uint32_t>(expBuf.GetPhyAddr()),
        static_cast<uint32_t>(softTmp.GetPhyAddr())};
    LocalTensor<float> nextQScale;
    if constexpr (SHARED_QUERY_GROUP > 1) nextQScale = nextQScaleB.Get<float>();
    QueryStageBuffers stageBuffers{};
    if constexpr (SHARED_QUERY_GROUP > 1) {
        stageBuffers = {static_cast<uint32_t>(q.GetPhyAddr()), static_cast<uint32_t>(x.GetPhyAddr()),
            static_cast<uint32_t>(absX.GetPhyAddr()), static_cast<uint32_t>(scaleB.Get<float>().GetPhyAddr()),
            static_cast<uint32_t>(i8.GetPhyAddr()), static_cast<uint32_t>(r.GetPhyAddr()),
            static_cast<uint32_t>(brc.GetPhyAddr()), static_cast<uint32_t>(nextQScale.GetPhyAddr()),
            static_cast<uint32_t>(softTmp.GetPhyAddr())};
    }
    {
        // SCFA InitSinks keeps/broadcasts sinks in UB. All grouped entries
        // require H8; cache the normal and head-four rotation once. A q3
        // tail gives 12 rows per vector partner, and a one-query tail gives
        // four. Both used to reread GM scalarly for the second partner.
        auto cachedSinks = sinksB.Get<float>();
        DataCopy(cachedSinks, Gm<float>(p.sinks), 8);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0); WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        for (uint32_t head = 0; head < 8; ++head)
            cachedSinks.SetValue(8 + head, cachedSinks.GetValue((head + 4) % 8));
        SetFlag<HardEvent::S_V>(EVENT_ID0); WaitFlag<HardEvent::S_V>(EVENT_ID0);
        if constexpr (SHARED_QUERY_GROUP > 1) {
            CreateVecIndex(indices.ReinterpretCast<float>(), 0.0f, N);
            PipeBarrier<PIPE_V>();
        }
    }
    // Frozen channel scales and sinks are invariant for the entire launch.
    // Keep both Q-folding scales and their PV factors resident, as upstream
    // SCFA keeps invariant sinks in UB. This removes per-task scale DMA,
    // scaling and synchronization without changing rounding order.
    DataCopy(scale, Gm<float>(p.oriScale), D);
    DataCopy(scale[D], Gm<float>(p.cmpScale), D);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    Muls(scale[2 * D], scale, 1.0f / 127.0f, D);
    Muls(scale[3 * D], scale[D], 1.0f / 127.0f, D);
    PipeBarrier<PIPE_V>();
    const uint32_t core = GetBlockIdx() / 2, sub = GetBlockIdx() % 2;
    const bool residual = SHARED_HEADS(p) <= 8;
    uint64_t scratch = p.workspace + static_cast<uint64_t>(core) * WORKSPACE_CORE_BYTES;
    auto qg = Gm<int8_t>(scratch + Q_OFFSET), pg = Gm<int8_t>(scratch + P_OFFSET);
    auto sg = Gm<int32_t>(scratch + SCORE_OFFSET), vg = Gm<int32_t>(scratch + PV_OFFSET);
    auto query = Gm<bfloat16_t>(p.q); auto output = Gm<bfloat16_t>(p.out);
    const uint32_t tasks = TaskCount(p, splits, uniformQueries);
    const uint32_t scheduled = SHARED_QUERY_GROUP == 4 ?
        (tasks + GetBlockNum() - 1) / GetBlockNum() * GetBlockNum() : tasks;
    bool queryStaged = false;
    for (uint32_t task = core; task < scheduled; task += GetBlockNum()) {
        const uint32_t index = TaskIndex(task, tasks, taskOrder);
        if constexpr (SHARED_QUERY_GROUP == 4) { if (index >= tasks) continue; }
        Selection sel(p, index, splits, uniformQueries);
        bool useStagedQuery = queryStaged;
        queryStaged = false;
        const bool pairPipeline = PipelineTwoKinds(p, sel, splits);
        if (SHARED_QUERY_GROUP == 1 && sel.NeedsCompact(p))
            CompactSelection(p, sel, compactB.Get<int32_t>(), absX, scratch);
        // H8/G1 always has four live heads on each vector partner. Express
        // this at the call site so vector/softmax tiling need not rediscover
        // it through the out-of-line Selection constructor.
        const uint32_t perVec = SHARED_QUERY_GROUP == 1 ? 4 : (sel.rows + 1) / 2;
        const uint32_t begin = sub * perVec;
        const uint32_t nr = SHARED_QUERY_GROUP == 1 ? 4 :
            (begin < sel.rows ? (sel.rows - begin < perVec ? sel.rows - begin : perVec) : 0);
        const uint64_t rowBase = static_cast<uint64_t>(sel.token) * SHARED_HEADS(p) + sel.head + begin;
        if constexpr (SHARED_PACKED_C4)
            BuildC4Membership(p, sel, begin, nr, membershipB.Get<float>(), x, absX);
        uint32_t state = 0;
        bool hasAccumulated = false;
        uint32_t stagedStart = 0xffffffffU;
        uint32_t cachedPages = 0;
        Duplicate(acc, 0.0f, R * D);
        Duplicate(sumBuf, sel.part == 0 ? 1.0f : 0.0f, 2 * R);
        PipeBarrier<PIPE_ALL>();
        {
            for (uint32_t row = 0; row < nr; row += 8)
                DataCopy(maxBuf[row], sinksB.Get<float>()[(begin & 4) ? 8 : 0], 8);
            // Rounded UB copies stay inside maxBuf[0:R]; nr controls softmax.
            PipeBarrier<PIPE_V>();
        }
        if (SHARED_QUERY_GROUP == 1 && SHARED_RATIO(p) == 4 && sel.Start(p, 1) < sel.Count(p, 1)) {
            cachedPages = CachedC4PageCount(p, sel);
            if (cachedPages) {
                DataCopyExtParams cp{1, cachedPages * 4, 0, 0, 0};
                DataCopyPad(pagesB.Get<int32_t>(), Gm<int32_t>(p.cmpTable)[sel.batch * p.cmpTableWidth], cp,
                            DataCopyPadExtParams<int32_t>{});
                SetFlag<HardEvent::MTE2_S>(EVENT_ID0); WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            }
        }
        // With G1/C128 the two scale-dependent Qs occupy the otherwise
        // unused UB rows. Encode them in one vector phase. Their products,
        // softmax order and separate channel dequantization stay unchanged.
        const bool dualQ = SHARED_QUERY_GROUP == 1 && C8_RATIO == 128 && splits == 1 && sel.cmpCount > 0;
        if (dualQ) {
            DataCopy(q, query[rowBase * D], nr * D);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Cast(x, q, RoundMode::CAST_NONE, nr * D); PipeBarrier<PIPE_V>();
            DataCopy(x[nr * D], x, nr * D); PipeBarrier<PIPE_V>();
            ScaleChannels(x, x, scaleB.Get<float>(), nr);
            ScaleChannels(x[nr * D], x[nr * D], scaleB.Get<float>()[D], nr);
            PipeBarrier<PIPE_V>();
            EncodeQueryRows(x, absX, r, brc, h, i8, 2 * nr, residual);
            DataCopy(r[32], r, (2 * nr + 7) / 8 * 8); PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0); WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            for (uint32_t kind = 0; kind < 2; ++kind) {
                auto dst = Gm<int8_t>(scratch + (kind ? Q_SECOND_OFFSET : Q_OFFSET));
                DataCopy(dst[begin * D], i8[kind * nr * D], nr * D);
                DataCopy(dst[(begin + RESIDUAL_ROW_OFFSET) * D], i8[R * D + kind * nr * D], nr * D);
            }
            PipeBarrier<PIPE_ALL>();
            CrossCoreSetFlag<2, PIPE_MTE3>(Q_READY);
            CrossCoreSetFlag<2, PIPE_MTE3>(Q_READY_SECOND);
        }
        if constexpr (SHARED_QUERY_GROUP > 1) {
            if (pairPipeline && !useStagedQuery) {
                // The current task and its two-kind eligibility are already
                // validated. Share the same encoding body with lookahead;
                // input metadata is immutable throughout this launch.
                StageGroupedQuery(p, task, tasks, taskOrder, uniformQueries, scratch, stageBuffers);
                useStagedQuery = true;
            }
            if (useStagedQuery && nr) {
                DataCopy(maxBuf[R], nextQScale, (nr + 7) / 8 * 8);
                DataCopy(r[32], nextQScale[R], (nr + 7) / 8 * 8);
                PipeBarrier<PIPE_V>();
            }
        }
        const bool jointTile = JOINT_DENSE_TILE && pairPipeline;
        if (jointTile) {
            const uint32_t nextTask = task + GetBlockNum();
            queryStaged = JointTileVector(p, sel, begin, nr, scratch, x, absX, r, brc, h, i8, mask,
                            indices.ReinterpretCast<float>(), scaleB.Get<float>(), acc, maxBuf, sumBuf, expBuf, softTmp,
                            nextTask < scheduled ? nextTask : 0xffffffffu, tasks, taskOrder, uniformQueries, stageBuffers);
            state = R;
            hasAccumulated = true;
        }
        if (!jointTile) for (uint32_t kind = 0; kind < (SHARED_RATIO(p) == 1 ? 1u : 2u); ++kind) {
            const uint32_t count = sel.Count(p, kind);
            if (sel.Start(p, kind) >= count) continue;
            // Grouped pairs use the shared stage above; G1 joint tasks use
            // dualQ. Only ordinary one-kind encoding reaches this block.
            if (!dualQ && !useStagedQuery) {
                auto scale = scaleB.Get<float>()[kind * D];
                auto qdst = Gm<int8_t>(scratch + Q_OFFSET);
                if (nr) {
                    DataCopy(q, query[rowBase * D], nr * D);
                    SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                    Cast(x, q, RoundMode::CAST_NONE, nr * D); PipeBarrier<PIPE_V>();
                    ScaleChannels(x, x, scale, nr);
                    PipeBarrier<PIPE_V>();
                    EncodeQueryRows(x, absX, r, brc, h, i8, nr, residual);
                    SetFlag<HardEvent::V_MTE3>(EVENT_ID0); WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                    DataCopy(qdst[begin * D], i8, nr * D);
                    if (residual) DataCopy(qdst[(begin + RESIDUAL_ROW_OFFSET) * D], i8[R * D], nr * D);
                    PipeBarrier<PIPE_ALL>();
                }
                CrossCoreSetFlag<2, PIPE_MTE3>(Q_READY);
            }
            auto scale = scaleB.Get<float>()[(2 + kind) * D];
            if (dualQ) {
                // Four-head halves need scalar rotation because UB DMA is
                // 32-byte aligned; eight-head halves use one aligned copy.
                if (nr == 8) {
                    DataCopy(r, r[32 + kind * nr], nr); PipeBarrier<PIPE_V>();
                } else {
                    DataCopy(r, r[32], 8); PipeBarrier<PIPE_V>();
                    if (kind) {
                        VectorToScalar();
                        for (uint32_t row = 0; row < nr; ++row) r.SetValue(row, r.GetValue(nr + row));
                        SetFlag<HardEvent::S_V>(EVENT_ID0); WaitFlag<HardEvent::S_V>(EVENT_ID0);
                    }
                }
            }
            if (pairPipeline && !dualQ && nr) {
                DataCopy(r, kind == 0 ? maxBuf[R] : r[32], (nr + 7) / 8 * 8);
                PipeBarrier<PIPE_V>();
            }
            for (uint32_t start = sel.Start(p, kind); start < count; start += N * splits) {
                if (SHARED_QUERY_GROUP == 1 && kind == 1 && SHARED_RATIO(p) == 4 && start != stagedStart) {
                    GatherSparse(p, sel, start, sub, cachedPages != 0, gatherB.Get<int8_t>(), indices, compactB.Get<int32_t>(), mask, x,
                                 absX.ReinterpretCast<uint8_t>(), pagesB.Get<int32_t>(), Gm<int8_t>(scratch + KV_OFFSET));
                    stagedStart = start;
                }
                CrossCoreWaitFlag(pairPipeline && kind ? SCORE_READY_SECOND : SCORE_READY);
                auto score = Gm<int32_t>(scratch + (pairPipeline && kind ? SCORE_SECOND_OFFSET : SCORE_OFFSET));
                if (nr) {
                    const uint32_t columns = TileColumns(count - start);
                    if (SHARED_QUERY_GROUP == 1 && (kind == 0 || (kind == 1 && SHARED_RATIO(p) == 128))) {
                        uint32_t valid = count - start < N ? count - start : N;
                        Duplicate(mask, 0.0f, columns);
                        if (valid < columns) {
                            // Only the first partial 32 B block requires
                            // scalar writes. Fill the aligned remainder with
                            // one vector operation, including wide-tile padding.
                            uint32_t aligned = (valid + 7) / 8 * 8;
                            if (aligned != valid) {
                                VectorToScalar();
                                for (uint32_t n = valid; n < aligned; ++n) mask.SetValue(n, -1e30f);
                                SetFlag<HardEvent::S_V>(EVENT_ID0); WaitFlag<HardEvent::S_V>(EVENT_ID0);
                            }
                            if (aligned < columns) {
                                PipeBarrier<PIPE_V>();
                                Duplicate(mask[aligned], -1e30f, columns - aligned);
                            }
                        }
                    }
                    // c220 GM stride is bytes; UB rows are compact. Fixed
                    // GM pitch and residual row offsets match StoreAcc.
                    DataCopyExtParams scoreCopy{static_cast<uint16_t>(nr), columns * 4, (N - columns) * 4, 0, 0};
                    DataCopyPad(sc, score[begin * N], scoreCopy, DataCopyPadExtParams<int32_t>{});
                    if (residual) DataCopyPad(absX.ReinterpretCast<int32_t>(),
                        score[(begin + RESIDUAL_ROW_OFFSET) * N], scoreCopy, DataCopyPadExtParams<int32_t>{});
                    SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                    Cast(x, sc, RoundMode::CAST_RINT, nr * columns); PipeBarrier<PIPE_V>();
                    if (residual) AddResidualProduct(x, absX.ReinterpretCast<int32_t>(), nr * columns);
                    Muls(r[R], r, p.softmaxScale, nr); PipeBarrier<PIPE_V>();
                    Brcb(brc, r[R], (nr + 7) / 8, {1, 8}); PipeBarrier<PIPE_V>();
                    ScaleRows(x, x, brc, nr, columns);
                    PipeBarrier<PIPE_V>();
                    for (uint32_t row = 0; row < nr; ++row) {
                        if constexpr (SHARED_QUERY_GROUP > 1) {
                            if (row == 0 || (begin + row) % SHARED_HEADS(p) == 0) {
                                if (SHARED_PACKED_C4 && kind == 1) {
                                    uint32_t localQ = (begin + row) / SHARED_HEADS(p) - begin / SHARED_HEADS(p);
                                    DataCopy(mask, membershipB.Get<float>()[localQ * N], columns);
                                    PipeBarrier<PIPE_V>();
                                } else {
                                int32_t pos = sel.position + (begin + row) / SHARED_HEADS(p);
                                int32_t low = kind == 0 ? pos + 1 - static_cast<int32_t>(WINDOW_ROWS) -
                                    static_cast<int32_t>(sel.firstOri + start) : 0;
                                int32_t high = kind == 0 ? pos + 1 - static_cast<int32_t>(sel.firstOri + start)
                                    : (pos + 1) / static_cast<int32_t>(SHARED_RATIO(p)) - static_cast<int32_t>(start);
                                auto bits = absX.ReinterpretCast<uint8_t>();
                                CompareScalar(bits, indices.ReinterpretCast<float>(), static_cast<float>(low), CMPMODE::GE, columns);
                                PipeBarrier<PIPE_V>();
                                CompareScalar(bits[64], indices.ReinterpretCast<float>(), static_cast<float>(high), CMPMODE::LT, columns);
                                PipeBarrier<PIPE_V>();
                                And(bits.ReinterpretCast<uint16_t>(), bits.ReinterpretCast<uint16_t>(),
                                    bits[64].ReinterpretCast<uint16_t>(), columns / 16); PipeBarrier<PIPE_V>();
                                Duplicate(mask, 0.0f, columns); PipeBarrier<PIPE_V>();
                                Select(mask, bits, mask, -1e30f, SELMODE::VSEL_TENSOR_SCALAR_MODE, columns); PipeBarrier<PIPE_V>();
                                }
                            }
                        }
                        Add(x[row * columns], x[row * columns], mask, columns);
                    }
                    PipeBarrier<PIPE_V>();
                    const uint32_t next = state ^ R;
                    CompactSoftmax<R, R, true>(nr, columns, softmaxBuffers, state);
                    state = next;
                    PipeBarrier<PIPE_V>();
                    if (hasAccumulated) {
                        Brcb(brc, expBuf, (nr + 7) / 8, {1, 8}); PipeBarrier<PIPE_V>();
                        ScaleRows(acc, acc, brc, nr, D);
                    }
                    Muls(x, x, 127.0f, nr * columns); PipeBarrier<PIPE_V>();
                    if (residual) RoundResidualInt8(i8, i8[R * D], h, x, absX, nr * columns);
                    else RoundInt8(i8, h, x, nr * columns);
                    SetFlag<HardEvent::V_MTE3>(EVENT_ID0); WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                    DataCopyExtParams probabilityCopy{static_cast<uint16_t>(nr), columns, 0, N - columns, 0};
                    DataCopyPad(pg[begin * N], i8, probabilityCopy);
                    if (residual) DataCopyPad(pg[(begin + RESIDUAL_ROW_OFFSET) * N], i8[R * D], probabilityCopy);
                    PipeBarrier<PIPE_ALL>();
                }
                CrossCoreSetFlag<2, PIPE_MTE3>(P_READY);
                // The final P has retired from Vector and both current Qs
                // have already been consumed. Encode the next retained-KV
                // task while Cube computes this PV. Q0/Q1 GM are disjoint
                // from P/PV; only their alpha values need extra UB storage.
                if constexpr (SHARED_QUERY_GROUP > 1) {
                    const uint32_t nextTask = task + GetBlockNum();
                    if (pairPipeline && kind == 1 && nextTask < scheduled) {
                        queryStaged = StageGroupedQuery(p, nextTask, tasks, taskOrder, uniformQueries, scratch, stageBuffers);
                    }
                }
                // Once SCORE_READY was observed, the current KV is retained
                // in L1. Its GM staging tile can hold the next selection while
                // Cube computes PV. Mirrors upstream SCFA's staggered V0/C2
                // schedule, without reusing live L1 or score/P buffers.
                if (SHARED_QUERY_GROUP == 1 && SHARED_RATIO(p) == 4 && (kind == 1 || start + N * splits >= count)) {
                    uint32_t next = kind == 1 ? start + N * splits : sel.Start(p, 1);
                    if (next < sel.Count(p, 1) && next != stagedStart) {
                        GatherSparse(p, sel, next, sub, cachedPages != 0, gatherB.Get<int8_t>(), indices, compactB.Get<int32_t>(), mask, x,
                                     absX.ReinterpretCast<uint8_t>(), pagesB.Get<int32_t>(), Gm<int8_t>(scratch + KV_OFFSET));
                        stagedStart = next;
                    }
                }
                CrossCoreWaitFlag(PV_READY);
                if (nr) {
                    if constexpr (SHARED_QUERY_GROUP > 1) {
                        if (queryStaged) {
                            SetFlag<HardEvent::V_MTE2>(EVENT_ID0); WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                        }
                    }
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
                Brcb(brc, r[R], (nr + 7) / 8, {1, 8}); PipeBarrier<PIPE_V>();
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
