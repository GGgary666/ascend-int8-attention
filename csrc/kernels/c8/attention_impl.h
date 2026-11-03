// C8 production core: frozen-channel KV, Q/P double residuals.
#include "args.h"
#include "vector_math.h"
#include "residual_quant.h"
#include "joint_softmax.h"
#include "lib/activation/softmaxflashv2.h"

using namespace AscendC;
using namespace sharedkv;
#ifndef SHARED_KERNEL_NAMESPACE
#define SHARED_KERNEL_NAMESPACE sharedkv_kernel
#endif

namespace SHARED_KERNEL_NAMESPACE {
namespace {
#include "config.h"
constexpr IsResetLoad3dConfig FUSED_LOAD_CFG{true, true};
template<class T> __aicore__ inline GlobalTensor<T> Gm(uint64_t address)
{
    GlobalTensor<T> tensor;
    tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(address));
    return tensor;
}



struct Selection {
    uint32_t token, head, rows, batch, position, firstOri, oriCount, cmpCount, part, sparseCount, splitCount;
    __aicore__ Selection(const SharedKVArgs &p, uint32_t task, uint32_t splits, uint32_t uniformQueries)
    {
        splitCount = splits;
        sparseCount = C4_TOPK;
        auto cu = Gm<int32_t>(p.cuQ);
#if SHARED_QUERY_GROUP > 1
        batch = 0;
        uint32_t qBegin = 0, qEnd;
        if (uniformQueries) {
            const uint32_t groups = (uniformQueries + SHARED_QUERY_GROUP - 1) / SHARED_QUERY_GROUP;
            batch = task / groups; task %= groups;
            qBegin = batch * uniformQueries; qEnd = qBegin + uniformQueries;
        } else {
            qEnd = cu.GetValue(1);
            while (batch + 1 < p.batch) {
                uint32_t groups = (qEnd - qBegin + SHARED_QUERY_GROUP - 1) / SHARED_QUERY_GROUP;
                if (task < groups) break;
                task -= groups; ++batch; qBegin = qEnd; qEnd = cu.GetValue(batch + 1);
            }
        }
        part = 0; head = 0; token = qBegin + task * SHARED_QUERY_GROUP;
        uint32_t qCount = qEnd - token < SHARED_QUERY_GROUP ? qEnd - token : SHARED_QUERY_GROUP;
        rows = qCount * SHARED_HEADS(p);
        position = Gm<int32_t>(p.seqLens).GetValue(batch) - qEnd + token;
        firstOri = position + 1 > WINDOW_ROWS ? position + 1 - WINDOW_ROWS : 0;
        oriCount = position + qCount - firstOri;
        cmpCount = SHARED_RATIO(p) == 1 ? 0 : (position + qCount) / SHARED_RATIO(p);
        if constexpr (SHARED_PACKED_C4) sparseCount = cmpCount;
#else
        const uint32_t headTiles = (SHARED_HEADS(p) + M - 1) / M;
        part = task % splits; task /= splits;
        token = task / headTiles; head = task % headTiles * M;
        rows = SHARED_HEADS(p) - head < M ? SHARED_HEADS(p) - head : M;
        uint32_t qEnd;
        if (uniformQueries) {
            batch = token / uniformQueries; qEnd = (batch + 1) * uniformQueries;
        } else {
            batch = 0;
            while (batch + 1 < p.batch && token >= static_cast<uint32_t>(cu.GetValue(batch + 1))) ++batch;
            qEnd = cu.GetValue(batch + 1);
        }
        position = Gm<int32_t>(p.seqLens).GetValue(batch) - qEnd + token;
        oriCount = position + 1 < WINDOW_ROWS ? position + 1 : WINDOW_ROWS;
        firstOri = position + 1 - oriCount;
        cmpCount = SHARED_RATIO(p) == 1 ? 0 : (position + 1) / SHARED_RATIO(p);
#endif
    }
    __aicore__ uint32_t Count(const SharedKVArgs &p, uint32_t kind) const
    { return kind == 0 ? oriCount : (SHARED_RATIO(p) == 4 ? sparseCount : cmpCount); }
    __aicore__ bool NeedsCompact(const SharedKVArgs &p) const
    { return SHARED_QUERY_GROUP == 1 && SHARED_RATIO(p) == 4 && cmpCount < C4_TOPK && C4_TOPK <= 1024; }
    __aicore__ uint32_t Start(const SharedKVArgs &p, uint32_t kind) const
    {
        uint32_t previous = kind == 0 ? 0 : (oriCount + N - 1) / N;
        return ((part + splitCount - previous % splitCount) % splitCount) * N;
    }
    __aicore__ int32_t Index(const SharedKVArgs &p, uint32_t kind, uint32_t slot) const
    {
        if (slot >= Count(p, kind)) return -1;
        if (kind == 0) return firstOri + slot;
        int32_t index = SHARED_RATIO(p) == 4 ? Gm<int32_t>(p.indices).GetValue(static_cast<uint64_t>(token) * C4_TOPK + slot) : slot;
        return index >= 0 && static_cast<uint32_t>(index) < cmpCount ? index : -1;
    }
};

__aicore__ inline bool PipelineTwoKinds(const SharedKVArgs &p, const Selection &sel, uint32_t splits)
{
    if constexpr (SHARED_QUERY_GROUP > 1 || JOINT_DENSE_TILE)
        return (SHARED_RATIO(p) == 128 || SHARED_PACKED_C4) && splits == 1 && sel.oriCount > 0 &&
            sel.oriCount <= N && sel.cmpCount > 0 && sel.cmpCount <= N;
    return false;
}

__aicore__ inline uint32_t ActualBatch(const SharedKVArgs &p)
{
    // vLLM FULL_DECODE_ONLY pads cuQ with synthetic six-token requests,
    // while seqLens for those requests is zero. Other paths use -1 cuQ
    // sentinels. Captured p.batch is capacity, not the live request count.
    auto lengths = Gm<int32_t>(p.seqLens);
    auto cu = Gm<int32_t>(p.cuQ);
    uint32_t batch = p.batch;
    while (batch && (lengths.GetValue(batch - 1) <= 0 || cu.GetValue(batch) < 0)) --batch;
    return batch;
}

__aicore__ inline uint32_t ActualTokens(const SharedKVArgs &p)
{
    uint32_t batch = ActualBatch(p);
    int32_t count = batch ? Gm<int32_t>(p.cuQ).GetValue(batch) : 0;
    return count <= 0 ? 0 : static_cast<uint32_t>(count) < p.tokens ? static_cast<uint32_t>(count) : p.tokens;
}

__aicore__ inline SharedKVLiveShape ReadLiveShape(const SharedKVArgs &p)
{
    auto lengths = Gm<int32_t>(p.seqLens);
    auto cu = Gm<int32_t>(p.cuQ);
    uint32_t batch = p.batch;
    int32_t end = 0;
    while (batch) {
        if (lengths.GetValue(batch - 1) > 0) {
            end = cu.GetValue(batch);
            if (end >= 0) break;
        }
        --batch;
    }
    const uint32_t tokens = !batch || end <= 0 ? 0 :
        static_cast<uint32_t>(end) < p.tokens ? static_cast<uint32_t>(end) : p.tokens;
    return SharedKVLiveShape{batch, tokens};
}

__aicore__ inline uint32_t TaskCount(const SharedKVArgs &p, uint32_t splits, uint32_t uniformQueries)
{
#if SHARED_QUERY_GROUP > 1
    if (uniformQueries) return SHARED_LIVE_BATCH * ((uniformQueries + SHARED_QUERY_GROUP - 1) / SHARED_QUERY_GROUP);
    auto cu = Gm<int32_t>(p.cuQ);
    uint32_t count = 0, begin = 0;
    const uint32_t batch = SHARED_LIVE_BATCH;
    for (uint32_t b = 0; b < batch; ++b) {
        uint32_t end = cu.GetValue(b + 1);
        count += (end - begin + SHARED_QUERY_GROUP - 1) / SHARED_QUERY_GROUP; begin = end;
    }
    return count;
#else
    return SHARED_LIVE_TOKENS * ((SHARED_HEADS(p) + M - 1) / M) * splits;
#endif
}

__aicore__ inline void LoadA(LocalTensor<int8_t> a0, LocalTensor<int8_t> a1, uint32_t k)
{
    // At M16, L1 NZ and L0A ZZ have identical fractal order. The ordinary
    // c220 LoadData2D needs one instruction and no image-to-column setup.
    // Upstream A2 copy_l1_to_l0a uses the same K-fractal stride. V2 is not
    // implemented on this architecture; wider M still needs Load3D.
    if constexpr (M == 16) {
        LoadData2DParams lp{};
        lp.repeatTimes = k / 32; lp.srcStride = 1;
        LoadData(a0, a1, lp);
        return;
    }
    LoadData3DParamsV2<int8_t> lp{};
    lp.l1H = M / 16; lp.l1W = 16; lp.channelSize = k; lp.padList[3] = 255;
    lp.mExtension = M; lp.kExtension = k;
    lp.strideW = lp.strideH = lp.filterW = lp.filterH = 1;
    lp.dilationFilterW = lp.dilationFilterH = 1;
    LoadData<int8_t, FUSED_LOAD_CFG>(a0, a1, lp);
}

__aicore__ inline void StoreAcc(GlobalTensor<int32_t> dst, LocalTensor<int32_t> c,
                               uint32_t rows, uint32_t cols, uint32_t stride)
{
    SetFlag<HardEvent::M_FIX>(EVENT_ID0); WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
    DataCopyCO12DstParams fp{};
    fp.mSize = rows; fp.nSize = cols; fp.srcStride = M; fp.dstStride = stride;
    fp.quantPre = QuantMode_t::NoQuant; fp.nz2ndEn = true;
    SetFixpipeNz2ndFlag(1, 1, 1);
    DataCopy(dst, c, fp);
    // Close the L0C read lifetime before any subsequent MMAD reuses it.
    // Upstream arch22 msa_index_score likewise uses FIX_M after Fixpipe;
    // a generic PIPE_ALL barrier is not this resource's reuse contract.
    SetFlag<HardEvent::FIX_M>(EVENT_ID0); WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
}

// NZ INT8 KV is [D/32, physicalRows,32]. Only the padded rows that Cube
// actually reads need zeroing; valid rows are completely replaced by DMA.
// c220 InitConstValue's L1 blockNum/dstGap are both in32-byte blocks.
__aicore__ inline void ZeroKvTail(LocalTensor<int8_t> kv, uint32_t valid,
                                  uint32_t physicalRows, uint32_t computedRows)
{
    if (valid < computedRows) {
        const uint32_t zeros = computedRows - valid;
        InitConstValue(kv[valid * 32].ReinterpretCast<uint32_t>(),
            InitConstValueParams<uint32_t>{D / 32, static_cast<uint16_t>(zeros),
                                          static_cast<uint16_t>(physicalRows - zeros), 0});
        PipeBarrier<PIPE_MTE2>();
    }
}

// Upstream A2 copy_l1_to_l0b_a2.hpp: repeat across the16 D-planes,
// with one instruction per32-key slice. INT8 srcStride counts1024B;
// dstGap=1 places consecutive pairs of512B destination fractals together.
__aicore__ inline void LoadValueB(LocalTensor<int8_t> dst, LocalTensor<int8_t> kv,
                                 uint32_t physicalRows, uint32_t start, uint32_t count)
{
    LoadData2dTransposeParams lp{};
    lp.repeatTimes = D / 32; lp.srcStride = physicalRows / 32;
    lp.dstGap = 1; lp.dstFracGap = 0;
    for (uint32_t k = 0; k < count; k += 32)
        LoadDataWithTranspose(dst[k * D], kv[(start + k) * 32], lp);
}

#include "cube_pair.h"

// Reordering changes only task ownership; physical-core scratch and all
// Cube/Vector producer-consumer indices retain their original layout.
__aicore__ inline uint32_t TaskIndex(uint32_t task, uint32_t tasks, uint32_t order)
{
    // G4 compares linear and alternating waves by peak cost; the unrelated
    // two-wave packed-index encoding is compile-pruned from this path.
    if constexpr (SHARED_QUERY_GROUP == 4) {
        if constexpr (!SHARED_ALTERNATING_WAVES)
            if (order & sharedkv_c8::kPairedOrder)
                return task < sharedkv_c8::kCores ? order & 63u : (order >> 6) & 63u;
        return SHARED_ALTERNATING_WAVES ? sharedkv_c8::SerpentineTask(task) : task;
    }
    if (!order) return task;
    return task < sharedkv_c8::kCores ? order & 63u : (order >> 6) & 63u;
}

template<bool Pipeline>
__aicore__ void Cube(const SharedKVArgs &p, uint32_t splits, uint32_t taskOrder, uint32_t uniformQueries)
{
    static_assert(M * D <= 64 * 1024 && M * D * 4 <= 128 * 1024,
                  "Cube L0A/L0C exceed dav_c220 capacity");
    static_assert(KV_L1_SLOTS * N * D + M * (D + N) <= 512 * 1024 - 128,
                  "Cube L1 exceeds dav_c220 user capacity");
    TPipe pipe;
    TBuf<TPosition::A1> qb, pb;
    TBuf<TPosition::B1> kb;
    TBuf<TPosition::A2> ab;
    TBuf<TPosition::B2> bb;
    TBuf<TPosition::CO1> cb;
    pipe.InitBuffer(qb, M * D); pipe.InitBuffer(pb, M * N);
    pipe.InitBuffer(kb, KV_L1_SLOTS * N * D); pipe.InitBuffer(ab, M * D);
    pipe.InitBuffer(bb, 65536); pipe.InitBuffer(cb, M * D * 4);
    auto q1 = qb.Get<int8_t>(); auto p1 = pb.Get<int8_t>();
    auto k1 = kb.Get<int8_t>(); auto a0 = ab.Get<int8_t>();
    auto b0 = bb.Get<int8_t>(); auto c = cb.Get<int32_t>();
    uint64_t scratch = p.workspace + static_cast<uint64_t>(GetBlockIdx()) * WORKSPACE_CORE_BYTES;
    auto qg = Gm<int8_t>(scratch + Q_OFFSET), pg = Gm<int8_t>(scratch + P_OFFSET);
    auto sg = Gm<int32_t>(scratch + SCORE_OFFSET), vg = Gm<int32_t>(scratch + PV_OFFSET);
    const uint32_t tasks = TaskCount(p, splits, uniformQueries);
    const uint32_t scheduled = SHARED_QUERY_GROUP == 4 ?
        (tasks + GetBlockNum() - 1) / GetBlockNum() * GetBlockNum() : tasks;
    for (uint32_t task = GetBlockIdx(); task < scheduled; task += GetBlockNum()) {
        const uint32_t index = TaskIndex(task, tasks, taskOrder);
        if constexpr (SHARED_QUERY_GROUP == 4) { if (index >= tasks) continue; }
        Selection sel(p, index, splits, uniformQueries);
        // Eight-head model shards leave half of the physical M=16 Cube
        // rows unused. Compute the Q/P rounding residual in those rows.
        const uint32_t calcRows = SHARED_QUERY_GROUP == 1 ? 16 : sel.rows + RESIDUAL_ROW_OFFSET;
        if (sel.NeedsCompact(p)) {
            CrossCoreWaitFlag(COMPACT_READY);
            auto slots = Gm<uint32_t>(scratch + SLOTS_OFFSET);
            sel.sparseCount = slots.GetValue(0);
            // Retire the scalar DCache line before the next AIV DMA write.
            // See upstream fused_quant_lightning_indexer_manage.cpp:105.
            DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(slots);
            CrossCoreSetFlag<2, PIPE_MTE2>(COMPACT_CONSUMED);
        }
        if constexpr (SHARED_QUERY_GROUP > 1 || JOINT_DENSE_TILE) {
            if (PipelineTwoKinds(p, sel, splits)) {
                CubePair<Pipeline, JOINT_DENSE_TILE>(p, sel, calcRows, scratch, q1, p1, k1, a0, b0, c);
                continue;
            }
        }
        for (uint32_t kind = 0; kind < (SHARED_RATIO(p) == 1 ? 1u : 2u); ++kind) {
            const uint32_t count = sel.Count(p, kind);
            if (sel.Start(p, kind) >= count) continue;
            const bool dualQ = SHARED_QUERY_GROUP == 1 && C8_RATIO == 128 && splits == 1 && sel.cmpCount > 0;
            CrossCoreWaitFlag(dualQ && kind ? Q_READY_SECOND : Q_READY);
            Nd2NzParams qp{};
            qp.ndNum = 1; qp.nValue = calcRows; qp.dValue = D; qp.srcDValue = D;
            qp.dstNzC0Stride = M; qp.dstNzNStride = 1;
            DataCopy(q1, dualQ && kind ? Gm<int8_t>(scratch + Q_SECOND_OFFSET) : qg, qp);
            auto cache = Gm<int8_t>(kind == 0 ? p.ori : p.cmp);
            auto table = Gm<int32_t>(kind == 0 ? p.oriTable : p.cmpTable);
            uint32_t block = kind == 0 ? BLOCK_ROWS : BLOCK_ROWS;
            uint32_t tableWidth = kind == 0 ? p.oriTableWidth : p.cmpTableWidth;
            uint64_t stride = kind == 0 ? p.oriStride : p.cmpStride;
            for (uint32_t start = sel.Start(p, kind); start < count; start += N * splits) {
                // The SWA side of a wide C4 tile usually has only 128 keys.
                // Upstream SCFA sizes Cube work from actualSingleProcessSInnerSize.
                // Retain workspace/softmax strides but omit padded Cube columns.
                const uint32_t remaining = count - start;
                const uint32_t cubeN = TileColumns(remaining);
                // Load each selected token directly into the reusable L1 KV
                // tile. QK and PV consume the same bytes, with different L0B
                // transforms. Invalid slots read a safe token then get masked.
                Nd2NzParams kp{};
                kp.ndNum = 1; kp.nValue = 1; kp.dValue = D; kp.srcDValue = D;
                kp.dstNzC0Stride = cubeN; kp.dstNzNStride = 1;
                if (SHARED_QUERY_GROUP == 1 && (kind == 1 && SHARED_RATIO(p) == 4)) {
                    CrossCoreWaitFlag(KV_READY);
                    kp.nValue = cubeN;
                    DataCopy(k1, Gm<int8_t>(scratch + KV_OFFSET), kp);
                } else {
                    uint32_t valid = count - start < N ? count - start : N;
                    ZeroKvTail(k1, valid, cubeN, cubeN);
                    for (uint32_t n = 0; n < valid;) {
                        uint32_t idx = (kind == 0 ? sel.firstOri : 0) + start + n;
                        uint32_t run = block - idx % block;
                        run = run < valid - n ? run : valid - n;
                        kp.nValue = run;
                        uint32_t page = table.GetValue(sel.batch * tableWidth + idx / block);
                        DataCopy(k1[n * 32], cache[page * stride + static_cast<uint64_t>(idx % block) * D], kp);
                        n += run;
                    }
                }
                SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0); WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
                MmadParams mm{};
                // SCFA uses two 32 KiB L0B slots. A slot-specific M->MTE1
                // event protects reuse while the next slot loads in parallel.
                const uint32_t qkK = (Pipeline ? 32768 : 65536) / cubeN;
                if constexpr (Pipeline) for (uint32_t slot = 0; slot < 2; ++slot)
                    SetFlag<HardEvent::M_MTE1>(EVENT_ID3 + slot);
                for (uint32_t k = 0; k < D; k += qkK) {
                    const uint32_t slot = Pipeline ? (k / qkK) % 2 : 0;
                    const uint32_t event = Pipeline ? EVENT_ID3 + slot : EVENT_ID0;
                    auto ak = a0[slot * M * qkK], bk = b0[slot * 32768];
                    if constexpr (Pipeline) WaitFlag<HardEvent::M_MTE1>(event);
                    LoadA(ak, q1[k * M], qkK);
                    LoadData2DParams lk{};
                    lk.srcStride = 1;
                    // The physical NZ pitch follows computed columns, as in
                    // the retained-KV path. One contiguous copy covers QK.
                    lk.repeatTimes = (cubeN / 16) * (qkK / 32);
                    LoadData(bk, k1[k * cubeN], lk);
                    SetFlag<HardEvent::MTE1_M>(event); WaitFlag<HardEvent::MTE1_M>(event);
                    mm.m = M; mm.n = cubeN; mm.k = qkK; mm.cmatrixInitVal = k == 0;
                    Mmad(c, ak, bk, mm);
                    if constexpr (Pipeline) SetFlag<HardEvent::M_MTE1>(event);
                    else if constexpr (N > 128) PipeBarrier<PIPE_ALL>();
                }
                if constexpr (Pipeline) for (uint32_t slot = 0; slot < 2; ++slot)
                    WaitFlag<HardEvent::M_MTE1>(EVENT_ID3 + slot);
                StoreAcc(sg, c, calcRows, cubeN, N);
                CrossCoreSetFlag<2, PIPE_FIX>(SCORE_READY);
                CrossCoreWaitFlag(P_READY);
                PipeBarrier<PIPE_ALL>();
                Nd2NzParams pp = qp;
                pp.dValue = cubeN; pp.srcDValue = N;
                DataCopy(p1, pg, pp);
                SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0); WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
                if constexpr (Pipeline) for (uint32_t slot = 0; slot < 2; ++slot)
                    SetFlag<HardEvent::M_MTE1>(EVENT_ID3 + slot);
                constexpr uint32_t pvK = Pipeline ? 64 : 128;
                for (uint32_t k = 0; k < cubeN; k += pvK) {
                    const uint32_t slot = Pipeline ? (k / pvK) % 2 : 0;
                    const uint32_t event = Pipeline ? EVENT_ID3 + slot : EVENT_ID0;
                    auto ak = a0[slot * M * pvK], bk = b0[slot * 32768];
                    if constexpr (Pipeline) WaitFlag<HardEvent::M_MTE1>(event);
                    LoadA(ak, p1[k * M], pvK);
                    LoadValueB(bk, k1, cubeN, k, pvK);
                    SetFlag<HardEvent::MTE1_M>(event); WaitFlag<HardEvent::MTE1_M>(event);
                    mm.n = D; mm.k = pvK; mm.cmatrixInitVal = k == 0;
                    Mmad(c, ak, bk, mm);
                    if constexpr (Pipeline) SetFlag<HardEvent::M_MTE1>(event);
                    else if constexpr (N > 128) PipeBarrier<PIPE_ALL>();
                }
                if constexpr (Pipeline) for (uint32_t slot = 0; slot < 2; ++slot)
                    WaitFlag<HardEvent::M_MTE1>(EVENT_ID3 + slot);
                StoreAcc(vg, c, calcRows, D, D);
                PipeBarrier<PIPE_ALL>();
                CrossCoreSetFlag<2, PIPE_FIX>(PV_READY);
                // The next QK may overlap the vector partner's PV accumulation,
                // as in upstream SCFA PreloadPipeline's staggered C1/V2 stages.
                // SCORE_READY/P_READY protect sg/pg. The next P_READY cannot be
                // issued until this vg has been consumed; KV_READY protects the
                // next gathered tile, and the barrier above retires L0C's FIX.
            }
        }
    }
}

#include "gather.h"

#include "vector.h"

__aicore__ void ReduceParts(const SharedKVArgs &p, uint32_t splits)
{
    TPipe pipe;
    TBuf<TPosition::VECCALC> db, wb, xb, tb, ob;
    pipe.InitBuffer(db, 24 * (D + 8) * 4); pipe.InitBuffer(wb, 32 * 4);
    pipe.InitBuffer(xb, D * 4); pipe.InitBuffer(tb, D * 4); pipe.InitBuffer(ob, D * 2);
    auto data = db.Get<float>(); auto weights = wb.Get<float>();
    auto x = xb.Get<float>(); auto tmp = tb.Get<float>(); auto out = ob.Get<bfloat16_t>();
    auto pg = Gm<float>(p.partial); auto og = Gm<bfloat16_t>(p.out);
    const uint32_t rows = SHARED_LIVE_TOKENS * SHARED_HEADS(p);
    for (uint32_t row = GetBlockIdx(); row < rows; row += GetBlockNum() * 2) {
        DataCopy(data, pg[static_cast<uint64_t>(row) * splits * (D + 8)], splits * (D + 8));
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0); WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        float maximum = data.GetValue(D);
        for (uint32_t s = 1; s < splits; ++s) {
            float m = data.GetValue(s * (D + 8) + D);
            maximum = m > maximum ? m : maximum;
        }
        for (uint32_t s = 0; s < splits; ++s)
            weights.SetValue(s, data.GetValue(s * (D + 8) + D) - maximum);
        SetFlag<HardEvent::S_V>(EVENT_ID0); WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Exp(weights, weights, splits); VectorToScalar();
        float denominator = 0.0f;
        Duplicate(x, 0.0f, D); PipeBarrier<PIPE_V>();
        for (uint32_t s = 0; s < splits; ++s) {
            float factor = weights.GetValue(s);
            denominator += factor * data.GetValue(s * (D + 8) + D + 1);
            Muls(tmp, data[s * (D + 8)], factor, D); PipeBarrier<PIPE_V>();
            Add(x, x, tmp, D); PipeBarrier<PIPE_V>();
        }
        Muls(x, x, 1.0f / denominator, D); PipeBarrier<PIPE_V>();
        Cast(out, x, RoundMode::CAST_RINT, D);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0); WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(og[static_cast<uint64_t>(row) * D], out, D);
        if (p.lse) {
            weights.SetValue(0, denominator);
            SetFlag<HardEvent::S_V>(EVENT_ID0); WaitFlag<HardEvent::S_V>(EVENT_ID0);
            Ln(weights, weights, 1); PipeBarrier<PIPE_V>();
            Adds(weights, weights, maximum, 1);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0); WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopyPad(Gm<float>(p.lse)[row], weights, DataCopyExtParams{1, 4, 0, 0, 0});
        }
        PipeBarrier<PIPE_ALL>();
    }
}

__aicore__ __attribute__((always_inline)) inline void ClearPaddedRows(const SharedKVArgs &p)
{
    const uint32_t actual = SHARED_LIVE_TOKENS;
    if (actual == p.tokens) return;
    TPipe pipe;
    TBuf<TPosition::VECCALC> zb;
    pipe.InitBuffer(zb, D * 2);
    auto zero = zb.Get<uint16_t>();
    Duplicate(zero, static_cast<uint16_t>(0), D);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0); WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    for (uint32_t row = actual * SHARED_HEADS(p) + GetBlockIdx(); row < p.tokens * SHARED_HEADS(p); row += GetBlockNum() * 2) {
        DataCopy(Gm<bfloat16_t>(p.out)[static_cast<uint64_t>(row) * D], zb.Get<bfloat16_t>(), D);
        // Scalar GM writes can dirty an entire DCache line and race with
        // another AIV's neighboring LSE row. Use an exact-width DMA store.
        if (p.lse) DataCopyPad(Gm<float>(p.lse)[row], zb.Get<float>(), DataCopyExtParams{1, 4, 0, 0, 0});
    }
    PipeBarrier<PIPE_ALL>();
}

#if C8_RATIO == 128
// VectorBatched is inlined into this phase. Keep its entry on a DAV2201
// instruction-fetch block boundary as unrelated planner functions change.
__aicore__ __attribute__((aligned(2048)))
#else
__aicore__
#endif
void RunWithSplits(const SharedKVArgs &p, uint32_t splits, uint32_t taskOrder = 0, uint32_t uniformQueries = 0)
{
    // TND tensors may have graph-capacity padding. Upstream determines work
    // from cumulative actual query lengths, not the tensor's first dimension.
    // Keep the by-value launch argument immutable: ASC can address it in
    // captured parameter storage, which is reused by subsequent graph replays.
    if ASCEND_IS_AIC {
        // The C128 wide plan is selected from the whole live workload.
        // Its L0 pipeline must not depend on which request happens to be first.
        const bool denseLong = SHARED_DENSE_L0_PIPELINE;
        if (!denseLong && (SHARED_RATIO(p) == 4 || SHARED_QUERY_GROUP == 1) && SHARED_LIVE_TOKENS <= 48) Cube<false>(p, splits, taskOrder, uniformQueries);
        else Cube<true>(p, splits, taskOrder, uniformQueries);
    }
    if ASCEND_IS_AIV { VectorBatched(p, splits, taskOrder, uniformQueries); }
    if (splits > 1) {
        // All active blocks reach the same single global barrier. No block
        // spins waiting for a block that has not been scheduled.
        SyncAll<false>();
        if ASCEND_IS_AIV { ReduceParts(p, splits); }
    }
    if ASCEND_IS_AIV { ClearPaddedRows(p); }
}

} // internal device functions
} // namespace SHARED_KERNEL_NAMESPACE

