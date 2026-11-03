#include "sharedkv.h"
#include "vector_math.h"
#include "residual_quant.h"
#include "lib/activation/softmaxflashv2.h"

using namespace AscendC;
using namespace sharedkv;
#ifndef SHARED_KERNEL_NAMESPACE
#define SHARED_KERNEL_NAMESPACE sharedkv_kernel
#endif

namespace SHARED_KERNEL_NAMESPACE {
namespace {
#include "sharedkv_config.h"
constexpr IsResetLoad3dConfig FUSED_LOAD_CFG{true, true};
template<class T> __aicore__ inline GlobalTensor<T> Gm(uint64_t address)
{
    GlobalTensor<T> tensor;
    tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(address));
    return tensor;
}

#if defined(SHARED_TAIL_BALANCE) || defined(SHARED_SHORT_TASK_MAP)
__aicore__ inline uint32_t ActualBatch(const SharedKVArgs &p);
#endif

#undef SHARED_TASK_PARAM
#undef SHARED_TASK_ARG
#if defined(SHARED_TAIL_BALANCE) && SHARED_TAIL_BALANCE == 2
struct TailTaskPlan {
    uint32_t batch, full, total;
    bool balanced, uniform3;
};
#define SHARED_TASK_PARAM , const TailTaskPlan &taskPlan
#define SHARED_TASK_ARG , taskPlan
__aicore__ inline TailTaskPlan MakeTailTaskPlan(const SharedKVArgs &p)
{
    TailTaskPlan plan{ActualBatch(p),0,0,p.plan==24 || p.plan==25,true};
    auto cu=Gm<int32_t>(p.cuQ);
    uint32_t begin=0;
    for (uint32_t b=0;b<plan.batch;++b) {
        const uint32_t end=cu.GetValue(b+1), count=end-begin;
        plan.full+=count/2; plan.total+=(count+1)/2;
        plan.uniform3=plan.uniform3 && count==3; begin=end;
    }
    return plan;
}
#elif defined(SHARED_SHORT_TASK_MAP)
struct ShortTaskPlan { uint32_t batch, total; bool direct; };
#define SHARED_TASK_PARAM , const ShortTaskPlan &taskPlan
#define SHARED_TASK_ARG , taskPlan
__aicore__ inline ShortTaskPlan MakeShortTaskPlan(const SharedKVArgs &p)
{
    static_assert(SHARED_QUERY_GROUP == 3);
    ShortTaskPlan plan{ActualBatch(p),0,p.plan == 16 || p.plan == 17};
    auto cu = Gm<int32_t>(p.cuQ);
    uint32_t begin = 0;
    for (uint32_t b = 0; b < plan.batch; ++b) {
        const uint32_t end = cu.GetValue(b + 1), count = end - begin;
        plan.total += (count + SHARED_QUERY_GROUP - 1) / SHARED_QUERY_GROUP;
        plan.direct = plan.direct && count <= SHARED_QUERY_GROUP;
        begin = end;
    }
    // In the bounded case, request index directly identifies its sole task.
    // An empty interior request gets a no-op task on both Cube and Vector.
    if (plan.direct) plan.total = plan.batch;
    return plan;
}
#else
#define SHARED_TASK_PARAM
#define SHARED_TASK_ARG
#endif

struct Selection {
    uint32_t token, head, rows, batch, position, firstOri, oriCount, cmpCount, part, sparseCount, splitCount;
    __aicore__ Selection(const SharedKVArgs &p, uint32_t task, uint32_t splits SHARED_TASK_PARAM)
    {
        splitCount = splits;
        sparseCount = p.topk;
        auto cu = Gm<int32_t>(p.cuQ);
#if SHARED_QUERY_GROUP > 1
#ifdef SHARED_TAIL_BALANCE
        // Upstream sharedkv metadata assigns different costs to full/tail M.
        // In this isolated probe, plans18/19 put complete groups first;
        // plans22/23 retain request-major order in THE SAME compiled body.
        // Only task ownership changes; query grouping and key order do not.
#if SHARED_TAIL_BALANCE == 2
        const bool balanced = taskPlan.balanced;
        const uint32_t batchLimit = taskPlan.batch, fullTasks = taskPlan.full;
#else
        const bool balanced = p.plan == 18 || p.plan == 19;
        const uint32_t batchLimit = ActualBatch(p);
        uint32_t fullTasks = 0, previous = 0;
        if (balanced) for (uint32_t b = 0; b < batchLimit; ++b) {
            const uint32_t end = cu.GetValue(b + 1);
            fullTasks += (end - previous) / SHARED_QUERY_GROUP; previous = end;
        }
#endif
        const bool tail = balanced && task >= fullTasks;
        if (tail) task -= fullTasks;
        batch = 0;
        uint32_t qBegin = 0, qEnd = cu.GetValue(1);
#if SHARED_TAIL_BALANCE == 2
        if (taskPlan.uniform3) {
            batch=balanced ? task : task/2;
            qBegin=batch*3; qEnd=qBegin+3;
            token=qBegin+(balanced ? (tail ? 2u : 0u) : (task%2)*2);
        } else {
#endif
        while (batch + 1 < batchLimit) {
            const uint32_t count = qEnd - qBegin;
            const uint32_t groups = balanced ?
                (tail ? (count % SHARED_QUERY_GROUP != 0) : count / SHARED_QUERY_GROUP) :
                (count + SHARED_QUERY_GROUP - 1) / SHARED_QUERY_GROUP;
            if (task < groups) break;
            task -= groups; ++batch; qBegin = qEnd; qEnd = cu.GetValue(batch + 1);
        }
        token = qBegin + (tail ? (qEnd - qBegin) / SHARED_QUERY_GROUP : task) * SHARED_QUERY_GROUP;
#if SHARED_TAIL_BALANCE == 2
        }
#endif
        part = 0; head = 0;
#else
        batch = 0;
        uint32_t qBegin = 0, qEnd = cu.GetValue(1);
#ifdef SHARED_SHORT_TASK_MAP
        if (taskPlan.direct) {
            batch = task; qBegin = cu.GetValue(batch); qEnd = cu.GetValue(batch + 1);
            task = 0;
        } else {
#endif
        while (batch + 1 < p.batch) {
            uint32_t groups = (qEnd - qBegin + SHARED_QUERY_GROUP - 1) / SHARED_QUERY_GROUP;
            if (task < groups) break;
            task -= groups; ++batch; qBegin = qEnd; qEnd = cu.GetValue(batch + 1);
        }
#ifdef SHARED_SHORT_TASK_MAP
        }
#endif
        part = 0; head = 0; token = qBegin + task * SHARED_QUERY_GROUP;
#endif
        uint32_t qCount = qEnd - token < SHARED_QUERY_GROUP ? qEnd - token : SHARED_QUERY_GROUP;
        rows = qCount * SHARED_HEADS(p);
        position = Gm<int32_t>(p.seqLens).GetValue(batch) - qEnd + token;
        firstOri = position + 1 > p.window ? position + 1 - p.window : 0;
        oriCount = position + qCount - firstOri;
        cmpCount = SHARED_RATIO(p) == 1 ? 0 : (position + qCount) / SHARED_RATIO(p);
        if constexpr (SHARED_PACKED_C4) sparseCount = cmpCount;
#else
        const uint32_t headTiles = (SHARED_HEADS(p) + M - 1) / M;
        part = task % splits; task /= splits;
        token = task / headTiles; head = task % headTiles * M;
        rows = SHARED_HEADS(p) - head < M ? SHARED_HEADS(p) - head : M;
        batch = 0;
        while (batch + 1 < p.batch && token >= static_cast<uint32_t>(cu.GetValue(batch + 1))) ++batch;
        position = Gm<int32_t>(p.seqLens).GetValue(batch) - cu.GetValue(batch + 1) + token;
        oriCount = position + 1 < p.window ? position + 1 : p.window;
        firstOri = position + 1 - oriCount;
        cmpCount = SHARED_RATIO(p) == 1 ? 0 : (position + 1) / SHARED_RATIO(p);
        if (!SHARED_SMALL_C4 && p.oriIndices) oriCount = p.oriTopk;
        if constexpr (SHARED_SMALL_C4) sparseCount = cmpCount;
#endif
    }
    __aicore__ uint32_t Count(const SharedKVArgs &p, uint32_t kind) const
    { return kind == 0 ? oriCount : (SHARED_RATIO(p) == 4 ? sparseCount : cmpCount); }
    __aicore__ bool NeedsCompact(const SharedKVArgs &p) const
    { return SHARED_QUERY_GROUP == 1 && !SHARED_SMALL_C4 && SHARED_RATIO(p) == 4 && cmpCount < p.topk && p.topk <= 1024; }
    __aicore__ uint32_t Start(const SharedKVArgs &p, uint32_t kind) const
    {
        uint32_t previous = kind == 0 ? 0 : (oriCount + N - 1) / N;
        return ((part + splitCount - previous % splitCount) % splitCount) * N;
    }
    __aicore__ int32_t Index(const SharedKVArgs &p, uint32_t kind, uint32_t slot) const
    {
        if (slot >= Count(p, kind)) return -1;
        if (kind == 0) return firstOri + slot;
        int32_t index = SHARED_RATIO(p) == 4 ? Gm<int32_t>(p.indices).GetValue(static_cast<uint64_t>(token) * p.topk + slot) : slot;
        return index >= 0 && static_cast<uint32_t>(index) < cmpCount ? index : -1;
    }
};

__aicore__ inline bool PipelineTwoKinds(const SharedKVArgs &p, const Selection &sel, uint32_t splits)
{
    if constexpr (SHARED_SMALL_C4) return sel.cmpCount > 0;
    if constexpr (SHARED_QUERY_GROUP > 1 || SHARED_SMALL_C4)
        return (SHARED_RATIO(p) == 128 || SHARED_PACKED_C4 || SHARED_SMALL_C4) && splits == 1 && !p.oriIndices && sel.oriCount > 0 &&
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

__aicore__ inline uint32_t TaskCount(const SharedKVArgs &p, uint32_t splits SHARED_LIVE_PARAM)
{
#if SHARED_QUERY_GROUP > 1
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
}

#include "cube_pair_pipeline.h"

template<bool Pipeline>
__aicore__ void Cube(const SharedKVArgs &p, uint32_t splits SHARED_LIVE_PARAM SHARED_TASK_PARAM)
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
    uint64_t scratch = p.workspace + static_cast<uint64_t>(GetBlockIdx()) * WORKSPACE_BYTES;
    auto qg = Gm<int8_t>(scratch + Q_OFFSET), pg = Gm<int8_t>(scratch + P_OFFSET);
    auto sg = Gm<int32_t>(scratch + SCORE_OFFSET), vg = Gm<int32_t>(scratch + PV_OFFSET);
#if (defined(SHARED_TAIL_BALANCE) && SHARED_TAIL_BALANCE == 2) || defined(SHARED_SHORT_TASK_MAP)
    const uint32_t tasks = taskPlan.total;
#else
    const uint32_t tasks = TaskCount(p, splits SHARED_LIVE_ARG);
#endif
    for (uint32_t task = GetBlockIdx(); task < tasks; task += GetBlockNum()) {
        Selection sel(p, task, splits SHARED_TASK_ARG);
#ifdef SHARED_SHORT_TASK_MAP
        if (!sel.rows) continue;
#endif
        // Eight-head model shards leave half of the physical M=16 Cube
        // rows unused. Compute the Q/P rounding residual in those rows.
        const uint32_t calcRows = sel.rows + (SHARED_HEADS(p) <= 8 ? RESIDUAL_ROW_OFFSET : 0);
        if (sel.NeedsCompact(p)) {
            CrossCoreWaitFlag(COMPACT_READY);
            auto slots = Gm<uint32_t>(scratch + SLOTS_OFFSET);
            sel.sparseCount = slots.GetValue(0);
            // Retire the scalar DCache line before the next AIV DMA write.
            // See upstream fused_quant_lightning_indexer_manage.cpp:105.
            DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(slots);
            CrossCoreSetFlag<2, PIPE_MTE2>(COMPACT_CONSUMED);
        }
        if constexpr (SHARED_QUERY_GROUP > 1 || SHARED_SMALL_C4) {
            if (PipelineTwoKinds(p, sel, splits)) {
                CubePair<Pipeline>(p, sel, calcRows, scratch, q1, p1, k1, a0, b0, c);
                continue;
            }
        }
        for (uint32_t kind = 0; kind < (SHARED_RATIO(p) == 1 ? 1u : 2u); ++kind) {
            const uint32_t count = sel.Count(p, kind);
            if (sel.Start(p, kind) >= count) continue;
            CrossCoreWaitFlag(Q_READY);
            Nd2NzParams qp{};
            qp.ndNum = 1; qp.nValue = calcRows; qp.dValue = D; qp.srcDValue = D;
            qp.dstNzC0Stride = M; qp.dstNzNStride = 1;
            DataCopy(q1, qg, qp);
            auto cache = Gm<int8_t>(kind == 0 ? p.ori : p.cmp);
            auto table = Gm<int32_t>(kind == 0 ? p.oriTable : p.cmpTable);
            uint32_t block = kind == 0 ? p.oriBlock : p.cmpBlock;
            uint32_t tableWidth = kind == 0 ? p.oriTableWidth : p.cmpTableWidth;
            uint64_t stride = kind == 0 ? p.oriStride : p.cmpStride;
            for (uint32_t start = sel.Start(p, kind); start < count; start += N * splits) {
                // The SWA side of a wide C4 tile usually has only 128 keys.
                // Upstream SCFA sizes Cube work from actualSingleProcessSInnerSize.
                // Retain workspace/softmax strides but omit padded Cube columns.
                const uint32_t remaining = count - start;
                const uint32_t activeN = remaining <= 128 ? 128 : remaining <= 256 ? 256 : N;
                const uint32_t cubeN = SHARED_CUBE_TAILS && activeN < N ? activeN : N;
                // Load each selected token directly into the reusable L1 KV
                // tile. QK and PV consume the same bytes, with different L0B
                // transforms. Invalid slots read a safe token then get masked.
                Nd2NzParams kp{};
                kp.ndNum = 1; kp.nValue = 1; kp.dValue = D; kp.srcDValue = D;
                kp.dstNzC0Stride = N; kp.dstNzNStride = 1;
                if (SHARED_QUERY_GROUP == 1 && !SHARED_SMALL_C4 && ((kind == 1 && SHARED_RATIO(p) == 4) || (kind == 0 && p.oriIndices))) {
                    CrossCoreWaitFlag(KV_READY);
                    kp.nValue = N;
                    DataCopy(k1, Gm<int8_t>(scratch + KV_OFFSET), kp);
                } else {
                    // Zero padding is local to L1 and needs no invalid GM reads.
                    // dav_c220 Fill does not accept int8; zero the same bytes via
                    // its supported uint32 view (blockNum is measured in 32 B).
                    InitConstValue(k1.ReinterpretCast<uint32_t>(),
                                   InitConstValueParams<uint32_t>{1, N * D / 32, 0, 0});
                    PipeBarrier<PIPE_MTE2>();
                    uint32_t valid = count - start < N ? count - start : N;
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
                    if (cubeN == N) {
                        lk.repeatTimes = (N / 16) * (qkK / 32);
                        LoadData(bk, k1[k * N], lk);
                    } else {
                        // Each L1 channel plane keeps the original N stride;
                        // compact its live rows into contiguous L0B planes.
                        lk.repeatTimes = cubeN / 16;
                        for (uint32_t channel = 0; channel < qkK; channel += 32)
                            LoadData(bk[channel * cubeN], k1[(k + channel) * N], lk);
                    }
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
                    LoadData2dTransposeParams lp{};
                    lp.repeatTimes = pvK / 32; lp.srcStride = 1; lp.dstGap = D / 16 - 1; lp.dstFracGap = 0;
                    for (uint32_t n = 0; n < D / 32; ++n)
                        LoadDataWithTranspose(bk[n * 1024], k1[n * N * 32 + k * 32], lp);
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

#include "sparse_gather.h"

#include "vector_batch.h"

__aicore__ void ReduceParts(const SharedKVArgs &p, uint32_t splits SHARED_LIVE_PARAM)
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

__aicore__ void ClearPaddedRows(const SharedKVArgs &p SHARED_LIVE_PARAM)
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

__aicore__ void RunWithSplits(const SharedKVArgs &p, uint32_t splits SHARED_LIVE_PARAM)
{
#if defined(SHARED_TAIL_BALANCE) && SHARED_TAIL_BALANCE == 2
    const auto taskPlan=MakeTailTaskPlan(p);
#elif defined(SHARED_SHORT_TASK_MAP)
    const auto taskPlan=MakeShortTaskPlan(p);
#endif
    // TND tensors may have graph-capacity padding. Upstream determines work
    // from cumulative actual query lengths, not the tensor's first dimension.
    // Keep the by-value launch argument immutable: ASC can address it in
    // captured parameter storage, which is reused by subsequent graph replays.
    if ASCEND_IS_AIC {
        const bool denseLong = SHARED_DENSE_L0_PIPELINE && SHARED_RATIO(p) == 128 &&
            p.batch && Gm<int32_t>(p.seqLens).GetValue(0) > 32768;
        if (!denseLong && (SHARED_RATIO(p) == 4 || SHARED_QUERY_GROUP == 1) && SHARED_LIVE_TOKENS <= 48) Cube<false>(p, splits SHARED_LIVE_ARG SHARED_TASK_ARG);
        else Cube<true>(p, splits SHARED_LIVE_ARG SHARED_TASK_ARG);
    }
    if ASCEND_IS_AIV { VectorBatched(p, splits SHARED_LIVE_ARG SHARED_TASK_ARG); }
    if (splits > 1) {
        // All active blocks reach the same single global barrier. No block
        // spins waiting for a block that has not been scheduled.
        SyncAll<false>();
        if ASCEND_IS_AIV { ReduceParts(p, splits SHARED_LIVE_ARG); }
    }
    if ASCEND_IS_AIV { ClearPaddedRows(p SHARED_LIVE_ARG); }
}

__aicore__ void Run(const SharedKVArgs &p SHARED_LIVE_PARAM)
{
    uint32_t splits = p.splits;
    if (SHARED_ADAPTIVE_DENSE && p.partial && splits == 1 && SHARED_RATIO(p) == 128 && SHARED_HEADS(p) == 8) {
        const uint32_t actual = SHARED_LIVE_TOKENS;
        if (actual && actual <= 12) {
            const uint32_t batch = SHARED_LIVE_BATCH;
            uint32_t maximum = 0;
            auto lengths = Gm<int32_t>(p.seqLens);
            for (uint32_t b = 0; b < batch; ++b) {
                const uint32_t value = lengths.GetValue(b);
                maximum = value > maximum ? value : maximum;
            }
            if (maximum > 32768) {
                const uint32_t tiles = 1 + (maximum / 128 + N - 1) / N;
                const uint32_t preferred = actual <= 6 ? 4 : 2;
                splits = preferred < tiles ? preferred : tiles;
            }
        }
    }
    if (SHARED_ADAPTIVE_SPLITS && p.partial && splits == 1 && SHARED_RATIO(p) == 4 && SHARED_HEADS(p) <= 8 && p.topk <= 1024) {
        const uint32_t actual = SHARED_LIVE_TOKENS;
        if (actual > 0 && actual <= 12 &&
            (actual <= 8 || Gm<int32_t>(p.seqLens).GetValue(0) > 4096)) {
            const uint32_t limit = 1 + (p.topk + N - 1) / N;
            const uint32_t preferred = actual <= 8 && Gm<int32_t>(p.seqLens).GetValue(0) > 4096 ? 3 : 2;
            splits = preferred < limit ? preferred : limit;
        }
    }
    RunWithSplits(p, splits SHARED_LIVE_ARG);
}

#if SHARED_CACHE_LIVE
__aicore__ void Run(const SharedKVArgs &p)
{
    Run(p, ReadLiveShape(p));
}
#endif

} // internal device functions
} // namespace SHARED_KERNEL_NAMESPACE

#ifndef SHARED_IMPLEMENTATION_ONLY
extern "C" __global__ __aicore__ void sharedkv_i8_fused(SharedKVArgs p)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    SHARED_KERNEL_NAMESPACE::Run(p);
}

extern "C" void launch_sharedkv(uint32_t blocks, void *stream, const SharedKVArgs *args)
{
    sharedkv_i8_fused<<<blocks, nullptr, stream>>>(*args);
}

#endif // SHARED_IMPLEMENTATION_ONLY
