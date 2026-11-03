// C8 production core: frozen-channel KV, Q/P double residuals.
// Private grouped C128 / short dense C4 pipeline. Reference: vllm-ascend a406c2ea7bf11,
// sparse_attn_sharedkv_scfa_kernel.h:790 PreloadPipeline. QK1 overlaps
// softmax0; both L1 KV tiles live until their PV stage. Online softmax and
// high+residual INT8 math retain the original order and representation.
// Included after Selection/LoadA/StoreAcc inside the implementation namespace.
template<bool Pipeline, bool Joint = false>
__aicore__ inline void CubePair(const SharedKVArgs &p, const Selection &sel,
    uint32_t calcRows, uint64_t scratch, LocalTensor<int8_t> q1,
    LocalTensor<int8_t> p1, LocalTensor<int8_t> kv, LocalTensor<int8_t> a0,
    LocalTensor<int8_t> b0, LocalTensor<int32_t> c)
{
    for (uint32_t kind = 0; kind < 2; ++kind) {
        auto k1 = kv[kind * N * D];
        auto cache = Gm<int8_t>(kind ? p.cmp : p.ori);
        auto table = Gm<int32_t>(kind ? p.cmpTable : p.oriTable);
        uint32_t block = kind ? BLOCK_ROWS : BLOCK_ROWS;
        uint32_t width = kind ? p.cmpTableWidth : p.oriTableWidth;
        uint64_t stride = kind ? p.cmpStride : p.oriStride;
        uint32_t count = kind ? sel.cmpCount : sel.oriCount;
        uint32_t cubeN = count <= 128 ? 128 : N;
        ZeroKvTail(k1, count, cubeN, cubeN);
        Nd2NzParams kp{};
        kp.ndNum = 1; kp.dValue = D; kp.srcDValue = D;
        kp.dstNzC0Stride = cubeN; kp.dstNzNStride = 1;
        for (uint32_t n = 0; n < count;) {
            uint32_t idx = (kind ? 0 : sel.firstOri) + n;
            uint32_t run = block - idx % block;
            run = run < count - n ? run : count - n;
            kp.nValue = run;
            uint32_t page = table.GetValue(sel.batch * width + idx / block);
            DataCopy(k1[n * 32], cache[page * stride + static_cast<uint64_t>(idx % block) * D], kp);
            n += run;
        }
    }
    // Both L1 tiles are retained through PV. Stage them while Vector loads
    // and encodes Q, rather than serializing their DMA behind Q-ready.
    for (uint32_t kind = 0; kind < 2; ++kind) {
        CrossCoreWaitFlag(kind ? Q_READY_SECOND : Q_READY);
        Nd2NzParams qp{};
        qp.ndNum = 1; qp.nValue = calcRows; qp.dValue = D; qp.srcDValue = D;
        qp.dstNzC0Stride = M; qp.dstNzNStride = 1;
        DataCopy(q1, Gm<int8_t>(scratch + (kind ? Q_SECOND_OFFSET : Q_OFFSET)), qp);
        auto k1 = kv[kind * N * D];
        uint32_t count = kind ? sel.cmpCount : sel.oriCount;
        uint32_t cubeN = count <= 128 ? 128 : N;
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0); WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        MmadParams mm{};
        // SCFA uses two 32 KiB L0B slots. A slot-specific M->MTE1
        // event protects reuse while the next slot loads in parallel.
        const uint32_t capacityK = ((Pipeline ? 32768 : 65536) / cubeN / 32) * 32;
        constexpr uint32_t maxK = Pipeline ? D / 2 : D;
        const uint32_t qkK = capacityK < maxK ? capacityK : maxK;
        if constexpr (Pipeline) for (uint32_t slot = 0; slot < 2; ++slot)
            SetFlag<HardEvent::M_MTE1>(EVENT_ID3 + slot);
        for (uint32_t k = 0; k < D; k += qkK) {
            const uint32_t liveK = D - k < qkK ? D - k : qkK;
            const uint32_t slot = Pipeline ? (k / qkK) % 2 : 0;
            const uint32_t event = Pipeline ? EVENT_ID3 + slot : EVENT_ID0;
            auto ak = a0[slot * M * qkK], bk = b0[slot * 32768];
            if constexpr (Pipeline) WaitFlag<HardEvent::M_MTE1>(event);
            LoadA(ak, q1[k * M], liveK);
            LoadData2DParams lk{};
            lk.srcStride = 1;
            // Each retained tile uses its actual32-row-aligned NZ pitch.
            // A contiguous LoadData replaces per-channel plane copies.
            lk.repeatTimes = (cubeN / 16) * (liveK / 32);
            LoadData(bk, k1[k * cubeN], lk);
            SetFlag<HardEvent::MTE1_M>(event); WaitFlag<HardEvent::MTE1_M>(event);
            mm.m = M; mm.n = cubeN; mm.k = liveK; mm.cmatrixInitVal = k == 0;
            Mmad(c, ak, bk, mm);
            if constexpr (Pipeline) SetFlag<HardEvent::M_MTE1>(event);
            else if constexpr (N > 128) PipeBarrier<PIPE_ALL>();
        }
        if constexpr (Pipeline) for (uint32_t slot = 0; slot < 2; ++slot)
            WaitFlag<HardEvent::M_MTE1>(EVENT_ID3 + slot);
        StoreAcc(Gm<int32_t>(scratch + (kind ? SCORE_SECOND_OFFSET : SCORE_OFFSET)), c, calcRows, cubeN, N);
        CrossCoreSetFlag<2, PIPE_FIX>(kind ? SCORE_READY_SECOND : SCORE_READY);
        PipeBarrier<PIPE_ALL>();
    }
    for (uint32_t kind = 0; kind < 2; ++kind) {
        auto k1 = kv[kind * N * D];
        uint32_t count = kind ? sel.cmpCount : sel.oriCount;
        uint32_t cubeN = count <= 128 ? 128 : N;
        CrossCoreWaitFlag(Joint && kind ? P_READY_SECOND : P_READY);
        PipeBarrier<PIPE_ALL>();
        Nd2NzParams pp{};
        pp.ndNum = 1; pp.nValue = calcRows;
        pp.dstNzC0Stride = M; pp.dstNzNStride = 1;
        MmadParams mm{}; mm.m = M;
        pp.dValue = cubeN; pp.srcDValue = N;
        DataCopy(p1, Gm<int8_t>(scratch + (Joint && kind ? P_SECOND_OFFSET : P_OFFSET)), pp);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0); WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        if constexpr (Pipeline) for (uint32_t slot = 0; slot < 2; ++slot)
            SetFlag<HardEvent::M_MTE1>(EVENT_ID3 + slot);
        constexpr uint32_t pvK = Pipeline ? 64 : 128;
        for (uint32_t k = 0; k < cubeN; k += pvK) {
            const uint32_t liveK = cubeN - k < pvK ? cubeN - k : pvK;
            const uint32_t slot = Pipeline ? (k / pvK) % 2 : 0;
            const uint32_t event = Pipeline ? EVENT_ID3 + slot : EVENT_ID0;
            auto ak = a0[slot * M * pvK], bk = b0[slot * 32768];
            if constexpr (Pipeline) WaitFlag<HardEvent::M_MTE1>(event);
            LoadA(ak, p1[k * M], liveK);
            LoadValueB(bk, k1, cubeN, k, liveK);
            SetFlag<HardEvent::MTE1_M>(event); WaitFlag<HardEvent::MTE1_M>(event);
            mm.n = D; mm.k = liveK; mm.cmatrixInitVal = k == 0;
            Mmad(c, ak, bk, mm);
            if constexpr (Pipeline) SetFlag<HardEvent::M_MTE1>(event);
            else if constexpr (N > 128) PipeBarrier<PIPE_ALL>();
        }
        if constexpr (Pipeline) for (uint32_t slot = 0; slot < 2; ++slot)
            WaitFlag<HardEvent::M_MTE1>(EVENT_ID3 + slot);
        StoreAcc(Gm<int32_t>(scratch + (Joint && kind ? PV_SECOND_OFFSET : PV_OFFSET)), c, calcRows, D, D);
        PipeBarrier<PIPE_ALL>();
        CrossCoreSetFlag<2, PIPE_FIX>(Joint && kind ? PV_READY_SECOND : PV_READY);
    }
}
