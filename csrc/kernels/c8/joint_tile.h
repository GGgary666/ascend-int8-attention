// A short C128 task has two independently scaled KV tiles. Keep both in L1,
// concatenate their dequantized scores, and run one softmax/P encoding phase.
// QK/PV remain separate INT8 products with Q/P high+low residual rows.
// Reference: upstream SCFA PreloadPipeline; the GM<->UB strided copies and
// non-basic-block FP32 SoftmaxFlashV2 are verified in CANN9.2 dav_c220.
__aicore__ inline bool JointTileVector(const SharedKVArgs &p, const Selection &sel,
    uint32_t begin, uint32_t nr, uint64_t scratch, LocalTensor<float> x,
    LocalTensor<float> residual, LocalTensor<float> r, LocalTensor<float> brc,
    LocalTensor<half> h, LocalTensor<int8_t> i8, LocalTensor<float> mask,
    LocalTensor<float> indices, LocalTensor<float> scales, LocalTensor<float> acc, LocalTensor<float> maxBuf,
    LocalTensor<float> sumBuf, LocalTensor<float> expBuf, LocalTensor<uint8_t> softTmp,
    uint32_t nextTask, uint32_t tasks, uint32_t taskOrder, uint32_t uniformQueries,
    const QueryStageBuffers &stageBuffers)
{
    bool staged = false;
    if constexpr (JOINT_DENSE_TILE) {
        static_assert(!JOINT_DENSE_TILE || 2 * N <= D, "Joined scores must fit the FP32 row capacity");
        CrossCoreWaitFlag(SCORE_READY);
        CrossCoreWaitFlag(SCORE_READY_SECOND);
        DataCopyExtParams cp{};
        cp.blockCount = nr; cp.blockLen = N * 4; cp.dstStride = N * 4 / 32;
        for (uint32_t kind = 0; kind < 2; ++kind) {
            auto src = Gm<int32_t>(scratch + (kind ? SCORE_SECOND_OFFSET : SCORE_OFFSET));
            DataCopyPad(x[kind * N].ReinterpretCast<int32_t>(), src[begin * N], cp,
                        DataCopyPadExtParams<int32_t>{});
            DataCopyPad(residual[kind * N].ReinterpretCast<int32_t>(),
                        src[(begin + RESIDUAL_ROW_OFFSET) * N], cp, DataCopyPadExtParams<int32_t>{});
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(x, x.ReinterpretCast<int32_t>(), RoundMode::CAST_RINT, nr * 2 * N);
        PipeBarrier<PIPE_V>();
        AddResidualProduct(x, residual.ReinterpretCast<int32_t>(), nr * 2 * N);
        if constexpr (SHARED_QUERY_GROUP == 1) {
            // r[32:40] retains the two independent four-head Q scales.
            Muls(r[R], r[32], p.softmaxScale, 2 * nr); PipeBarrier<PIPE_V>();
            Brcb(brc, r[R], R / 8, {1, 8}); PipeBarrier<PIPE_V>();
        }
        BinaryRepeatParams rp{};
        rp.dstBlkStride = rp.src0BlkStride = 1; rp.src1BlkStride = 0;
        rp.dstRepStride = rp.src0RepStride = 2 * N / 8; rp.src1RepStride = 1;
        for (uint32_t kind = 0; kind < 2; ++kind) {
            if constexpr (SHARED_QUERY_GROUP > 1) {
                Muls(r[R], kind ? r[32] : maxBuf[R], p.softmaxScale, nr); PipeBarrier<PIPE_V>();
                Brcb(brc, r[R], R / 8, {1, 8}); PipeBarrier<PIPE_V>();
            }
            for (uint32_t col = 0; col < N; col += 64)
                Mul(x[kind * N + col], x[kind * N + col],
                    brc[SHARED_QUERY_GROUP == 1 ? kind * nr * 8 : 0], 64, nr, rp);
            PipeBarrier<PIPE_V>();
          if constexpr (SHARED_QUERY_GROUP == 1) {
            uint32_t valid = kind ? sel.cmpCount : sel.oriCount;
            Duplicate(mask, 0.0f, N); PipeBarrier<PIPE_V>();
            const uint32_t aligned = (valid + 7) / 8 * 8;
            if (valid != aligned) {
                VectorToScalar();
                for (uint32_t col = valid; col < aligned; ++col) mask.SetValue(col, -1e30f);
                SetFlag<HardEvent::S_V>(EVENT_ID0); WaitFlag<HardEvent::S_V>(EVENT_ID0);
            }
            if (aligned < N) Duplicate(mask[aligned], -1e30f, N - aligned);
            PipeBarrier<PIPE_V>();
            for (uint32_t row = 0; row < nr; ++row)
                Add(x[row * 2 * N + kind * N], x[row * 2 * N + kind * N], mask, N);
            PipeBarrier<PIPE_V>();
          } else {
            // A group shares KV but each query keeps its own causal window.
            // Low-product scratch is dead after the residual merge above.
            for (uint32_t row = 0; row < nr; ++row) {
                if (!row || (begin + row) % SHARED_HEADS(p) == 0) {
                    const int32_t pos = sel.position + (begin + row) / SHARED_HEADS(p);
                    const int32_t low = kind ? 0 : pos + 1 - static_cast<int32_t>(WINDOW_ROWS + sel.firstOri);
                    const int32_t high = kind ? (pos + 1) / static_cast<int32_t>(SHARED_RATIO(p))
                                              : pos + 1 - static_cast<int32_t>(sel.firstOri);
                    auto bits = residual.ReinterpretCast<uint8_t>();
                    CompareScalar(bits, indices, static_cast<float>(low), CMPMODE::GE, N); PipeBarrier<PIPE_V>();
                    CompareScalar(bits[64], indices, static_cast<float>(high), CMPMODE::LT, N); PipeBarrier<PIPE_V>();
                    And(bits.ReinterpretCast<uint16_t>(), bits.ReinterpretCast<uint16_t>(),
                        bits[64].ReinterpretCast<uint16_t>(), N / 16); PipeBarrier<PIPE_V>();
                    Duplicate(mask, 0.0f, N); PipeBarrier<PIPE_V>();
                    Select(mask, bits, mask, -1e30f, SELMODE::VSEL_TENSOR_SCALAR_MODE, N); PipeBarrier<PIPE_V>();
                }
                Add(x[row * 2 * N + kind * N], x[row * 2 * N + kind * N], mask, N);
            }
            PipeBarrier<PIPE_V>();
          }
        }
        static_assert((R * 64 + R * 8) * sizeof(float) <= 32768);
        const SoftmaxBuffers softmaxBuffers{
            static_cast<uint32_t>(x.GetPhyAddr()), static_cast<uint32_t>(sumBuf.GetPhyAddr()),
            static_cast<uint32_t>(maxBuf.GetPhyAddr()), static_cast<uint32_t>(expBuf.GetPhyAddr()),
            static_cast<uint32_t>(softTmp.GetPhyAddr())};
        CompactSoftmax<R, R>(nr, 2 * N, softmaxBuffers);
        PipeBarrier<PIPE_V>();
        Muls(x, x, 127.0f, nr * 2 * N); PipeBarrier<PIPE_V>();
        RoundResidualInt8(i8, i8[R * D], h, x, residual, nr * 2 * N);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0); WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        cp.blockLen = N; cp.srcStride = N / 32; cp.dstStride = 0;
        for (uint32_t kind = 0; kind < 2; ++kind) {
            auto dst = Gm<int8_t>(scratch + (kind ? P_SECOND_OFFSET : P_OFFSET));
            DataCopyPad(dst[begin * N], i8[kind * N], cp);
            DataCopyPad(dst[(begin + RESIDUAL_ROW_OFFSET) * N], i8[R * D + kind * N], cp);
        }
        PipeBarrier<PIPE_ALL>();
        // Distinct flags retain both notifications while Cube consumes P0.
        CrossCoreSetFlag<2, PIPE_MTE3>(P_READY);
        CrossCoreSetFlag<2, PIPE_MTE3>(P_READY_SECOND);
        if constexpr (SHARED_QUERY_GROUP > 1) {
            // P DMA has retired. Separate Q slots and dead softmax scratch
            // can prepare the next task while Cube runs both current PVs.
            if (nextTask != 0xffffffffu)
                staged = StageGroupedQuery(p, nextTask, tasks, taskOrder, uniformQueries, scratch, stageBuffers);
            for (uint32_t kind = 0; kind < 2; ++kind) {
                CrossCoreWaitFlag(kind ? PV_READY_SECOND : PV_READY);
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0); WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                auto src = Gm<int32_t>(scratch + (kind ? PV_SECOND_OFFSET : PV_OFFSET));
                DataCopy(x.ReinterpretCast<int32_t>(), src[begin * D], nr * D);
                DataCopy(residual.ReinterpretCast<int32_t>(), src[(begin + RESIDUAL_ROW_OFFSET) * D], nr * D);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                Cast(x, x.ReinterpretCast<int32_t>(), RoundMode::CAST_RINT, nr * D); PipeBarrier<PIPE_V>();
                AddResidualProduct(x, residual.ReinterpretCast<int32_t>(), nr * D);
                AccumulateChannels(acc, x, scales[(2 + kind) * D], nr);
                PipeBarrier<PIPE_ALL>();
            }
        } else {
        CrossCoreWaitFlag(PV_READY);
        CrossCoreWaitFlag(PV_READY_SECOND);
        for (uint32_t kind = 0; kind < 2; ++kind) {
            auto src = Gm<int32_t>(scratch + (kind ? PV_SECOND_OFFSET : PV_OFFSET));
            DataCopy(x[kind * nr * D].ReinterpretCast<int32_t>(), src[begin * D], nr * D);
            DataCopy(residual[kind * nr * D].ReinterpretCast<int32_t>(),
                     src[(begin + RESIDUAL_ROW_OFFSET) * D], nr * D);
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(x, x.ReinterpretCast<int32_t>(), RoundMode::CAST_RINT, 2 * nr * D);
        PipeBarrier<PIPE_V>();
        AddResidualProduct(x, residual.ReinterpretCast<int32_t>(), 2 * nr * D);
        AccumulateChannels(acc, x, scales[2 * D], nr); PipeBarrier<PIPE_V>();
        AccumulateChannels(acc, x[nr * D], scales[3 * D], nr);
        PipeBarrier<PIPE_ALL>();
        }
    }
    return staged;
}
