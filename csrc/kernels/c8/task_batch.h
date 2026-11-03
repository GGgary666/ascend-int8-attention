// Several independent G1 tasks share Vector instruction setup and phase
// notifications. Cube QK/PV and softmax rows remain request-independent.
// Included in short C4/C128 namespaces, after the common primitives.
// Reference: vLLM SCFA PreloadPipeline's retained task metadata/KV lifetime;
// CANN9.2 dav_c220 INT8 Mmad, ND2NZ and transpose-load contracts.
constexpr uint32_t TB_CAP = sharedkv_c8::kRetainedTasks, TB_ROWS = 4 * TB_CAP;
constexpr uint32_t TB_SCALAR_ROWS = (TB_ROWS + 7) / 8 * 8;
constexpr uint32_t TB_MAX_N = 512, TB_SOFTMAX_BYTES = 24 * 1024;
// Keep the GM slot budget separate from the live column extent. Wider
// dense tasks retain fewer queries, so their total score/P footprint fits
// the same slots. C4 membership uses the same bounded key domain.
constexpr uint32_t TB_COLUMN_CAP = sharedkv_c8::kRetainedKeys;
constexpr uint32_t TB_MASK_BYTES = (TB_COLUMN_CAP + 255) / 256 * 32;
// Phase-major GM layout [task, matrix row, column]. Each AIV owns a
// contiguous half of the live query/head rows, sharing whole-query setup.
constexpr uint32_t TB_S0 = 2 * TB_CAP * M * D, TB_S1 = TB_S0 + TB_CAP * M * TB_MAX_N * 4;
// PV reuses scores only after P_READY. Q occupies separate storage so the
// next batch can be encoded while the preceding PV is still in flight.
constexpr uint32_t TB_P0 = 0, TB_P1 = TB_CAP * M * TB_MAX_N;
constexpr uint32_t TB_V0 = 2 * TB_CAP * M * TB_MAX_N, TB_V1 = TB_V0 + TB_CAP * M * D * 4;
constexpr uint32_t TB_PHASE_BYTES = TB_V1 + TB_CAP * M * D * 4;
constexpr uint32_t TB_Q0 = TB_PHASE_BYTES, TB_Q1 = TB_Q0 + TB_CAP * M * D;
constexpr uint32_t TB_WORKSPACE = TB_Q1 + TB_CAP * M * D;
static_assert(TB_S1 + TB_CAP * M * TB_MAX_N * 4 <= TB_PHASE_BYTES);
static_assert(TB_WORKSPACE <= WORKSPACE_CORE_BYTES, "Task batch exceeds host scratch");
static_assert(M == sharedkv_c8::kRetainedMatrixRows && D == sharedkv_c8::kRetainedChannels);
static_assert(M * D + (D + M) * TB_COLUMN_CAP <= sharedkv_c8::kRetainedL1Bytes);
static_assert(M * TB_COLUMN_CAP * 4 <= 65536, "Retained QK must fit CO1");
static_assert(TB_ROWS * D >= 4 * 2 * 1024, "Joint scores must fit the reserved UB rows");
static_assert(TB_SOFTMAX_BYTES >= TB_ROWS * (8 + 64) * sizeof(float));
static_assert(TB_SOFTMAX_BYTES >= TB_ROWS * D * sizeof(half));
static_assert(TB_CAP * TB_COLUMN_CAP * sizeof(float) <= TB_ROWS * D * 2,
              "C4 membership must fit the retired BF16 query buffer");
static_assert(TB_MASK_BYTES >= (TB_ROWS - 4) * sizeof(uint32_t),
              "Retired predicate storage must fit LSE gather offsets");

__aicore__ inline sharedkv_c8::TaskRange RetainedRange(const SharedKVArgs &p, uint32_t tasks,
    uint32_t uniformQueries, uint32_t width, uint32_t taskOrder)
{
    if constexpr (C8_RATIO == 128)
        if (taskOrder) return sharedkv_c8::PairedTaskRange(taskOrder);
    uint32_t core = GetBlockIdx(), partner = 0;
    if ASCEND_IS_AIV { partner = core % 2; core /= 2; }
    bool contiguous = false;
    // This whole predicate proves that this launch executed the metadata
    // producer. In particular a clamped cuQ end can give uniformQueries=1
    // while tasks<live batch; it must never read an old graph's hint.
    if (width == 1 &&
        tasks >= sharedkv_c8::kMetadataVectorMinimum && uniformQueries == 1 && tasks == ActualBatch(p)) {
        auto record = Gm<uint32_t>(p.partial + uint64_t(core) * 512 + partner * 64);
        DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(record);
        contiguous = record.GetValue(4) != 0;
    }
    return sharedkv_c8::PartitionTasks(tasks, core, GetBlockNum(), contiguous);
}

__aicore__ __attribute__((noinline, section(".text.sharedkv_task_batch"))) void TaskBatchCube(const SharedKVArgs &p, uint32_t tasks,
                             uint32_t uniformQueries, uint32_t oriRows, uint32_t cmpRows, uint32_t taskOrder)
{
    TPipe pipe;
    // c220 TPipe pre-sets M_MTE1 IDs 0/1/2 for matmul internals. An owned
    // event must not consume those initial credits in place of MMAD completion.
    const auto l0Consumed = pipe.AllocEventID<HardEvent::M_MTE1>();
    const auto resources = sharedkv_c8::PlanRetainedRows(oriRows, cmpRows);
    const uint32_t width = resources.width, scoreCols = resources.scoreColumns;
    const uint32_t qBytes = width * M * D, pBytes = width * M * scoreCols;
    const uint32_t kvBytes = width * resources.residentRows * D;
    TBuf<TPosition::A1> l1;
    TBuf<TPosition::A2> ab;
    TBuf<TPosition::B2> bb;
    TBuf<TPosition::CO1> cb;
    pipe.InitBuffer(l1, qBytes + pBytes + kvBytes); pipe.InitBuffer(ab, M * D);
    pipe.InitBuffer(bb, 65536); pipe.InitBuffer(cb, M * TB_COLUMN_CAP * 4);
    const auto l1Base = static_cast<uint32_t>(l1.Get<int8_t>().GetPhyAddr());
    LocalTensor<int8_t> q(TPosition::A1, l1Base, qBytes);
    LocalTensor<int8_t> prob(TPosition::A1, l1Base + qBytes, pBytes);
    LocalTensor<int8_t> kv(TPosition::B1, l1Base + qBytes + pBytes, kvBytes);
    auto a = ab.Get<int8_t>(), b = bb.Get<int8_t>(); auto c = cb.Get<int32_t>();
    const uint32_t cores = GetBlockNum(), core = GetBlockIdx();
    const bool reuseKvSlot = resources.reuseKvSlot;
    const uint32_t kvStride = resources.residentRows * D;
    const uint32_t compressedOffset = reuseKvSlot ? 0 : oriRows * D;
    const uint64_t scratch = p.workspace + static_cast<uint64_t>(core) * WORKSPACE_CORE_BYTES;
    const auto range = RetainedRange(p, tasks, uniformQueries, width, taskOrder);
    for (uint32_t base = 0; base < range.count; base += width) {
        const uint32_t tail = range.count - base;
        const uint32_t count = tail < width ? tail : width;
        uint32_t batches[TB_CAP], first[TB_CAP], oriCount[TB_CAP], cmpCount[TB_CAP];
        uint32_t activeOri = 0, activeCmp = 0;
        for (uint32_t t = 0; t < count; ++t) {
            Selection sel(p, range.first + (base + t) * range.stride, 1, uniformQueries);
            batches[t] = sel.batch; first[t] = sel.firstOri;
            oriCount[t] = sel.oriCount; cmpCount[t] = sel.cmpCount;
            activeOri = sel.oriCount > activeOri ? sel.oriCount : activeOri;
            activeCmp = sel.cmpCount > activeCmp ? sel.cmpCount : activeCmp;
        }
        const uint32_t activeOriRows = (activeOri + 31) / 32 * 32;
        // For uniform q1 the query index is also its request index. Both
        // AIVs can check the same sequence length without a request search.
        // General query indexing retains the common two-kind protocol.
        const uint32_t kinds = cmpRows &&
            (width > 1 || uniformQueries != 1 || activeCmp) ? 2 : 1;
        const uint32_t activeCmpRows = activeCmp ? (activeCmp + 31) / 32 * 32 : 32;
        // Co-resident kinds retain the existing preload-both/QK-both order.
        // If only one kind fits, each load is followed by its QK before the
        // same slot is reused. The QK release below closes the L1 read.
        const uint32_t stages = reuseKvSlot ? kinds : 2 * kinds;
        for (uint32_t stage = 0; stage < stages; ++stage) {
            const uint32_t kind = stage < kinds ? stage : stage - kinds;
            const uint32_t columns = kind ? activeCmpRows : activeOriRows;
          if (reuseKvSlot || stage < kinds) {
            auto cache = Gm<int8_t>(kind ? p.cmp : p.ori);
            auto table = Gm<int32_t>(kind ? p.cmpTable : p.oriTable);
            const uint32_t tableWidth = kind ? p.cmpTableWidth : p.oriTableWidth;
            const uint64_t stride = kind ? p.cmpStride : p.oriStride;
            for (uint32_t t = 0; t < count; ++t) {
                auto k = kv[t * kvStride + (kind ? compressedOffset : 0)];
                const uint32_t valid = kind ? cmpCount[t] : oriCount[t];
                ZeroKvTail(k, valid, columns, columns);
                Nd2NzParams kp{};
                kp.ndNum = 1; kp.dValue = D; kp.srcDValue = D;
                kp.dstNzC0Stride = columns; kp.dstNzNStride = 1;
                for (uint32_t n = 0; n < valid;) {
                    const uint32_t idx = (kind ? 0 : first[t]) + n;
                    uint32_t run = 32 - idx % 32; run = run < valid - n ? run : valid - n;
                    kp.nValue = run;
                    const uint32_t page = table.GetValue(batches[t] * tableWidth + idx / 32);
                    DataCopy(k[n * 32], cache[page * stride + static_cast<uint64_t>(idx % 32) * D], kp);
                    n += run;
                }
            }
          }
          if (reuseKvSlot || stage >= kinds) {
            CrossCoreWaitFlag(kind ? Q_READY_SECOND : Q_READY);
            Nd2NzParams qp{};
            qp.ndNum = count; qp.nValue = M; qp.dValue = D; qp.srcDValue = D;
            qp.srcNdMatrixStride = qp.dstNzMatrixStride = M * D;
            qp.dstNzC0Stride = M; qp.dstNzNStride = 1;
            DataCopy(q, Gm<int8_t>(scratch + (kind ? TB_Q1 : TB_Q0)), qp);
            SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0); WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
            for (uint32_t t = 0; t < count; ++t) {
                auto k = kv[t * kvStride + (kind ? compressedOffset : 0)];
                uint32_t tileK = (65536 / columns / 32) * 32;
                tileK = tileK < D ? tileK : D;
                for (uint32_t kk = 0; kk < D; kk += tileK) {
                    const uint32_t liveK = tileK < D - kk ? tileK : D - kk;
                    LoadA(a, q[t * M * D + kk * M], liveK);
                    LoadData2DParams lp{}; lp.srcStride = 1;
                    lp.repeatTimes = (columns / 16) * (liveK / 32);
                    LoadData(b, k[kk * columns], lp);
                    SetFlag<HardEvent::MTE1_M>(EVENT_ID0); WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
                    MmadParams mm{}; mm.m = M; mm.n = columns; mm.k = liveK; mm.cmatrixInitVal = kk == 0;
                    Mmad(c, a, b, mm);
                    SetFlag<HardEvent::M_MTE1>(l0Consumed); WaitFlag<HardEvent::M_MTE1>(l0Consumed);
                }
                StoreAcc(Gm<int32_t>(scratch + (kind ? TB_S1 : TB_S0))[t * M * scoreCols], c, M, columns, scoreCols);
            }
            // Release Q's L1 read before the next kind reloads it. StoreAcc
            // separately protects L0C, allowing its FIX output to overlap
            // the next task's L1->L0 transfer without a global barrier.
            SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0); WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
            CrossCoreSetFlag<2, PIPE_FIX>(kind ? SCORE_READY_SECOND : SCORE_READY);
          }
        }
        for (uint32_t phase = 0; phase < kinds; ++phase) {
            // Reuse the compressed kind left by QK first. Vector still
            // waits for original PV then compressed PV, preserving its
            // numeric order and the next-Q release after both GM reads.
            const uint32_t kind = reuseKvSlot ? kinds - 1 - phase : phase;
            if (reuseKvSlot && kinds == 2 && kind == 0) {
                // The preceding PV released every compressed L1 reader.
                // Width is one here; reload only the <=128-row window.
                ZeroKvTail(kv, oriCount[0], activeOriRows, activeOriRows);
                Nd2NzParams kp{}; kp.ndNum = 1; kp.dValue = D; kp.srcDValue = D;
                kp.dstNzC0Stride = activeOriRows; kp.dstNzNStride = 1;
                auto table = Gm<int32_t>(p.oriTable);
                for (uint32_t n = 0; n < oriCount[0];) {
                    const uint32_t idx = first[0] + n;
                    uint32_t run = 32 - idx % 32; run = run < oriCount[0] - n ? run : oriCount[0] - n;
                    kp.nValue = run;
                    const uint32_t page = table.GetValue(batches[0] * p.oriTableWidth + idx / 32);
                    DataCopy(kv[n * 32], Gm<int8_t>(p.ori)[page * p.oriStride + uint64_t(idx % 32) * D], kp);
                    n += run;
                }
            }
            CrossCoreWaitFlag(kind ? P_READY_SECOND : P_READY);
            const uint32_t columns = kind ? activeCmpRows : activeOriRows;
            Nd2NzParams pp{}; pp.ndNum = count; pp.nValue = M;
            // GM keeps its fixed score pitch, but PV consumes only this
            // batch's live columns. Pack those columns contiguously in L1;
            // kind release below protects the next compact P overwrite.
            pp.dValue = columns; pp.srcDValue = scoreCols; pp.dstNzC0Stride = M; pp.dstNzNStride = 1;
            pp.srcNdMatrixStride = M * scoreCols; pp.dstNzMatrixStride = M * columns;
            DataCopy(prob, Gm<int8_t>(scratch + (kind ? TB_P1 : TB_P0)), pp);
            SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0); WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
            for (uint32_t t = 0; t < count; ++t) {
                auto k = kv[t * kvStride + (kind ? compressedOffset : 0)];
                for (uint32_t kk = 0; kk < columns; kk += 128) {
                    const uint32_t liveK = columns - kk < 128 ? columns - kk : 128;
                    LoadA(a, prob[t * M * columns + kk * M], liveK);
                    LoadValueB(b, k, columns, kk, liveK);
                    SetFlag<HardEvent::MTE1_M>(EVENT_ID0); WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
                    MmadParams mm{}; mm.m = M; mm.n = D; mm.k = liveK; mm.cmatrixInitVal = kk == 0;
                    Mmad(c, a, b, mm);
                    SetFlag<HardEvent::M_MTE1>(l0Consumed); WaitFlag<HardEvent::M_MTE1>(l0Consumed);
                }
                StoreAcc(Gm<int32_t>(scratch + (kind ? TB_V1 : TB_V0))[t * M * D], c, M, D, D);
            }
            // This releases P and, on the final kind, every KV L1 read
            // before the next batch's MTE2 loads/zero fill can overwrite it.
            SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0); WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
            CrossCoreSetFlag<2, PIPE_FIX>(kind ? PV_READY_SECOND : PV_READY);
        }
    }
    PipeBarrier<PIPE_ALL>();
    pipe.ReleaseEventID<HardEvent::M_MTE1>(l0Consumed);
}

__aicore__ __attribute__((noinline, aligned(2048), section(".text.sharedkv_task_batch"))) void TaskBatchVector(const SharedKVArgs &p, uint32_t tasks,
                               uint32_t uniformQueries, uint32_t oriRows, uint32_t cmpRows, uint32_t taskOrder)
{
    constexpr uint32_t ubBytes = TB_ROWS * D * 14 + 4 * D * 4 + (32 + TB_ROWS * 8) * 4
        + 5 * TB_SCALAR_ROWS * 4 + TB_SOFTMAX_BYTES + TB_SCALAR_ROWS * 8 * 4 + 64
        + 2 * TB_SCALAR_ROWS * 4 + 2 * TB_COLUMN_CAP * 4 + TB_MASK_BYTES + VECTOR_OPERAND_SKEW;
    static_assert(ubBytes <= 184 * 1024, "Batched Vector exceeds user UB");
    TPipe pipe;
    VectorUbRegion<TB_ROWS * D * 2> qb;
    VectorUbRegion<TB_ROWS * D * 4, decltype(qb)::end> xb;
    VectorUbRegion<TB_ROWS * D * 4, decltype(xb)::end + VECTOR_OPERAND_SKEW> rb;
    VectorUbRegion<TB_ROWS * D * 4, decltype(rb)::end> ab;
    VectorUbRegion<4 * D * 4, decltype(ab)::end> sb;
    VectorUbRegion<(32 + TB_ROWS * 8) * 4, decltype(sb)::end> reduceB;
    VectorUbRegion<2 * TB_SCALAR_ROWS * 4, decltype(reduceB)::end> maxB;
    VectorUbRegion<2 * TB_SCALAR_ROWS * 4, decltype(maxB)::end> sumB;
    VectorUbRegion<TB_SCALAR_ROWS * 4, decltype(sumB)::end> expB;
    VectorUbRegion<TB_SOFTMAX_BYTES, decltype(expB)::end> softB;
    VectorUbRegion<TB_SCALAR_ROWS * 8 * 4, decltype(softB)::end> brcB;
    VectorUbRegion<64, decltype(brcB)::end> sinkB;
    VectorUbRegion<2 * TB_SCALAR_ROWS * 4, decltype(sinkB)::end> alphaB;
    VectorUbRegion<TB_COLUMN_CAP * 4, decltype(alphaB)::end> maskB;
    VectorUbRegion<TB_COLUMN_CAP * 4, decltype(maskB)::end> indexB;
    VectorUbRegion<TB_MASK_BYTES, decltype(indexB)::end> bitsB;
    static_assert(decltype(bitsB)::end == ubBytes, "Fixed views must match the UB arena");
    TBuf<TPosition::VECCALC> arena;
    pipe.InitBuffer(arena, ubBytes);
    auto x = xb.Get<float>(), residual = rb.Get<float>();
    // x/residual (including the bank skew) are dead before product DMA. Pack
    // the two live product extents there; Q/P encoding retains its original
    // fixed scratch views. No storage beyond rb or GM layout is changed.
    LocalTensor<float> products(TPosition::VECCALC, x.GetPhyAddr(), 2 * TB_ROWS * D);
    // Raw BF16 Q retires after one conversion into the dead FP32 accumulator.
    // Both channel-scale encodings read that retained Q. Their half-cast
    // scratch uses softB before softmax starts; acc becomes PV accumulation
    // only after both Q encodings retire. qb is reused by encoded Q/P,
    // previous output and sparse membership after each prior consumer drains.
    auto q = qb.Get<bfloat16_t>();
    auto out = qb.Get<bfloat16_t>();
    auto i8 = qb.Get<int8_t>(); auto membership = qb.Get<float>();
    auto acc = ab.Get<float>(), scales = sb.Get<float>();
    auto r = reduceB.Get<float>(), maxBuf = maxB.Get<float>(), sumBuf = sumB.Get<float>();
    auto expBuf = expB.Get<float>(), brc = brcB.Get<float>(), sinks = sinkB.Get<float>();
    auto alpha = alphaB.Get<float>(), mask = maskB.Get<float>(), indices = indexB.Get<float>();
    auto bits = bitsB.Get<uint8_t>();
    static_assert((TB_ROWS * 64 + TB_SCALAR_ROWS * 8) * sizeof(float) <= TB_SOFTMAX_BYTES);
    const SoftmaxBuffers softmaxBuffers{
        static_cast<uint32_t>(x.GetPhyAddr()), static_cast<uint32_t>(sumBuf.GetPhyAddr()),
        static_cast<uint32_t>(maxBuf.GetPhyAddr()), static_cast<uint32_t>(expBuf.GetPhyAddr()),
        static_cast<uint32_t>(softB.Get<uint8_t>().GetPhyAddr())};
    const uint32_t cores = GetBlockNum(), core = GetBlockIdx() / 2, sub = GetBlockIdx() % 2;
    const uint32_t availableKinds = cmpRows ? 2 : 1;
    DataCopy(sinks, Gm<float>(p.sinks), 8);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0); WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    for (uint32_t h = 0; h < 8; ++h) sinks.SetValue(8 + h, sinks.GetValue((h + 4) % 8));
    SetFlag<HardEvent::S_V>(EVENT_ID0); WaitFlag<HardEvent::S_V>(EVENT_ID0);
    DataCopy(scales, Gm<float>(p.oriScale), D);
    if (cmpRows) DataCopy(scales[D], Gm<float>(p.cmpScale), D);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    Muls(scales[2 * D], scales, 1.0f / 127.0f, availableKinds * D);
    const auto resources = sharedkv_c8::PlanRetainedRows(oriRows, cmpRows);
    const uint32_t scoreCols = resources.scoreColumns;
    // GM keeps independent matrix slots. In UB, join just the live window
    // and 64-lane-aligned extents. Neither cache kind inflates the other's
    // Vector work; an early window need not compute its full128-key capacity.
    const uint32_t membershipColumns = (cmpRows + 63) / 64 * 64;
    CreateVecIndex(indices, 0.0f, scoreCols); PipeBarrier<PIPE_V>();
    const uint32_t width = resources.width;
    const bool directKindLookup = width == 1 && uniformQueries == 1 && cmpRows;
    const uint64_t scratch = p.workspace + static_cast<uint64_t>(core) * WORKSPACE_CORE_BYTES;
    const auto range = RetainedRange(p, tasks, uniformQueries, width, taskOrder);
    uint32_t previousBase = 0, previousCount = 0, previousKinds = 0;
    for (uint32_t base = 0; base < range.count || previousCount; base += width) {
        const uint32_t tail = base < range.count ? range.count - base : 0;
        const uint32_t count = tail < width ? tail : width, nr = count * 4;
        const uint32_t begin = sub * nr;
        uint32_t oriCount[TB_CAP], cmpCount[TB_CAP];
        uint32_t kinds = availableKinds;
        if (count && directKindLookup) {
            const uint32_t request = range.first + base * range.stride;
            kinds = Gm<int32_t>(p.seqLens).GetValue(request) >= int32_t(C8_RATIO) ? 2 : 1;
        }
      if (count) {
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0); WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            for (uint32_t row = 0; row < nr;) {
                const uint32_t t = (begin + row) / 8, head = (begin + row) % 8;
                const uint32_t live = nr - row < 8 - head ? nr - row : 8 - head;
                const uint32_t token = range.first + (base + t) * range.stride;
                DataCopy(q[row * D], Gm<bfloat16_t>(p.q)[(static_cast<uint64_t>(token) * 8 + head) * D], live * D);
                row += live;
            }
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Cast(acc, q, RoundMode::CAST_NONE, nr * D); PipeBarrier<PIPE_V>();
        for (uint32_t kind = 0; kind < kinds; ++kind) {
            ScaleChannels(x, acc, scales[kind * D], nr); PipeBarrier<PIPE_V>();
            EncodeQueryRows<TB_ROWS>(x, residual, r, brc, softB.Get<half>(), i8, nr, true);
            DataCopy(alpha[kind * TB_SCALAR_ROWS], r, (nr + 7) / 8 * 8); PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0); WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            auto dst = Gm<int8_t>(scratch + (kind ? TB_Q1 : TB_Q0));
            for (uint32_t row = 0; row < nr;) {
                const uint32_t t = (begin + row) / 8, head = (begin + row) % 8;
                const uint32_t live = nr - row < 8 - head ? nr - row : 8 - head;
                DataCopy(dst[(t * M + head) * D], i8[row * D], live * D);
                DataCopy(dst[(t * M + head + RESIDUAL_ROW_OFFSET) * D], i8[TB_ROWS * D + row * D], live * D);
                row += live;
            }
            PipeBarrier<PIPE_ALL>();
            // Warm-up has no previous PV alias to protect. Steady-state
            // Q readiness is released after the final previous-PV DMA.
            if (!previousCount) CrossCoreSetFlag<2, PIPE_MTE3>(kind ? Q_READY_SECOND : Q_READY);
        }
      }
        // Q's MTE3_V is followed by the V_MTE2 edge below. Once the last
        // old PV enters UB, next QK may reuse GM while Vector finishes output.
        if (previousCount) {
            const uint32_t previousRows = previousCount * 4;
            const uint32_t previousBegin = sub * previousRows;
            for (uint32_t kind = 0; kind < previousKinds; ++kind) {
                CrossCoreWaitFlag(kind ? PV_READY_SECOND : PV_READY);
                // The preceding Q encoding / PV accumulation must retire
                // before MTE2 overwrites the shared x/residual UB storage.
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0); WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                auto src = Gm<int32_t>(scratch + (kind ? TB_V1 : TB_V0));
                for (uint32_t row = 0; row < previousRows;) {
                    const uint32_t t = (previousBegin + row) / 8, head = (previousBegin + row) % 8;
                    const uint32_t live = previousRows - row < 8 - head ? previousRows - row : 8 - head;
                    DataCopy(x[row * D].ReinterpretCast<int32_t>(), src[(t * M + head) * D], live * D);
                    DataCopy(products[(previousRows + row) * D].ReinterpretCast<int32_t>(), src[(t * M + head + RESIDUAL_ROW_OFFSET) * D], live * D);
                    row += live;
                }
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                if (count && kind + 1 == previousKinds) {
                    CrossCoreSetFlag<2, PIPE_MTE2>(Q_READY);
                    if (kinds == 2) CrossCoreSetFlag<2, PIPE_MTE2>(Q_READY_SECOND);
                }
                DecodeResidualProducts(products, previousRows * D);
                // Both next-Q encodings have retired before this drain.
                // The first live PV contribution defines acc completely;
                // subsequent kinds add to it. No zero-fill/read is needed.
                if (kind == 0) ScaleChannels(acc, x, scales[2 * D], previousRows);
                else AccumulateChannels(acc, x, scales[(2 + kind) * D], previousRows);
                PipeBarrier<PIPE_V>();
            }
            Duplicate(r[TB_SCALAR_ROWS], 1.0f, TB_SCALAR_ROWS); PipeBarrier<PIPE_V>();
            Div(r[TB_SCALAR_ROWS], r[TB_SCALAR_ROWS], sumBuf[TB_SCALAR_ROWS], previousRows); PipeBarrier<PIPE_V>();
            Brcb(brc, r[TB_SCALAR_ROWS], (previousRows + 7) / 8, {1, 8}); PipeBarrier<PIPE_V>();
            ScaleRows(x, acc, brc, previousRows, D); PipeBarrier<PIPE_V>();
            Cast(out, x, RoundMode::CAST_RINT, previousRows * D);
            if (p.lse) {
                Ln(r, sumBuf[TB_SCALAR_ROWS], previousRows); PipeBarrier<PIPE_V>();
                Add(r, r, maxBuf[TB_SCALAR_ROWS], previousRows); PipeBarrier<PIPE_V>();
                // r now holds the live LSE values. Only the second
                // AIV of an odd batch starts at head4: after its first
                // four heads, shift r[4:] once to restore32B DMA alignment.
                // Score predicates and the mask are dead until next score.
                if ((previousBegin & 4) && previousRows > 4) {
                    auto offsets = bits.ReinterpretCast<int32_t>();
                    ArithProgression<int32_t>(offsets, int32_t(16), int32_t(4), previousRows - 4);
                    PipeBarrier<PIPE_V>();
                    Gather(mask, r, offsets.ReinterpretCast<uint32_t>(), uint32_t(0), previousRows - 4);
                }
            }
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0); WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            for (uint32_t row = 0; row < previousRows;) {
                const uint32_t t = (previousBegin + row) / 8, head = (previousBegin + row) % 8;
                const uint32_t live = previousRows - row < 8 - head ? previousRows - row : 8 - head;
                const uint32_t token = range.first + (previousBase + t) * range.stride;
                const uint64_t outRow = static_cast<uint64_t>(token) * 8 + head;
                DataCopy(Gm<bfloat16_t>(p.out)[outRow * D], out[row * D], live * D);
                if (p.lse) {
                    const auto source = (row & 7) ? mask[row - 4] : r[row];
                    DataCopyPad(Gm<float>(p.lse)[outRow], source, DataCopyExtParams{1, live * 4, 0, 0, 0});
                }
                row += live;
            }
            PipeBarrier<PIPE_ALL>();
        }
        if (!count) break;  // drain iteration owes no new Q/P notifications
        // The Cube can consume Q while Vector prepares sparse membership.
        // Previous output DMA has retired; Q's local bytes are now dead.
        uint32_t activeOri = 0, activeCmp = 0;
        for (uint32_t row = 0; row < nr;) {
            const uint32_t t = (begin + row) / 8, head = (begin + row) % 8;
            const uint32_t live = nr - row < 8 - head ? nr - row : 8 - head;
            Selection sel(p, range.first + (base + t) * range.stride, 1, uniformQueries);
            oriCount[t] = sel.oriCount; cmpCount[t] = sel.cmpCount;
            activeOri = sel.oriCount > activeOri ? sel.oriCount : activeOri;
            activeCmp = sel.cmpCount > activeCmp ? sel.cmpCount : activeCmp;
            if constexpr (C8_RATIO == 4) {
                if (kinds == 2) BuildC4Membership<TB_COLUMN_CAP, TB_ROWS>(p, sel, head, live,
                    membership[t * TB_COLUMN_CAP], x, residual, membershipColumns);
            }
            row += live;
        }
        const uint32_t originalCols = (activeOri + 63) / 64 * 64;
        const uint32_t compressedCols = kinds == 2 ? (activeCmp ? (activeCmp + 63) / 64 * 64 : 64) : 0;
        const uint32_t scoreWidth = originalCols + compressedCols;
        CrossCoreWaitFlag(SCORE_READY);
        if (kinds == 2) CrossCoreWaitFlag(SCORE_READY_SECOND);
        if constexpr (C8_RATIO == 4) {
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0); WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        }
        DataCopyExtParams cp{}; cp.blockCount = nr;
        for (uint32_t kind = 0; kind < kinds; ++kind) {
            const uint32_t columns = kind ? compressedCols : originalCols;
            const uint32_t offset = kind ? originalCols : 0;
            cp.blockLen = columns * 4;
            cp.srcStride = (scoreCols - columns) * 4;
            cp.dstStride = (scoreWidth - columns) * 4 / 32;
            auto src = Gm<int32_t>(scratch + (kind ? TB_S1 : TB_S0));
            for (uint32_t row = 0; row < nr;) {
                const uint32_t t = (begin + row) / 8, head = (begin + row) % 8;
                const uint32_t live = nr - row < 8 - head ? nr - row : 8 - head;
                cp.blockCount = live;
                DataCopyPad(x[row * scoreWidth + offset].ReinterpretCast<int32_t>(), src[(t * M + head) * scoreCols], cp, DataCopyPadExtParams<int32_t>{});
                DataCopyPad(products[(nr + row) * scoreWidth + offset].ReinterpretCast<int32_t>(), src[(t * M + head + RESIDUAL_ROW_OFFSET) * scoreCols], cp, DataCopyPadExtParams<int32_t>{});
                row += live;
            }
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        DecodeResidualProducts(products, nr * scoreWidth);
        for (uint32_t kind = 0; kind < kinds; ++kind) {
            const uint32_t columns = kind ? compressedCols : originalCols;
            const uint32_t offset = kind ? originalCols : 0;
            Muls(r[TB_SCALAR_ROWS], alpha[kind * TB_SCALAR_ROWS], p.softmaxScale, nr); PipeBarrier<PIPE_V>();
            Brcb(brc, r[TB_SCALAR_ROWS], (nr + 7) / 8, {1, 8}); PipeBarrier<PIPE_V>();
            BinaryRepeatParams rp{}; rp.dstBlkStride = rp.src0BlkStride = 1; rp.src1BlkStride = 0;
            rp.dstRepStride = rp.src0RepStride = scoreWidth / 8; rp.src1RepStride = 1;
            for (uint32_t col = 0; col < columns; col += 64)
                Mul(x[offset + col], x[offset + col], brc, 64, nr, rp);
            PipeBarrier<PIPE_V>();
            for (uint32_t row = 0; row < nr;) {
                const uint32_t t = (begin + row) / 8, head = (begin + row) % 8;
                const uint32_t live = nr - row < 8 - head ? nr - row : 8 - head;
                if constexpr (C8_RATIO == 4) {
                    if (kind == 1) {
                        for (uint32_t h = 0; h < live; ++h)
                            Add(x[(row + h) * scoreWidth + offset], x[(row + h) * scoreWidth + offset],
                                membership[t * TB_COLUMN_CAP], columns);
                        PipeBarrier<PIPE_V>();
                        row += live;
                        continue;
                    }
                }
                CompareScalar(bits, indices, static_cast<float>(static_cast<int32_t>(kind ? cmpCount[t] : oriCount[t])), CMPMODE::LT, columns);
                // The public C220 mode1 overload allows scalar/CMPMASK setup
                // once per query. Every head restarts at the same predicate;
                // repeats consume its consecutive64-bit column masks.
                Duplicate(mask, -1e30f, 8); PipeBarrier<PIPE_V>();
                SetCmpMask(mask.ReinterpretCast<uint64_t>()); PipeBarrier<PIPE_V>();
                SetMaskNorm(); SetVectorMask<float>(uint64_t(0), ~uint64_t(0));
                for (uint32_t h = 0; h < live; ++h) {
                    auto score = x[(row + h) * scoreWidth + offset];
                    Select(score, bits, score, static_cast<uint8_t>(columns / 64), BinaryRepeatParams{1, 1, 1, 8, 8, 8});
                }
                PipeBarrier<PIPE_V>(); ResetMask();
                row += live;
            }
        }
        // Non-online softmax initializes the next sum from its sink. The
        // previous sum has already been consumed by the PV drain above.
        for (uint32_t row = 0; row < nr; row += 8)
            DataCopy(maxBuf[row], sinks[(begin & 4) ? 8 : 0], 8);
        PipeBarrier<PIPE_V>();
        CompactSoftmax<TB_ROWS, TB_SCALAR_ROWS>(nr, scoreWidth, softmaxBuffers);
        PipeBarrier<PIPE_V>(); Muls(x, x, 127.0f, nr * scoreWidth); PipeBarrier<PIPE_V>();
        RoundResidualInt8(i8, i8[TB_ROWS * D], acc.ReinterpretCast<half>(), x, residual, nr * scoreWidth);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0); WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        for (uint32_t kind = 0; kind < kinds; ++kind) {
            const uint32_t columns = kind ? compressedCols : originalCols;
            const uint32_t offset = kind ? originalCols : 0;
            cp.blockLen = columns; cp.srcStride = (scoreWidth - columns) / 32;
            cp.dstStride = scoreCols - columns;
            auto dst = Gm<int8_t>(scratch + (kind ? TB_P1 : TB_P0));
            for (uint32_t row = 0; row < nr;) {
                const uint32_t t = (begin + row) / 8, head = (begin + row) % 8;
                const uint32_t live = nr - row < 8 - head ? nr - row : 8 - head;
                cp.blockCount = live;
                DataCopyPad(dst[(t * M + head) * scoreCols], i8[row * scoreWidth + offset], cp);
                DataCopyPad(dst[(t * M + head + RESIDUAL_ROW_OFFSET) * scoreCols], i8[TB_ROWS * D + row * scoreWidth + offset], cp);
                row += live;
            }
        }
        // The next loop reloads Q into P's former UB bytes with MTE2;
        // its existing MTE2_V then protects all subsequent Vector writers.
        PipeBarrier<PIPE_ALL>();
        CrossCoreSetFlag<2, PIPE_MTE3>(P_READY);
        if (kinds == 2) CrossCoreSetFlag<2, PIPE_MTE3>(P_READY_SECOND);
        previousBase = base; previousCount = count; previousKinds = kinds;
    }
    // Every live output and its source UB have drained above. Reuse the
    // retired query/output buffer for graph padding instead of constructing
    // another TPipe and rescanning the immutable metadata. `tasks` is the
    // live token count at this entry, not this core's assigned range count.
    if (tasks < p.tokens) {
        const uint64_t padding = uint64_t(p.tokens - tasks) * 8;
        const uint32_t lanes = GetBlockNum() * 2;
        const uint64_t perLane = (padding + lanes - 1) / lanes;
        const uint64_t first = uint64_t(GetBlockIdx()) * perLane;
        if (first < padding) {
            const uint64_t rows = padding - first < perLane ? padding - first : perLane;
            const uint32_t zeroRows = rows < TB_ROWS ? uint32_t(rows) : TB_ROWS;
            Duplicate(q.ReinterpretCast<uint16_t>(), uint16_t(0), zeroRows * D);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0); WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            for (uint64_t offset = 0; offset < rows; offset += zeroRows) {
                const uint32_t count = rows - offset < zeroRows ? uint32_t(rows - offset) : zeroRows;
                const uint64_t row = uint64_t(tasks) * 8 + first + offset;
                DataCopy(Gm<bfloat16_t>(p.out)[row * D], q, count * D);
                // Exact-width DMA preserves neighboring live LSE values.
                if (p.lse) DataCopyPad(Gm<float>(p.lse)[row], q.ReinterpretCast<float>(),
                    DataCopyExtParams{1, count * 4, 0, 0, 0});
            }
            PipeBarrier<PIPE_ALL>();
        }
    }
}

__aicore__ __attribute__((always_inline)) inline void RunTaskBatch(const SharedKVArgs &p,
    uint32_t tasks, uint32_t uniformQueries, uint32_t maximumCompressed, uint32_t maximumOriginal,
    uint32_t taskOrder = 0)
{
    const uint32_t aligned = (maximumCompressed + 31) / 32 * 32;
    const uint32_t cmpRows = aligned;
    const uint32_t window = maximumOriginal < WINDOW_ROWS ? maximumOriginal : WINDOW_ROWS;
    const uint32_t oriRows = (window + 31) / 32 * 32;
    if ASCEND_IS_AIC { TaskBatchCube(p, tasks, uniformQueries, oriRows, cmpRows, taskOrder); }
    if ASCEND_IS_AIV { TaskBatchVector(p, tasks, uniformQueries, oriRows, cmpRows, taskOrder); }
}
