// C8 production core: frozen-channel KV, Q/P double residuals.
// Intentionally included inside each implementation namespace.
#ifndef SHARED_TILE
#define SHARED_TILE 128
#endif
#ifndef SHARED_QUERY_GROUP
#define SHARED_QUERY_GROUP 1
#endif
#ifndef SHARED_ALTERNATING_WAVES
#define SHARED_ALTERNATING_WAVES 1
#endif
#ifndef SHARED_PACKED_C4
#define SHARED_PACKED_C4 0
#endif
#ifndef SHARED_ADAPTIVE_SPLITS
#define SHARED_ADAPTIVE_SPLITS 0
#endif
#ifndef SHARED_CUBE_TAILS
#define SHARED_CUBE_TAILS 0
#endif
#ifndef SHARED_ADAPTIVE_DENSE
#define SHARED_ADAPTIVE_DENSE 0
#endif
#ifndef SHARED_DENSE_L0_PIPELINE
#define SHARED_DENSE_L0_PIPELINE 0
#endif
// Both residual products share the fixed H8/D512 C8 ABI.
#undef SHARED_HEADS
#undef SHARED_RATIO
#define SHARED_HEADS(p) 8u
#define SHARED_RATIO(p) C8_RATIO
#define SHARED_LIVE_BATCH ActualBatch(p)
#define SHARED_LIVE_TOKENS ActualTokens(p)
// Guarded by c8_attention.supported; no run-time division for fixed page geometry.
constexpr uint32_t BLOCK_ROWS = 32, WINDOW_ROWS = 128, C4_TOPK = 1024;
constexpr uint32_t M = 16 * SHARED_QUERY_GROUP, N = SHARED_TILE, D = 512;
// Four real rows per query on each AIV; the minimum eight-row reserve keeps
// reductions and rounded vector copies aligned. Cube retains both residuals.
constexpr uint32_t R = SHARED_QUERY_GROUP > 1 ? 4 * SHARED_QUERY_GROUP : 8;
// The Cube and ordinary Vector phases share the same live column extent.
// GM row pitch remains N; tail tiles compact only their temporary UB rows.
__aicore__ inline uint32_t TileColumns(uint32_t remaining)
{
    const uint32_t active = remaining <= 128 ? 128 : remaining <= 256 ? 256 : N;
    return SHARED_CUBE_TAILS && active < N ? active : N;
}
static_assert(SHARED_QUERY_GROUP == 1 || SHARED_QUERY_GROUP == 2 || SHARED_QUERY_GROUP == 4);
constexpr uint32_t RESIDUAL_ROW_OFFSET = M / 2;
static_assert(N == 128 || N == 256 || N == 512);
constexpr uint32_t Q_OFFSET = 0;
constexpr uint32_t SCORE_OFFSET = Q_OFFSET + M * D;
constexpr uint32_t P_OFFSET = SCORE_OFFSET + M * N * 4;
constexpr uint32_t PV_OFFSET = P_OFFSET + M * N;
constexpr uint32_t KV_OFFSET = PV_OFFSET + M * D * 4;
constexpr uint32_t SLOTS_OFFSET = KV_OFFSET + N * D;
// Isolate the scalar-read count and keep every core's DMA buffers 512 B
// aligned, as in upstream sharedkv's cached MTE2-size workspace allocation.
constexpr uint16_t COMPACT_CONSUMED = 3, Q_READY = 4, KV_READY = 5, COMPACT_READY = 6, SCORE_READY = 7, P_READY = 8, PV_READY = 9, TILE_DONE = 10;
// dav_c220 reserves cross-core flags 11..14 for SyncAll/superkernel.
// Grouped dense tasks do not use the GM sparse-KV staging region; hold a
// second quantized Q and score there without changing the host workspace ABI.
constexpr uint16_t Q_READY_SECOND = 1, SCORE_READY_SECOND = 2;
constexpr uint32_t Q_SECOND_OFFSET = KV_OFFSET;
constexpr uint32_t SCORE_SECOND_OFFSET = Q_SECOND_OFFSET + M * D;
constexpr bool JOINT_DENSE_TILE = C8_RATIO == 128 && N <= 256;
constexpr uint32_t P_SECOND_OFFSET = SCORE_SECOND_OFFSET + M * N * 4;
constexpr uint32_t PV_SECOND_OFFSET = P_SECOND_OFFSET + M * N;
constexpr uint32_t JOINT_WORKSPACE_END = PV_SECOND_OFFSET + M * D * 4;
constexpr uint32_t WORKSPACE_BYTES = JOINT_DENSE_TILE && JOINT_WORKSPACE_END > SLOTS_OFFSET + 512
    ? JOINT_WORKSPACE_END : SLOTS_OFFSET + 512;
// Every plan occupies its own physical-core arena. AIC/AIV pairs may choose
// different local extents, so their base stride cannot depend on that plan.
constexpr uint32_t WORKSPACE_CORE_BYTES = 491520;
constexpr uint16_t P_READY_SECOND = 3, PV_READY_SECOND = 6;
constexpr uint32_t KV_L1_SLOTS = SHARED_QUERY_GROUP > 1 || JOINT_DENSE_TILE ? 2 : 1;
static_assert(!JOINT_DENSE_TILE || JOINT_WORKSPACE_END <= WORKSPACE_BYTES);
static_assert(WORKSPACE_BYTES <= WORKSPACE_CORE_BYTES, "Per-core workspace exceeds host reservation");
static_assert(SHARED_QUERY_GROUP == 1 || M * D + M * N * 4 <= N * D);
