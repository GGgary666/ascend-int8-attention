// Intentionally included inside each implementation namespace.
#ifndef SHARED_TILE
#define SHARED_TILE 128
#endif
#ifndef SHARED_QUERY_GROUP
#define SHARED_QUERY_GROUP 1
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
#ifndef SHARED_CACHE_LIVE
#define SHARED_CACHE_LIVE 0
#endif
#ifndef SHARED_DENSE_L0_PIPELINE
#define SHARED_DENSE_L0_PIPELINE 0
#endif
#ifndef SHARED_ROW_SCALE
#define SHARED_ROW_SCALE 0
#endif
#ifndef SHARED_SMALL_C4
#define SHARED_SMALL_C4 0
#endif
// Keep ordinary variants as the original field expressions. Inline helper
// functions changed their code generation despite returning the same fields.
#undef SHARED_HEADS
#undef SHARED_RATIO
#if SHARED_SMALL_C4 || defined(SHARED_FIXED_H8_C4)
#define SHARED_HEADS(p) 8u
#define SHARED_RATIO(p) 4u
#else
#define SHARED_HEADS(p) p.heads
#define SHARED_RATIO(p) p.ratio
#endif
// Optional reuse for the legacy small-C128 entry. Other variants retain
// their measured call signatures; a register-allocation cause was not proved.
#undef SHARED_LIVE_PARAM
#undef SHARED_LIVE_ARG
#undef SHARED_LIVE_BATCH
#undef SHARED_LIVE_TOKENS
#if SHARED_CACHE_LIVE
#define SHARED_LIVE_PARAM , SharedKVLiveShape live
#define SHARED_LIVE_ARG , live
#define SHARED_LIVE_BATCH live.batch
#define SHARED_LIVE_TOKENS live.tokens
#else
#define SHARED_LIVE_PARAM
#define SHARED_LIVE_ARG
#define SHARED_LIVE_BATCH ActualBatch(p)
#define SHARED_LIVE_TOKENS ActualTokens(p)
#endif
constexpr uint32_t M = 16 * SHARED_QUERY_GROUP, N = SHARED_TILE, D = 512;
// A paired query uses 16 real rows and 16 residual rows. Each AIV owns at
// most eight real rows; allocating R=16 would needlessly exceed A2/A3 UB.
// G3 has twelve real rows per vector partner; keep its vector buffers and
// Brcb repeat count aligned to eight without rounding the Cube M up to64.
constexpr uint32_t R = SHARED_QUERY_GROUP == 3 ? 16 : SHARED_QUERY_GROUP > 1 ? M / 4 : M / 2;
constexpr uint32_t RESIDUAL_ROW_OFFSET = M / 2;
static_assert(N == 128 || N == 256 || N == 512);
static_assert(!SHARED_SMALL_C4 || (N <= 256 && SHARED_QUERY_GROUP <= 2 && SHARED_ROW_SCALE));
constexpr uint32_t Q_OFFSET = 0;
constexpr uint32_t SCORE_OFFSET = Q_OFFSET + M * D;
constexpr uint32_t P_OFFSET = SCORE_OFFSET + M * N * 4;
constexpr uint32_t PV_OFFSET = P_OFFSET + M * N;
constexpr uint32_t KV_OFFSET = PV_OFFSET + M * D * 4;
constexpr uint32_t SLOTS_OFFSET = KV_OFFSET + N * D;
// Isolate the scalar-read count and keep every core's DMA buffers 512 B
// aligned, as in upstream sharedkv's cached MTE2-size workspace allocation.
constexpr uint32_t WORKSPACE_BYTES = SLOTS_OFFSET + 512;
constexpr uint16_t COMPACT_CONSUMED = 3, Q_READY = 4, KV_READY = 5, COMPACT_READY = 6, SCORE_READY = 7, P_READY = 8, PV_READY = 9, TILE_DONE = 10;
// dav_c220 reserves cross-core flags 11..14 for SyncAll/superkernel.
// Grouped dense tasks do not use the GM sparse-KV staging region; hold a
// second quantized Q and score there without changing the host workspace ABI.
constexpr uint16_t Q_READY_SECOND = 1, SCORE_READY_SECOND = 2;
constexpr uint32_t Q_SECOND_OFFSET = KV_OFFSET;
constexpr uint32_t SCORE_SECOND_OFFSET = Q_SECOND_OFFSET + M * D;
constexpr uint32_t KV_L1_SLOTS = SHARED_QUERY_GROUP > 1 || SHARED_SMALL_C4 ? 2 : 1;
static_assert(SHARED_QUERY_GROUP == 1 || M * D + M * N * 4 <= N * D);
