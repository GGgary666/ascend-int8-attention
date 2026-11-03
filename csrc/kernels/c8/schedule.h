#pragma once
#include "args.h"
#include "kernel_operator.h"
#include "metadata.h"
#include "task_partition.h"

namespace sharedkv_c8 {
constexpr uint32_t kCores = 24;
constexpr uint32_t kMaxSplits = 4;

// Derived solely from live metadata, never graph capacity or a case name.
struct Workload {
    uint32_t batch = 0, queries = 0, pairedTasks = 0, packedTasks = 0;
    uint32_t maximum = 0;
    uint32_t wideCost = 0, pairedCost = 0, packedCost = 0;
    uint32_t taskOrder = 0;
    uint32_t uniformQueries = 0;
    uint32_t packedOrder = 1u << 13;
};

// A packed per-core pair needs no GM task table. Both AIC and its two
// AIV partners derive the same indices from the same live metadata.
constexpr uint32_t kPairedOrder = 1u << 12;
constexpr uint32_t kSerpentineOrder = 1u << 13;
__aicore__ inline uint32_t SerpentineTask(uint32_t task)
{
    const uint32_t wave = task / kCores, lane = task % kCores;
    return wave * kCores + ((wave & 1) ? kCores - 1 - lane : lane);
}

// Exact cyclic wave load for arbitrary task counts. Only the physical cores
// need local counters; storage is independent of batch and query capacity.
__aicore__ inline void AddWaveCost(uint32_t (&tails)[kCores], uint32_t &base,
                                 uint32_t begin, uint32_t count, uint32_t cost)
{
    base += (count / kCores) * cost;
    const uint32_t remainder = count % kCores;
    const uint32_t start = begin % kCores;
    for (uint32_t i = 0; i < remainder; ++i) tails[(start + i) % kCores] += cost;
}

__aicore__ inline void AddPackedWaveCost(uint32_t (&tails)[kCores], uint32_t &base,
    uint32_t begin, uint32_t count, uint32_t cost)
{
    // Alternating wave direction distributes repeated full/tail groups across
    // different cores. Every complete 48-task period visits each core twice.
    base += (count / (2 * kCores)) * 2 * cost;
    for (uint32_t i = 0; i < count % (2 * kCores); ++i)
        tails[SerpentineTask(begin + i) % kCores] += cost;
}

// Heterogeneous requests can favor linear ownership. Keep the extra cost
// pass out of the common uniform-work planner and compare exact peak loads.
static __aicore__ __attribute__((noinline)) uint32_t LinearPackedCost(const SharedKVArgs &p, uint32_t batch)
{
    AscendC::GlobalTensor<int32_t> cu, lengths;
    cu.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(p.cuQ));
    lengths.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(p.seqLens));
    uint32_t tails[kCores] = {}, base = 0, begin = 0, task = 0;
    for (uint32_t b = 0; b < batch; ++b) {
        const uint32_t end = cu.GetValue(b + 1), count = end - begin;
        if (count) {
            const int32_t length = lengths.GetValue(b);
            const uint32_t keys = length > 0 ? static_cast<uint32_t>(length) / 128 : 0;
            const uint32_t groups = (count + 3) / 4, tiles = 1 + (keys + 255) / 256;
            AddWaveCost(tails, base, task, groups - 1, 3 * tiles);
            AddWaveCost(tails, base, task + groups - 1, 1, ((count - 1) % 4 < 2 ? 2 : 3) * tiles);
            task += groups;
        }
        begin = end;
    }
    uint32_t peak = 0;
    for (uint32_t core = 0; core < kCores; ++core) peak = tails[core] > peak ? tails[core] : peak;
    return peak + base;
}

static __aicore__ __attribute__((noinline, section(".text.sharedkv_schedule")))
Workload BalancePackedTwoWaves(const SharedKVArgs &p, Workload w);

template<uint32_t Ratio>
__aicore__ __attribute__((noinline, section(".text.sharedkv_schedule")))
Workload ReadManyWaves(const SharedKVArgs &p, Workload w)
{
    AscendC::GlobalTensor<int32_t> cu, lengths;
    cu.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(p.cuQ));
    lengths.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(p.seqLens));
    // When every request has one query, grouping cannot remove any tasks.
    // Validate the entire live prefix before using direct request indexing;
    // equal totals alone do not exclude zero-query metadata rows.
    if (w.queries == w.batch) {
        bool single = true;
        if (w.batch >= kMetadataVectorMinimum) {
            const auto summary = ReadVectorMetadata(p, w.batch);
            single = summary.minimumQueries == 1 && summary.maximumQueries == 1;
            w.maximum = summary.maximum;
        } else {
            for (uint32_t b = 0; b < w.batch; ++b) {
                single = single && cu.GetValue(b + 1) == b + 1;
                const int32_t length = lengths.GetValue(b);
                if (length > 0 && static_cast<uint32_t>(length) > w.maximum) w.maximum = length;
            }
        }
        if (single) {
            w.uniformQueries = 1;
            w.pairedTasks = w.packedTasks = w.queries;
            const uint32_t keys = w.maximum / Ratio;
            w.wideCost = ((w.queries + kCores - 1) / kCores) * (1 + (keys + 511) / 512);
            w.pairedCost = ((w.queries + kCores - 1) / kCores) * (1 + (keys + 255) / 256);
            w.packedCost = 2 * w.pairedCost;
            return w;
        }
        w.maximum = 0;
    }
    uint32_t wideTail[kCores] = {}, pairedTail[kCores] = {}, packedTail[kCores] = {};
    uint32_t wideBase = 0, pairedBase = 0, packedBase = 0, begin = 0;
    uint32_t minimum = 0xffffffffu;
    if constexpr (Ratio == 128) w.uniformQueries = cu.GetValue(1);
    for (uint32_t b = 0; b < w.batch; ++b) {
        const int32_t length = lengths.GetValue(b);
        const uint32_t end = cu.GetValue(b + 1);
        const uint32_t count = end - begin;
        if constexpr (Ratio == 128)
            if (count != w.uniformQueries) w.uniformQueries = 0;
        if (count) {
            const uint32_t safeLength = length > 0 ? static_cast<uint32_t>(length) : 0;
            w.maximum = safeLength > w.maximum ? safeLength : w.maximum;
            minimum = safeLength < minimum ? safeLength : minimum;
            uint32_t keys = safeLength / Ratio;
            if constexpr (Ratio == 4) keys = keys < 1024 ? keys : 1024;
            AddWaveCost(wideTail, wideBase, begin, count, 1 + (keys + 511) / 512);
            const uint32_t pairs = (count + 1) / 2;
            AddWaveCost(pairedTail, pairedBase, w.pairedTasks, pairs, 1 + (keys + 255) / 256);
            w.pairedTasks += pairs;
            if constexpr (Ratio == 128) {
                const uint32_t groups = (count + 3) / 4;
                const uint32_t tiles = 1 + (keys + 255) / 256;
                // Half-task units: a full G4 costs 3, a <=2-query tail
                // costs 2. Tail Vector work follows live rows, not capacity.
                AddPackedWaveCost(packedTail, packedBase, w.packedTasks, groups - 1, 3 * tiles);
                AddPackedWaveCost(packedTail, packedBase, w.packedTasks + groups - 1, 1,
                    ((count - 1) % 4 < 2 ? 2 : 3) * tiles);
                w.packedTasks += groups;
            }
        }
        begin = end;
    }
    for (uint32_t core = 0; core < kCores; ++core) {
        w.wideCost = wideTail[core] > w.wideCost ? wideTail[core] : w.wideCost;
        w.pairedCost = pairedTail[core] > w.pairedCost ? pairedTail[core] : w.pairedCost;
        w.packedCost = packedTail[core] > w.packedCost ? packedTail[core] : w.packedCost;
    }
    w.wideCost += wideBase;
    w.pairedCost += pairedBase;
    w.packedCost += packedBase;
    // With one or two groups per uniform request, the alternating full/tail
    // pattern cannot have a higher peak than linear ownership on 24 cores.
    // Larger periods and heterogeneous costs require the exact second pass.
    if constexpr (Ratio == 128)
        if (w.packedTasks > kCores && w.packedTasks < w.pairedTasks &&
            (!w.uniformQueries || (w.uniformQueries + 3) / 4 > 2 ||
             (minimum / 128 + 255) / 256 != (w.maximum / 128 + 255) / 256)) {
            const uint32_t linear = LinearPackedCost(p, w.batch);
            if (linear < w.packedCost) { w.packedCost = linear; w.packedOrder = 0; }
        }
    if constexpr (Ratio == 128)
        if (w.packedTasks > kCores && w.packedTasks <= 2 * kCores &&
            w.packedTasks < w.pairedTasks &&
            (minimum / 128 + 255) / 256 != (w.maximum / 128 + 255) / 256)
            w = BalancePackedTwoWaves(p, w);
    return w;
}

// The uncommon two-wave rank search is kept out of the shared hot entry.
// Input arrays still contain at most two core waves; this is a planning
// bound, never a limit on accepted batch/query capacities.
struct TwoWavePair { uint32_t cost, order; };
__aicore__ __attribute__((noinline, section(".text.sharedkv_schedule")))
static TwoWavePair BalanceTwoWaves(const uint32_t *ends, const uint32_t *costs,
    uint32_t cached, uint32_t queries, uint32_t wideCost)
{
    uint32_t core = AscendC::GetBlockIdx();
    if ASCEND_IS_AIV { core /= 2; }
    const uint32_t pairs = queries - kCores;
    const uint32_t singles = 2 * kCores - queries;
    const uint32_t rankFirst = core < pairs ? singles + core : core - pairs;
    const uint32_t rankSecond = queries - 1 - core;
    uint32_t firstTask = 0, secondTask = 63;
    uint32_t firstBound = 0, secondBound = 0, begin = 0;
    uint32_t uniqueCosts[2 * kCores], nextRanks[2 * kCores], unique = 0;
    for (uint32_t b = 0; b < cached; ++b) {
        const uint32_t count = ends[b] - begin;
        uint32_t slot = 0;
        while (slot < unique && uniqueCosts[slot] != costs[b]) ++slot;
        if (slot == unique) {
            uint32_t rank = 0, prior = 0;
            for (uint32_t j = 0; j < cached; ++j) {
                if (costs[j] > costs[b]) rank += ends[j] - prior;
                prior = ends[j];
            }
            uniqueCosts[unique] = costs[b]; nextRanks[unique++] = rank;
        }
        // Tile costs repeat across requests. Stable equal-cost order follows
        // source order, so its next rank can be advanced without another scan.
        const uint32_t rank = nextRanks[slot]; nextRanks[slot] += count;
        if (rankFirst >= rank && rankFirst < rank + count)
            firstTask = begin + rankFirst - rank;
        if (core < pairs && rankSecond >= rank && rankSecond < rank + count)
            secondTask = begin + rankSecond - rank;
        if (singles >= rank && singles < rank + count) firstBound = costs[b];
        if (kCores >= rank && kCores < rank + count) secondBound = costs[b];
        begin = ends[b];
    }
    uint32_t bound = firstBound + secondBound;
    bound = bound > wideCost ? bound : wideCost;
    return TwoWavePair{bound, kPairedOrder | firstTask | (secondTask << 6)};
}

// Two waves of grouped work can reuse the bounded opposite-cost pairing
// without a global task table. The limit bounds this planner's stack only;
// arbitrary request/query capacities retain the common workload planner.
static __aicore__ __attribute__((noinline, section(".text.sharedkv_schedule")))
Workload BalancePackedTwoWaves(const SharedKVArgs &p, Workload w)
{
    AscendC::GlobalTensor<int32_t> cu, lengths;
    cu.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(p.cuQ));
    lengths.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(p.seqLens));
    uint32_t ends[2 * kCores], costs[2 * kCores];
    uint32_t cached = 0, begin = 0, tasks = 0, maximumCost = 0;
    for (uint32_t b = 0; b < w.batch; ++b) {
        const uint32_t end = cu.GetValue(b + 1), count = end - begin;
        if (count) {
            const int32_t length = lengths.GetValue(b);
            const uint32_t keys = length > 0 ? static_cast<uint32_t>(length) / 128 : 0;
            const uint32_t tiles = 1 + (keys + 255) / 256;
            const uint32_t groups = (count + 3) / 4;
            if (groups > 1) {
                tasks += groups - 1; ends[cached] = tasks; costs[cached++] = 3 * tiles;
                maximumCost = maximumCost > 3 * tiles ? maximumCost : 3 * tiles;
            }
            ++tasks; ends[cached] = tasks;
            const uint32_t cost = ((count - 1) % 4 < 2 ? 2 : 3) * tiles;
            costs[cached++] = cost; maximumCost = maximumCost > cost ? maximumCost : cost;
        }
        begin = end;
    }
    const auto pair = BalanceTwoWaves(ends, costs, cached, tasks, maximumCost);
    if (pair.cost < w.packedCost) { w.packedCost = pair.cost; w.packedOrder = pair.order; }
    return w;
}

template<uint32_t Ratio = 128>
__aicore__ inline Workload PlanWorkload(const SharedKVArgs &p, Workload w)
{
    AscendC::GlobalTensor<int32_t> cu, lengths;
    cu.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(p.cuQ));
    lengths.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(p.seqLens));
    if (!w.batch) return w;
    if (w.queries > 2 * kCores) return ReadManyWaves<Ratio>(p, w);
    if constexpr (Ratio == 128) w.uniformQueries = w.batch > kCores && w.queries > kCores ? cu.GetValue(1) : 0;
    uint32_t begin = 0;
    uint32_t prefixCost = 0, tailCost = 0, minimumCost = 0xffffffffU;
    // At most two waves can contain at most 2*kCores nonempty requests.
    // This cache bounds an optimization, never the admitted metadata shape.
    uint32_t ends[2 * kCores] = {}, costs[2 * kCores] = {}, cached = 0;
    uint32_t packedTail[kCores] = {}, packedBase = 0;
    for (uint32_t b = 0; b < w.batch; ++b) {
        const int32_t length = lengths.GetValue(b);
        if (length > 0 && static_cast<uint32_t>(length) > w.maximum) w.maximum = length;
        // At <=24 queries the paired task count cannot exceed24, even
        // with one-query tails. Avoid reading every cuQ in that common case.
        if (w.queries > kCores) {
            const uint32_t end = cu.GetValue(b + 1);
            if constexpr (Ratio == 128)
                if (end - begin != w.uniformQueries) w.uniformQueries = 0;
            // At <=48 queries G1 has at most two waves. Bound their
            // overlapping prefix/tail separately, instead of charging every
            // core twice for the longest request in a heterogeneous batch.
            uint32_t keys = static_cast<uint32_t>(length) / Ratio;
            if constexpr (Ratio == 4) keys = keys < 1024 ? keys : 1024;
            const uint32_t tiles = 1 + (keys + 511) / 512;
            if (end > begin) {
                ends[cached] = end; costs[cached++] = tiles;
                w.wideCost = tiles > w.wideCost ? tiles : w.wideCost;
                if (begin < w.queries - kCores) prefixCost = tiles > prefixCost ? tiles : prefixCost;
                if (end > kCores) tailCost = tiles > tailCost ? tiles : tailCost;
                minimumCost = tiles < minimumCost ? tiles : minimumCost;
            }
            if constexpr (Ratio == 128) {
                const uint32_t count = end - begin;
                if (count) {
                    const uint32_t groups = (count + 3) / 4;
                    const uint32_t packedTiles = 1 + (keys + 255) / 256;
                    AddPackedWaveCost(packedTail, packedBase, w.packedTasks, groups - 1, 3 * packedTiles);
                    AddPackedWaveCost(packedTail, packedBase, w.packedTasks + groups - 1, 1,
                        ((count - 1) % 4 < 2 ? 2 : 3) * packedTiles);
                    w.packedTasks += groups;
                }
            }
            w.pairedTasks += (end - begin + 1) / 2; begin = end;
        }
    }
    const uint32_t twoWaves = prefixCost + tailCost;
    const uint32_t forwardCost = twoWaves > w.wideCost ? twoWaves : w.wideCost;
    uint32_t bestCost = forwardCost;
    // With <=48 queries, at most two tasks share one of24 cores. Keep
    // the largest (48-Q) tasks alone and pair the remaining short tasks
    // in opposite cost order. Stable request order breaks equal-cost ties.
    // The conservative pair bound must strictly improve on linear order.
    if (w.queries > kCores && minimumCost < w.wideCost && (Ratio == 4 || w.maximum > 32768)) {
        const auto pair = BalanceTwoWaves(ends, costs, cached, w.queries, w.wideCost);
        if (pair.cost < bestCost) { bestCost = pair.cost; w.taskOrder = pair.order; }
    }
    w.wideCost = bestCost;
    w.pairedCost = (1 + (w.maximum / 128 + 255) / 256) * ((w.pairedTasks + kCores - 1) / kCores);
    if constexpr (Ratio == 128) {
        for (uint32_t core = 0; core < kCores; ++core)
            w.packedCost = packedTail[core] > w.packedCost ? packedTail[core] : w.packedCost;
        w.packedCost += packedBase;
        if (w.packedTasks > kCores && w.packedTasks < w.pairedTasks) {
            const uint32_t linear = LinearPackedCost(p, w.batch);
            if (linear < w.packedCost) { w.packedCost = linear; w.packedOrder = 0; }
        }
    }
    return w;
}

template<uint32_t Ratio = 128>
__aicore__ inline Workload ReadWorkload(const SharedKVArgs &p)
{
    AscendC::GlobalTensor<int32_t> cu, lengths;
    cu.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(p.cuQ));
    lengths.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(p.seqLens));
    Workload w;
    w.batch = p.batch;
    while (w.batch && (lengths.GetValue(w.batch - 1) <= 0 || cu.GetValue(w.batch) < 0)) --w.batch;
    if (w.batch) w.queries = cu.GetValue(w.batch);
    return PlanWorkload<Ratio>(p, w);
}

__aicore__ inline uint32_t SplitCount(const Workload &w, uint32_t kvTiles)
{
    // Parallelize KV only while query tasks occupy <= half the AICs.
    if (!w.queries || w.queries > kCores / 2) return 1;
    uint32_t parts = kCores / w.queries;
    parts = parts < kMaxSplits ? parts : kMaxSplits;
    return parts < kvTiles ? parts : kvTiles;
}

enum class DensePlan { Narrow, Medium, Paired, Packed, Wide };
__aicore__ inline DensePlan DenseSchedule(const Workload &w)
{
    const uint32_t keys = w.maximum / 128;
    // Independent requests have no adjacent-query KV reuse to give up.
    // Their retained batch width is set by L1 capacity inside TaskBatch.
    if (w.queries > kCores && w.pairedTasks == w.queries && keys > 256 && keys <= kRetainedKeys)
        return DensePlan::Medium;
    if (w.packedTasks && w.packedTasks < w.pairedTasks && w.packedCost < 2 * w.pairedCost
            && w.packedCost < 2 * w.wideCost) return DensePlan::Packed;
    if (w.queries > 2 * kCores) {
        // G4 doubles M and vector rows relative to G2, while sharing KV and
        // per-task setup. Charge 1.5 task units conservatively; require fewer
        // groups and a strictly shorter peak-core load than both alternatives.
        // Odd request tails are included, so q5/q6 do not blindly select G4.
        if (w.pairedTasks < w.queries && w.pairedCost < w.wideCost) return DensePlan::Paired;
        return keys <= 128 ? DensePlan::Narrow : keys <= 256 ? DensePlan::Medium : DensePlan::Wide;
    }
    if (keys > 256) return w.queries > kCores && w.pairedCost < w.wideCost
        ? DensePlan::Paired : DensePlan::Wide;
    // G2 either removes a work wave or reuses KV across >=2 query pairs
    // per request. Short partial requests otherwise retain query parallelism.
    if (w.pairedTasks <= kCores && (w.queries > kCores || w.queries >= 4 * w.batch))
        return DensePlan::Paired;
    return keys <= 128 ? DensePlan::Narrow : DensePlan::Medium;
}
}
