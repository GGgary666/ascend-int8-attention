// Exercise the production scalar planner, including both AIV partners.
#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
// Device section placement is irrelevant to this scalar-only CPU check.
// Host GCC can reject a shared section for COMDAT templates and static helpers.
#pragma push_macro("section")
#define section(...)
#include "schedule.h"
#pragma pop_macro("section")
bool c8_test_aiv = false;
uint32_t c8_test_block = 0;
bool c8_test_poison_paired_summary = false;
namespace sharedkv_c8 {
// Independent CPU provider for the planner's device-summary interface. This
// tests decisions, not Vector arithmetic or flags; hardware tests cover those.
static MetadataSummary ReadVectorMetadata(const SharedKVArgs &p, uint32_t batch) {
    const auto *cu = reinterpret_cast<const int32_t *>(p.cuQ);
    const auto *lengths = reinterpret_cast<const int32_t *>(p.seqLens);
    MetadataSummary result{0, 0x7fffffffu, 0, 0};
    for (uint32_t b = 0; b < batch; ++b) {
        const uint32_t count = cu[b + 1] - cu[b];
        result.maximum = std::max(result.maximum, uint32_t(std::max(lengths[b], 0)));
        result.minimumQueries = std::min(result.minimumQueries, count);
        result.maximumQueries = std::max(result.maximumQueries, count);
        result.paired += count / 2 + count % 2;
    }
    if (c8_test_poison_paired_summary) result.paired = 0xdeadbeefu;
    return result;
}
}
template<unsigned Ratio> void Check(std::mt19937 &rng, unsigned count, bool large = false) {
    for (unsigned trial = 0; trial < count; ++trial) {
        std::vector<int32_t> cu(large ? 257 : 9, 0), lengths(large ? 256 : 8, 0);
        SharedKVArgs p{};
        p.batch = 1 + rng() % (large ? 256 : 8);
        for (unsigned b = 0; b < p.batch; ++b) {
            cu[b+1] = cu[b] + (large ? rng() % 32 : 1 + rng() % 6);
            lengths[b] = 1 + rng() % 135000;
        }
        p.tokens = cu[p.batch];
        p.cuQ = reinterpret_cast<uint64_t>(cu.data());
        p.seqLens = reinterpret_cast<uint64_t>(lengths.data());
        std::vector<unsigned> seen(cu[p.batch], 0), costs, paired, packed;
        for (unsigned b = 0; b < p.batch; ++b) {
            unsigned keys = lengths[b] / Ratio;
            if constexpr (Ratio == 4) keys = std::min(keys, 1024u);
            costs.insert(costs.end(), cu[b+1]-cu[b], 1+(keys+511)/512);
            paired.insert(paired.end(), (cu[b+1]-cu[b]+1)/2, 1+(keys+255)/256);
            for (unsigned start=cu[b]; start<static_cast<unsigned>(cu[b+1]); start+=4)
                packed.push_back((std::min(4u,static_cast<unsigned>(cu[b+1])-start)>2 ? 3 : 2)*(1+(keys+255)/256));
        }
        std::vector<unsigned> packedSeen(packed.size(), 0);
        std::vector<unsigned> packedByCost(packed.size());
        std::iota(packedByCost.begin(), packedByCost.end(), 0u);
        std::stable_sort(packedByCost.begin(), packedByCost.end(),
                         [&](unsigned a, unsigned b) { return packed[a] > packed[b]; });
        for (unsigned core = 0; core < 24; ++core) {
            c8_test_aiv = false; c8_test_block = core;
            const auto w = sharedkv_c8::ReadWorkload<Ratio>(p);
            if (w.uniformQueries) {
                for(unsigned b=0;b<w.batch;++b) assert(cu[b+1]-cu[b]==w.uniformQueries);
            }
            unsigned load = 0;
            for (unsigned task = core; task < costs.size(); task += 24) {
                unsigned index = w.taskOrder ? (task < 24 ? w.taskOrder & 63 : (w.taskOrder >> 6) & 63) : task;
                assert(index < costs.size());
                ++seen[index]; load += costs[index];
            }
            if (w.queries > 24) assert(load <= w.wideCost);
            if (w.queries > 48) {
                unsigned pairLoad=0,packedLoad=0;
                for (unsigned task=core;task<paired.size();task+=24) pairLoad+=paired[task];
                for (unsigned task=core;task<(packed.size()+23)/24*24;task+=24) {
                    unsigned index=(w.packedOrder & sharedkv_c8::kPairedOrder) ?
                        (task<24 ? w.packedOrder & 63u : (w.packedOrder>>6)&63u) :
                        w.packedOrder ? sharedkv_c8::SerpentineTask(task) : task;
                    if(index<packed.size()) { packedLoad+=packed[index]; ++packedSeen[index]; }
                }
                assert(w.pairedTasks==paired.size() && pairLoad<=w.pairedCost);
                if constexpr (Ratio==128) assert(w.packedTasks==packed.size() && packedLoad<=w.packedCost);
                if constexpr (Ratio==128) if (w.packedOrder & sharedkv_c8::kPairedOrder) {
                    const unsigned pairs=packed.size()-24, singles=48-packed.size();
                    const unsigned firstRank=core<pairs ? singles+core : core-pairs;
                    assert((w.packedOrder&63u)==packedByCost[firstRank]);
                    assert(((w.packedOrder>>6)&63u)==
                           (core<pairs ? packedByCost[packed.size()-1-core] : 63u));
                }
            }
            for (unsigned partner = 0; partner < 2; ++partner) {
                c8_test_aiv = true; c8_test_block = 2*core+partner;
                const auto v = sharedkv_c8::ReadWorkload<Ratio>(p);
                assert(v.taskOrder == w.taskOrder);
                assert(v.wideCost == w.wideCost && v.maximum == w.maximum);
                assert(v.pairedCost == w.pairedCost && v.packedCost == w.packedCost);
                assert(v.packedOrder == w.packedOrder);
                assert(sharedkv_c8::DenseSchedule(v)==sharedkv_c8::DenseSchedule(w));
            }
        }
        assert(std::all_of(seen.begin(), seen.end(), [](unsigned n){ return n == 1; }));
        if constexpr (Ratio == 128)
            if (cu[p.batch] > 48)
                assert(std::all_of(packedSeen.begin(), packedSeen.end(), [](unsigned n){ return n == 1; }));
    }
}
void CheckUniform() {
    for (unsigned batch : {1u,7u,9u,23u,24u,31u,48u,49u,64u,128u,256u,512u}) {
        for(unsigned queries : {1u,2u,3u,5u,6u,7u,17u}) {
            std::vector<int32_t> cu(batch+1),lengths(batch);
            for(unsigned b=0;b<batch;++b) { cu[b+1]=cu[b]+queries;lengths[b]=129+(b*8191)%134000; }
            SharedKVArgs p{};p.batch=batch;p.tokens=cu.back();
            p.cuQ=reinterpret_cast<uint64_t>(cu.data());p.seqLens=reinterpret_cast<uint64_t>(lengths.data());
            const auto w=sharedkv_c8::ReadWorkload<128>(p);
            if(w.uniformQueries)assert(w.uniformQueries==queries);
            if(batch>24 && p.tokens>24)assert(w.uniformQueries==queries);
            // Equal totals with an empty request must never imply q1 metadata.
            if(queries==1 && batch>=2) {
                cu[1]=0;
                const auto holes=sharedkv_c8::ReadWorkload<128>(p);
                assert(holes.uniformQueries==0);
            }
            for(unsigned group : {1u,2u,4u}) {
                unsigned tasks=batch*((queries+group-1)/group);
                std::vector<unsigned> seen(tasks,0);
                for(unsigned core=0;core<24;++core)
                    for(unsigned task=core;task<(tasks+23)/24*24;task+=24) {
                        unsigned index=sharedkv_c8::SerpentineTask(task);
                        if(index<tasks)++seen[index];
                    }
                assert(std::all_of(seen.begin(),seen.end(),[](unsigned x){return x==1;}));
            }
        }
    }
}
void CheckUniformCosts() {
    for (unsigned batch : {1u,7u,9u,17u,25u,31u,48u,49u,67u,128u,512u})
        for (unsigned queries : {1u,2u,3u,4u,5u,6u,7u,8u,9u,17u,29u,33u,45u,49u,61u,65u,93u,97u,189u,193u,197u,4096u}) {
            if (batch*queries<=48 || batch*queries>65536) continue;
            for (unsigned length : {129u,16384u,16385u,32768u,32769u,65537u,131073u}) {
                if (queries>length) continue;
                std::vector<int32_t> cu(batch+1),lengths(batch);
                for (unsigned b=0;b<batch;++b) { cu[b+1]=cu[b]+queries;lengths[b]=length+b%7; }
                SharedKVArgs p{};p.batch=batch;p.tokens=cu.back();
                p.cuQ=reinterpret_cast<uint64_t>(cu.data());p.seqLens=reinterpret_cast<uint64_t>(lengths.data());
                const auto w=sharedkv_c8::ReadWorkload<128>(p);
                std::vector<unsigned> wide(24),paired(24),linear(24),alternating(24);
                unsigned ti=0,pi=0,gi=0;
                for(unsigned b=0;b<batch;++b) {
                    unsigned keys=lengths[b]/128,tiles=1+(keys+255)/256;
                    for(unsigned q=0;q<queries;++q) wide[ti++%24]+=1+(keys+511)/512;
                    for(unsigned q=0;q<queries;q+=2) paired[pi++%24]+=tiles;
                    for(unsigned q=0;q<queries;q+=4,++gi) {
                        unsigned cost=(std::min(queries-q,4u)>2 ? 3 : 2)*tiles;
                        linear[gi%24]+=cost;
                        alternating[sharedkv_c8::SerpentineTask(gi)%24]+=cost;
                    }
                }
                auto peak=[](const auto &v){return *std::max_element(v.begin(),v.end());};
                assert(w.wideCost==peak(wide) && w.pairedCost==peak(paired));
                assert(w.packedCost==std::min(peak(linear),peak(alternating)));
                assert(w.packedCost==peak(w.packedOrder ? alternating : linear));
                assert(w.pairedTasks==pi && w.packedTasks==gi);
                if (peak(linear)==peak(alternating)) assert(w.packedOrder==sharedkv_c8::kSerpentineOrder);
                if (batch>24) assert(w.uniformQueries==queries);
            }
        }
}
void CheckSummaryPlannerBoundary() {
    for (unsigned batch : {64u, 97u, 513u}) {
        std::vector<int32_t> cu(batch + 1), lengths(batch, 129);
        std::iota(cu.begin(), cu.end(), 0);
        SharedKVArgs p{}; p.batch = batch; p.tokens = batch;
        p.cuQ = reinterpret_cast<uint64_t>(cu.data());
        p.seqLens = reinterpret_cast<uint64_t>(lengths.data());
        for (unsigned core = 0; core < 24; ++core) {
            for (unsigned participant = 0; participant < 3; ++participant) {
                c8_test_aiv = participant != 0;
                c8_test_block = participant ? 2 * core + participant - 1 : core;
                cu[1] = 1; lengths[1] = 129;
                const auto single = sharedkv_c8::ReadWorkload<128>(p);
                assert(single.uniformQueries == 1 && single.maximum == 129);
                c8_test_poison_paired_summary = true;
                const auto poisonedSingle = sharedkv_c8::ReadWorkload<128>(p);
                c8_test_poison_paired_summary = false;
                assert(poisonedSingle.pairedTasks==single.pairedTasks && poisonedSingle.pairedCost==single.pairedCost);
                // Same total, but counts [2,0,1,...]. The all-row summary's
                // large length belongs to an inactive request and must be
                // discarded before the general weighted planner runs.
                cu[1] = 2; lengths[1] = 131073;
                const auto ragged = sharedkv_c8::ReadWorkload<128>(p);
                assert(ragged.uniformQueries == 0 && ragged.maximum == 129);
                c8_test_poison_paired_summary = true;
                const auto poisonedRagged = sharedkv_c8::ReadWorkload<128>(p);
                c8_test_poison_paired_summary = false;
                assert(poisonedRagged.pairedTasks==ragged.pairedTasks && poisonedRagged.pairedCost==ragged.pairedCost);
                assert(poisonedRagged.packedCost==ragged.packedCost && poisonedRagged.packedOrder==ragged.packedOrder);
            }
        }
    }
}
void CheckRetainedPairOwnership() {
    std::mt19937 rng(20261030);
    unsigned pairedPlans = 0;
    for (unsigned queries = 25; queries <= 48; ++queries) {
        for (unsigned trial = 0; trial < 32; ++trial) {
            std::vector<int32_t> cu(queries + 1), lengths(queries);
            std::iota(cu.begin(), cu.end(), 0);
            for (auto &length : lengths) length = 1 + rng() % (960 * 128);
            SharedKVArgs p{}; p.batch = p.tokens = queries;
            p.cuQ = reinterpret_cast<uint64_t>(cu.data());
            p.seqLens = reinterpret_cast<uint64_t>(lengths.data());
            // Enumerate every possible retained task width, including odd
            // batches whose middle query is split across the AIV partners.
            for (unsigned width = 1; width <= 5; ++width) {
                std::vector<unsigned> seen(queries * 8, 0);
                for (unsigned core = 0; core < 24; ++core) {
                    c8_test_aiv = false; c8_test_block = core;
                    const auto w = sharedkv_c8::ReadWorkload<128>(p);
                    const auto range = w.taskOrder ? sharedkv_c8::PairedTaskRange(w.taskOrder)
                        : sharedkv_c8::PartitionTasks(queries, core, 24, false);
                    if (w.taskOrder) {
                        ++pairedPlans;
                        std::vector<unsigned> expected, actual;
                        for (unsigned task=core; task<queries; task+=24)
                            expected.push_back(task<24 ? w.taskOrder&63u : (w.taskOrder>>6)&63u);
                        for (unsigned t=0;t<range.count;++t) actual.push_back(range.first+t*range.stride);
                        std::sort(expected.begin(),expected.end());
                        assert(actual==expected);
                    }
                    for (unsigned base = 0; base < range.count; base += width) {
                        const unsigned count = std::min(width, range.count - base);
                        for (unsigned sub = 0; sub < 2; ++sub) {
                            for (unsigned row = sub * count * 4; row < (sub + 1) * count * 4; ++row) {
                                const unsigned token = range.first + (base + row / 8) * range.stride;
                                assert(token < queries);
                                ++seen[token * 8 + row % 8];
                            }
                        }
                    }
                }
                for (auto count : seen) assert(count == 1);
            }
        }
    }
    assert(pairedPlans > 0);
}
void CheckTaskRanges() {
    for (uint32_t tasks : {0u,1u,23u,24u,25u,49u,97u,129u,513u,1025u,8193u,16777217u,2147483647u})
        for (bool contiguous : {false,true}) {
            uint64_t total = 0;
            std::vector<unsigned> seen(tasks <= 8193 ? tasks : 0, 0);
            for (uint32_t core = 0; core < 24; ++core) {
                const auto range = sharedkv_c8::PartitionTasks(tasks,core,24,contiguous);
                total += range.count;
                if (range.count) assert(uint64_t(range.first)+(uint64_t(range.count)-1)*range.stride < tasks);
                if (tasks <= 8193)
                    for (uint32_t i=0;i<range.count;++i) ++seen[range.first+i*range.stride];
                if (contiguous) {
                    assert(range.first==uint64_t(tasks)*core/24);
                    assert(uint64_t(range.first)+range.count==uint64_t(tasks)*(core+1)/24);
                } else assert(range.first==core && range.stride==24);
            }
            assert(total==tasks);
            for (auto n : seen) assert(n==1);
        }
    for (uint32_t ratio : {4u,128u})
        for (uint32_t length : {0u,1u,63u,64u,127u,128u,513u,32769u,98304u,16777217u,2147483647u}) {
            const uint32_t expected=4+(std::min(length,128u)+63)/64+(std::min(length/ratio,960u)+63)/64;
            assert(sharedkv_c8::RetainedQueryCost(length,ratio)==expected);
        }
    assert(!sharedkv_c8::PreferContiguous(60,58));
    assert(sharedkv_c8::PreferContiguous(60,47));
    assert(!sharedkv_c8::PreferContiguous(100,91));
}
void CheckRetainedResources() {
    using namespace sharedkv_c8;
    static_assert(kRetainedKeys==960);
    // Independently enumerate feasible layouts instead of reproducing the
    // production quotient. Include empty and all aligned resource boundaries.
    for(unsigned original=0;original<=128;original+=32)
        for(unsigned compressed=0;compressed<=960;compressed+=32) {
            const auto plan=PlanRetainedRows(original,compressed);
            assert(plan.scoreColumns>=compressed && plan.scoreColumns%64==0);
            unsigned expectedWidth=0;
            bool expectedReuse=false;
            for(bool reuse : {false,true}) {
                unsigned rows=reuse?std::max(original,compressed):original+compressed;
                for(unsigned width=1;width<=5;++width) {
                    const uint64_t q=uint64_t(width)*16*512,p=uint64_t(width)*16*plan.scoreColumns;
                    const uint64_t kv=uint64_t(width)*rows*512;
                    if(q+p+kv<=512*1024-128 && width>expectedWidth)expectedWidth=width;
                }
                if(expectedWidth){expectedReuse=reuse;break;}
            }
            assert(plan.width==expectedWidth && plan.reuseKvSlot==expectedReuse);
            assert(plan.residentRows==(expectedReuse?std::max(original,compressed):original+compressed));
            assert(!plan.reuseKvSlot || plan.width==1);
            assert(plan.width*plan.scoreColumns<=5*512);
            const unsigned joint=(original+63)/64*64+(compressed+63)/64*64;
            assert(4*plan.width*joint<=20*512 && joint/8<=255);
            if(compressed<=896 && original+compressed) {
                const unsigned oldResident=original+compressed<=896?original+compressed:std::max(original,compressed);
                assert(plan.width==std::min(5u,896/oldResident));
            }
        }
    // Publisher must use actual width; half of the enlarged KV limit would
    // incorrectly exclude this plan and let partner1 read an old mode record.
    assert(PlanRetainedRows(128,352).width==1);
    assert(!PlanRetainedRows(128,832).reuseKvSlot);
    assert(PlanRetainedRows(128,864).reuseKvSlot);
}
int main() {
    CheckRetainedResources();
    CheckTaskRanges();
    CheckRetainedPairOwnership();
    CheckSummaryPlannerBoundary();
    CheckUniform();
    CheckUniformCosts();
    std::mt19937 rng(20261028);
    Check<4>(rng, 10000); Check<128>(rng, 10000);
    Check<4>(rng, 1000, true); Check<128>(rng, 1000, true);
    std::cout << "PASS: 22000 randomized live workloads, 84 uniform/boundary workloads and exact uniform-cost enumeration through B512; bijective mapping, uniform-metadata validation, peak-cost bounds, AIC/AIV agreement\n";
}
