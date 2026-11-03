# Frozen-channel C8 attention core

This is the frozen **V65-p101 stage1** implementation for `sharedkv-c8-v1`.
The stage decision and boundaries are in `docs/STAGE_RELEASE.md`; verify its
source/device identity with `python3 tools/check_release.py --include-build`.
Historical
experiments live in the external archive indexed by `docs/archive.json`.
Unsupported-shape compatibility kernels remain in the parent directory.
`SHAREDKV_C8_DECODE=0` explicitly rolls back; no v63/v64 selectors are imported.

## Contract

- Ascend910_9382, DAV2201, CANN9.2; 24 AICs and their 48 AIV partners.
- BF16 Q and output, H8/D512, C4 or C128, SWA128, 32-row physical cache pages.
- Symmetric INT8 KV and frozen FP32 channel scales; no KV residual storage.
- Q **and** P residuals, INT8 Cube QK/PV, FP32 softmax/accumulation and optional LSE.
- No fixed batch/query capacity ceiling. C4 top-k1024. Query metadata
  must be monotone, within capacity, and describe valid causal cache positions.
- Decode, prefill and larger captured graphs share this implementation. Other
  ABIs, including explicit draft SWA slots and H!=8, retain compatibility kernels.
  An unsupported input is never silently re-encoded.

`c8_attention.py` owns the ABI guard and per-stream maximum scratch reservation.
`host/c8_launch.cpp` copies launch arguments into the torch_npu queue before the
Python object can be rebound. Neither path reads sequence lengths back to CPU.

## Code map

| File | Responsibility |
| --- | --- |
| `schedule.h` | Live workload, tile/wave cost bounds and split selection |
| `task_partition.h` | Bounded resource choices, cyclic/contiguous ownership and two-wave pair decoding |
| `metadata.h`, `metadata_impl.h`, `metadata_balance.h` | Bounded exact metadata summaries and optional per-core cost comparison |
| `task_batch.h` | Retained task phases, compact Vector extents and shared resource plan |
| `c4.cpp`, `c4_small.cpp` | Dynamic C4 and capacity<24 entry images |
| `c4_wide.h` | Shared ungrouped C4 implementation and bounded split selection |
| `c4_narrow.h` | Short N128 C4 with one online state, without SplitKV reduction |
| `c128.cpp` | Five tile/group plans and two compiled G4 ownership orders |
| `attention_impl.h` | Task selection, Cube, split reduction, padding cleanup |
| `vector.h` | Q encoding, softmax, P encoding and final normalization |
| `joint_tile.h` | Short C128 joint softmax/P encoding with separate INT8 PV products |
| `cube_pair.h` | Two-kind SWA/compressed pipeline and joint buffer protocol |
| `gather.h` | Sparse selection, page caching and gather |
| `residual_quant.h`, `vector_math.h` | Shared numeric primitives |
| `config.h`, `args.h` | Compile-time buffer layout and unchanged launch ABI |

There is no case-name table, calibrated-length list or benchmark lookup. C4
grouping requires all compressed keys to fit a membership tile and enough query
tasks to benefit. C128 compares an upper bound on peak per-core tile work. For
at most two waves, the longest tasks run alone and the remaining short tasks
are paired in opposite cost order. A conservative peak-cost bound must improve
before reordering is used. At most48 nonempty request descriptors are cached
for two-wave reordering; empty metadata rows do not consume cache entries.
Larger workloads use per-core cost counters, with storage independent of B/T;
no GM scheduling tensor or extra kernel is needed. G4 full/three-query groups
cost3 half-task units; one/two-query tails cost2. Linear and alternating-wave
ownership are compared when heterogeneous costs or longer request periods
can favor linear order. Uniform one/two-group requests avoid the extra pass;
the alternating order is no worse under this cost model on24 cores. Ties
retain alternating G4 ownership. Exact CPU enumeration checks the model.

Uniform-query direct indexing is used only after validating every cumulative
boundary; total queries equaling batch alone does not establish q1. It is
compiled into C128 G1 and C4 G2 where measurements support it. Other plans
retain metadata traversal. Short C4 with at most128 compressed keys and less
than one wave uses N128/splits1; larger workloads keep grouped/wide tiling.

The smaller C4 entry exists because capacity<24 proves the grouped path is
unreachable. Keeping its instruction image small addresses measured IB stalls;
it uses the same implementation header, arithmetic and event protocol. Keep
device function ordering stable unless matched profiling/timing supports a
change: merely moving the large C4 wrappers changed measured instruction-cache
behavior on this compiler/device.

## Memory and synchronization

G1 uses a fixed16-row Cube tile (8 main + 8 residual); both AIV partners
process4 heads. These and the block/window/top-k constants follow the guarded
ABI, not measured case IDs. Unsupported explicit-slot gathers are absent from
this core and remain in the legacy fallback.

Every plan uses a fixed491520-byte workspace stride per physical AIC, also
when cores select different live extents. Ordinary G1/N512 needs344576 live
bytes, G2/N256 needs254464 and G4/N256 needs377344; retained phases determine
the maximum reservation for24 AICs. Split reduction
reserves four FP32 partial slots of520 floats/head for at most12 live queries,
even in large captured graphs (maximum798720 bytes). Allocation selects C8
before reserving legacy scratch. KV cache capacity/encoding do not change.

G4 L0C uses128 KiB. Static assertions bound each user UB arena by184 KiB;
the retained-task arena uses186464 bytes. Channel scales and sinks stay resident during
a launch. Short G1/C128 tasks encode the two scale-dependent Qs together,
keep both KV tiles in L1, and combine scores for one softmax/P encoding phase.
Q/P high and low rows remain intact; separate INT8 QK/PV products retain each
KV kind's channel scale. Joint P rounding differs from sequential online P
rounding and has a separate independent integer reference.

Retained tasks share one L1 resource calculation between metadata, Cube and
Vector. The524160-byte L1 budget admits one to five tasks and up to960 keys;
larger work continues with the existing INT8 tile loop. When both cache kinds
do not fit simultaneously, their KV slot is reused with the existing release
events. A resource limit selects granularity and residency, never input
admission or a BF16 path. C128 retained execution consumes the same two-wave
pair ownership selected by the cost planner; both AIVs and AIC use one range.

The second Q/score/P/PV regions use otherwise unused C128 sparse staging;
compile-time bounds prevent overlap. Q0/Q1 flags are4/1, score flags7/2,
joint P flags8/3 and joint PV flags9/6. C4 compaction also uses3/6 but is a
different compile-time ratio path. Task reordering applies the same index to
AIC and both AIV partners; workspace stays indexed by physical core. The next
Q is emitted after preceding PV consumption, protecting cross-task region
reuse. Within a sequential tile loop, the next P follows the previous PV read.
The named TILE_DONE constant is not an emitted event. Complete protocol notes
and all static audit candidates are retained in the current report; passing
tests is not a formal synchronization proof.

## Building and validating

Only incremental targets are needed:

```sh
cmake --build build --target sharedkv_c8 -j2
python3 tools/build_host.py --component core
env -u SHAREDKV_C8_DECODE SHAREDKV_HOST_QUEUE=1 python3 -m tests.test_attention
env -u SHAREDKV_C8_DECODE SHAREDKV_HOST_QUEUE=1 python3 -m tests.test_dynamic
env -u SHAREDKV_C8_DECODE SHAREDKV_HOST_QUEUE=1 python3 -m tests.test_queue
```

Never overlap NPU benchmarks or profiler sessions. The maintained tests compare
the independent residual oracle and BF16; an optional frozen V65 device library
supports same-address graph timing. The archived full acceptance additionally
compares original INT8 and fixed-schedule v64, whose CPU-fixed schedule is only
a standalone control. See the reports in `docs/reports` for measured limits and
source/binary hashes. Default-promotion and capacity-extension results are
historical; current changes and remaining performance limits are indexed by
`docs/OPTIMIZATION_20261029.md`. Read its A/A measurement caveat before treating
a small version difference as an algorithmic regression.

Capacity tests (results should live outside the source tree):

```sh
python3 -m tests.test_capacity --output /absolute/results/capacity.json
python3 -m tests.test_capacity_dynamic --output /absolute/results/dynamic.json
```
