# Final MTP KV-reuse components

This directory retains the final, separately validated two-query implementations:

- `int8/`: INT8 QK/PV, Q/P double residuals, FP32 softmax/accumulation,
  N256 gather/QK with the original two N128 numerical stages.
- `bf16/`: BF16 QK/PV with FP32 softmax/accumulation; outputs matched the
  original BF16 implementation bitwise in the frozen 60-shape / 30-real-input checks.

Both require H8/D512, explicit physical slot lists of width256, and a caller-proven
contract that queries within each request use the same ordered slots. Query count
or context length alone does not establish this contract. They retain KV for QK/PV
and reuse it between two queries; a one-query group tail remains supported.

These are optional source components. The installed package, general vLLM plugin,
and maintained p101 device sources keep their existing behavior. No experimental
hybrid-PV, no-reuse ablation, discarded scheduling branches, profiles, model
inputs, or prebuilt shared libraries are included here.

## Build

Use the previously validated A3/DAV2201, CANN9.2, torch2.10 / torch_npu2.10
environment. Install and build the parent package first. For the INT8 component:

```bash
cd extras/mtp/int8
source /usr/local/Ascend/cann-9.2.0/set_env.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
mkdir -p lib
cp build/libsharedkv_mtp.so lib/
python3 build_host.py
```

For BF16, run the same CMake commands and `python3 build_host.py` from
`extras/mtp/bf16`; its launcher loads `build/libbf16_mtp_g2.so` directly.
These commands compile code; they do not execute an NPU validation suite.

## Call contract

From a source checkout, the INT8 component exposes
`extras.mtp.int8.bind(original, contract=SHARED_SLOT_CONTRACT)`.
The caller must establish the ordered-slot invariant before passing the exported
contract object. Unsupported ABI shapes retain the original prepared operation.
For repeated serving calls, use `rebind(candidate, original)` and clear retained
input references with `clear_inputs(candidate)` after launching, as in the
validated adapter. Each candidate owns its larger workspace.

BF16 exposes `extras.mtp.bf16.Prepared(case, contract=SHARED_SLOT_CONTRACT)`.
Its case has BF16 Q/KV, explicit INT32 physical slots and BF16 element strides.
The retained implementation is G2 only. It supports `bind` with fresh inputs and
an ordered queue that captures launch arguments by value.

The common raw-pointer host bridge is retained as source with each component.
Integrating either component into another framework requires preserving slot
provenance, workspace ownership and stream/queue ordering; importing this directory
alone does not install a serving hook.

## Evidence and limits

`source-identity.json` freezes every retained source and records the device-library
identities from the original validation. Relocation did not change device source;
the BF16 build/API was narrowed to the final G2 implementation. New builds must be
validated on their target environment rather than assumed to have identical ELF hashes.

BF16 has 60 sampled/held-out and30 actual-input checks, plus96 fresh bindings and96
graph updates. The INT8 final numerical route has the same60/30 coverage. The actual
input set contains15 post-prompt and15 chunked-prefill-boundary captures.
No model activations or captured tokens are published here.

Kernel speedups are not E2E speedups. With equally optimized BF16 KV reuse,
INT8 MTP was slower in the earlier real-input comparison. Main C4/C128 INT8
precision limitations are inherited and not resolved by these MTP components.
