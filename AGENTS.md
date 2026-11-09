# Maintained baseline

This is the standalone organization of v65-p101-stage1, package 0.65.1+memcache.10. The p101 device baseline remains unchanged.
Read README.md, docs/architecture.md and docs/validation.md before changing it.

- Default checks are CPU-only. Run NPU regression or serving only for an
  explicitly authorized validation task with available devices. Preserve
  unrelated tests and services; isolate installations and restore any existing
  service configuration after comparison. Compilation and CPU tests are allowed.
- Preserve Q/P double residuals, true INT8 Cube QK/PV, FP32 softmax/accumulation,
  and frozen-channel sharedkv-c8-v1 encoding. KV has no residual.
- Validated platform: Ascend910_9382 / DAV2201, CANN 9.2. No context parallel.
- Keep every csrc/kernels and csrc/host source byte and translation-unit order
  stable during organization. Parent kernels contain required writer/ABI paths.
- Do not restore rejected p102-p110 experiments or import sibling workspaces.
- Default verification is bash tools/test_cpu.sh. Source identity check:
  python3 tools/check_release.py; add --include-build after a full build.
- release/upstream-v65-p101.json is immutable. It is not the reorganized tree's
  manifest. Do not regenerate it to accept an unintended change.
- Never rebuild or replace libraries used by an active service.

- MemCache integration requires the paired patch and generated compatibility manifest.
- Keep pressure datasets, private deployment snapshots, profiles and machine paths outside the repository.
