# Changelog

## 2026-11-09：最终 MTP 组件整理

- 保存最终 INT8/BF16 G2 同序物理 KV 复用源码、构建和有序 launcher，采用手动接入。
- 附原60 shape/30真实输入及270请求 MemCache 对比的聚合证据，明确公平比较和精度边界。
- 保持维护版 p101 默认行为；不引入混合 PV、消融或调试候选。

## 2026-11 — MemCache integration (`0.65.1+memcache.10`)

- Add a versioned compact KV namespace, numerical scale identity and paired framework validation.
- Transfer real component bytes, precompute complete block plans, and preserve shared-backing guards.
- Keep requests waiting until synchronous GET and TP consensus complete; isolate failed receives.
- Add deployment/preflight/worker checks, explicit NPU roundtrip tests and external FIFO replay tools.
- Preserve all 45 frozen native source files and the p101 algorithm. No blanket BF16 performance parity claim.

## 2026-11-03 — 独立工程整理

- 从 `v65-p101-stage1` 整理 INT8 attention、KV writer、分页布局与 vLLM integration。
- 分离 Python 包、C++/AscendC 源码、CPU / NPU 测试、示例、benchmark 与文档。
- 增加包内 native 资源路径、可迁移的设备库 RPATH、wheel / sdist 和 CPU CI。
- 保留 43 份设备源码、两份 host 源码、校准参数、必要 ABI 和原冻结清单。
- 重新编译两个设备库，SHA-256 与原定版一致；本次未运行 NPU 或服务评测。

算法 / cache 协议版本继续为 `v65-p101-stage1` / `sharedkv-c8-v1`，包版本 `0.65.1`。
