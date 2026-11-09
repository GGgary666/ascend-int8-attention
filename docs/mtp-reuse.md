# MTP 最终实现与验证边界

最终保存的两个 G2 组件位于 [extras/mtp](../extras/mtp/README.md)：INT8 MTP 和
BF16 MTP 均复用同一请求内两个 query 的同序物理 KV，KV 驻留供 QK/PV 使用。
它们是独立、手动接入的源码组件；维护版 p101 和默认 vLLM plugin 保持原有入口。

两者都要求调用方证明物理索引同序，不能只根据 query 数、batch 或序列长度判断。
INT8 保留 Q/P 双残差及原 N128 online 数值阶段；BF16 保留 FP32 softmax/累积。
构建和输入生命周期契约见组件 README，源码摘要见
[source-identity.json](../extras/mtp/source-identity.json)。

## 已有 kernel 验证

BF16 G2 在60个采样/留出 shape 和30份真实输入回放中与原 BF16 输出逐位一致，
另外通过96次新参数绑定、96次图数据更新。INT8 最终路线具有同样的60/30输入覆盖。
真实输入各有15份来自完整 prompt 后、15份来自 chunked-prefill 边界。

| 比较集合 | BF16 复用版相对原 BF16 降时 | INT8 相对 BF16 复用版 |
|---|---:|---:|
| 60个采样/留出 shape | 77.4% | 慢约9.8% |
| 30份真实输入 | 72.9% | 慢约13.5% |

以上为原验证的几何平均，逐 case 时延见
[kernel-comparison.csv](../release/mtp/kernel-comparison.csv)。因此旧 INT8 对旧 BF16 的
3–4倍差距，包含 KV 复用和数据搬运实现的差异，不能全部归因于 INT8 算术。

## 已有 MemCache 对比

原 BF16、最终 INT8、BF16 复用版完成过三组顺序部署，每组90条计时请求，共270条。
使用同3篇真实文档、128固定输出、并发1/4/8以及冷请求/跨节点 MemCache 命中。
缓存 GET 与 token 计数经过检查；32个 worker 的实际加载身份也有原验证记录。
聚合数据见 [memcache-comparison.csv](../release/mtp/memcache-comparison.csv)。

这些数据来自三次顺序部署，生成轨迹、推测接受率及动态 batch 不完全相同；
不能当作固定计算工作量的随机交错因果实验。kernel 倍率不能直接当作整网倍率，
目前不能宣称任一版本在所有端到端场景都更快。

主 C4/C128 的真实输入相对完整 BF16 的既有精度失败未由 MTP 组件修复；
本记录不代表整网模型质量发布通过。源码迁移保持设备代码身份，目录整理阶段
只做 CPU、文件和发布检查，没有重新执行 NPU 或端到端测试。

## 暂停状态

按用户要求，2026-11-09暂停后续优化与设备测试，先提交已保存的实现。
探索性质的混合精度候选、消融分支、调试代码及原始 profile 不进入仓库；
探索过程另行提供下载文档。恢复优化和端到端采集需要用户后续指令。
