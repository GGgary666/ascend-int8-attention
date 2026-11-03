# 当前实现与探索证据索引

当前交付为 **V65-p101 stage1**，完整说明见 [STAGE_RELEASE.md](STAGE_RELEASE.md)。
设备 SHA256 `3484f5fb08b26fde5b23880398cd4fd6f8980c63742fc31c428fd16cb881643e`。
p102–p110 均已明确撤回；当前43份 kernel 源码与设备库匹配稳定p101。

以下外部文件用于溯源，不是构建、运行或默认测试的依赖：

- p101主矩阵与容量验收：`historical-artifacts/vector_pipeline_20261029/REPORT_p101_current.md`（未随仓库发布）：331个case ID、8个BF16+5%观察；不能解释为全部shape保证。
- 配平证据：`historical-artifacts/vector_pipeline_20261029/P0_P1_FINDINGS_p101.md`（未随仓库发布）：保留两波任务归属，B31收益伴随最慢AIC43.5→31.0µs。
- A/A稳定性限制：`historical-artifacts/vector_pipeline_20261029/P0_AA_STABILITY.md`（未随仓库发布）及graph任务诊断：`historical-artifacts/vector_pipeline_20261029/graph_stability/REPORT.md`（未随仓库发布）：独立库上下文仍有差异，B97的真实回退不能全部归为计时误差。
- p108：`historical-artifacts/vector_pipeline_20261029/p108_findings.md`（未随仓库发布）：normal-mask缩减setup，B31回退，未采纳。
- p109：`historical-artifacts/vector_pipeline_20261029/p109_findings.md`（未随仓库发布）：max归约按成本选轴，未命中新分支的短shape仍回退，未采纳。
- p110：`historical-artifacts/vector_pipeline_20261029/p110_findings.md`（未随仓库发布）：索引前缀扩展，B97局部收益伴随S32/B31回退，未采纳。
- 指令流水：`historical-artifacts/vector_pipeline_20261029/timeline_capture/REPORT.md`（未随仓库发布）：独立配置模式仿真已完成；自动TimelineDetail仍有地址信息问题。仿真不替代板上graph计时。
- 原始同步候选：`historical-artifacts/vector_pipeline_20261029/p101_sync_candidates.md`（未随仓库发布）：全部保留，未宣称形式同步证明。

更早候选、全部失败日志和profile保存在同一个外部实验目录。
阶段冻结前的完整AGENTS.md、此索引旧版及原validation目录见[归档索引](archive.json)。
新的阶段交付只整理身份、文档、配置和复现入口，kernel算术/调度/同步/指令图不变。
