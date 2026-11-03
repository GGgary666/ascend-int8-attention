# V65 默认实现交付记录（2026-10-28）

> 以下是默认化/清理交付时的历史快照。后续解除 B8/T48 限制的实现和验收见
> [容量扩展记录](CAPACITY.md)，下述设备库哈希不代表扩展后的库。

本轮完成默认化和目录整理，未修改设备数学、tiling、调度或同步指令。
`PreparedAttention` 与 vLLM INT8 adapter 共用默认选择规则；未设置
`SHAREDKV_C8_DECODE` 即启用受支持的 V65 decode，设 `0` 回退。
`SHAREDKV_INT8_MODE=int8` 仍是启用 vLLM INT8 KV/attention 的总开关。

## 交付结构与备份

- 项目根目录普通文件从 157 个减至 19 个；kernel `.cpp` 从 80 个减至 16 个。
  V65 有 3 个 TU；其余 13 个提供 KV writer、prefill/draft 和兼容 variant ABI。
  这些兼容路径是运行依赖，不能作为废弃实验删除。
- 核心目录约 17 MiB，包含编译缓存、测试和约 9 MiB 的真实激活小样本。
  构建仅保留两个设备库和两个 host bridge；生成器、独立消融、旧实验目标及
  72 份冗余构建产物移出维护目录，完整副本在备份中。
- Node0 原完整项目约 37 GiB，通过同文件系统 rename 保存，60,819 个目录项有
  清单，原目录 inode/device 未变。Node1 独立历史目录约 9.4 GiB、6,021 个目录项
  另行完整备份，保留只存在于 Node1 的原始采集。
- 两台机器分别保存本地备份，归档标识为 `v65_before_default_20261028T100041`。
  备份未随仓库发布，机器路径已移除；历史文件身份见 [索引](archive.json)。
  历史二进制的绝对 RPATH/CMake 缓存须按备份 README 处理。
- `data/scales_v1.pt` 保留原值；`results/scales_v1.pt` 是兼容软链接。
  历史原始数据完全移出核心；维护运行、构建和默认测试不读取备份。

## 等价性与构建

35 个保留的设备 `.cpp/.h` 文件与备份逐字节一致。

| 对象 | 验证 |
| --- | --- |
| V65 设备库 | 整库 SHA256 与接受验收的版本完全相同 |
| 兼容设备库 | 保留的 26 个 AIC/AIV ELF image 全部不变；移除 56 个实验 image |
| C8 host bridge | 源码保持不变，重新编译并验证 200-byte ABI |
| 通用 host bridge | 删除无调用者的 probe/row/dual 导出，保留 attention variant 0..11、writer 与队列语义 |

V65 设备库 SHA256：
`698a8ae6d674c70ed651d9e4edaeaa9a7c8d81dd3195b7eb0f01bb1bfbb108ca`。
兼容库 SHA256：
`cd660affe8a4d9c96f6e2960a690209db048819de0302ed778e9ab4c8f75e6d0`。

清理时发现历史 `c8_launch.o` 残留已撤销的 `pair_wide` 符号，虽源码已恢复但目标文件
时间戳更晚。已在 `tools/build_host.py` 加入源码/头文件内容签名：不匹配时仅重编译
对应 host 目标。修复后重复 `./build.sh` 的两个设备目标及两个 host 均报告 no work。

## 本轮验证

所有 NPU 用例串行运行，未设置 `SHAREDKV_C8_DECODE`，启用有序 host queue。

| 检查 | 结果 |
| --- | --- |
| 真实激活 19 个代表 shape | 整数残差 oracle、BF16 精度阈值和 buffer guard 全通过 |
| 16 个 V65 shape 对冻结 V65 库 | 同地址输出/LSE bitexact；每图 256 调用、9 轮交替计时 |
| 图重放延迟相对冻结 V65 | 中位 -0.0125%，最差 +0.3195%，最好 -0.3016% |
| 同一 graph 更新活跃 metadata | 14 个状态通过，包含 batch=0、异构长度、NaN padding/scratch |
| CPU 成本调度 | 20,000 个随机负载通过任务唯一性、成本界、AIC/AIV 一致性检查 |
| 默认/回退策略 | unset/0/1/非法值、9 个 ABI 边界、显式 V65 和 bitexact 回退通过 |
| 队列生命周期 | 3 个 ratio × 2 个 stream × 32 次重绑，以及 writer 可变行数通过 |
| 实际 vLLM adapter | compact/普通两种 page 布局，写入、重绑、队列及 graph 全通过 |

Node1 已同步相同运行源码、scale 和四个库，并独立通过默认/回退策略及 compact
page adapter 单测；两节点均安装 editable 版本 `0.65.0`。未启动推理服务。

原始本轮验证现位于阶段归档：`historical-artifacts/stage_release_v65_p101/before/validation/`（未随仓库发布）；设备等价性为 `device_code_equivalence.json`，
计时/精度为 `attention.json`，整体机器可读记录为 `delivery.json`。
便于随源码保存的简版记录为 [v65_default_delivery.json](evidence/v65_default_delivery.json)。
独立测试小样本每层保留一份，共 11 层；本轮 19 个样本不是历史 207 个输入的全量重跑。
性能数字衡量默认化前后变化，不能替代新的 INT8/BF16 全量或端到端结论。

历史 [kernel](reports/v65_kernel.md) 与 [双机端到端](reports/v65_e2e.md) 结果继续有效
且保留其局部回退；本轮不重复端到端。同步静态审计仍保留原始 16 个红线、5 个高级别
候选及数据流证据，见 `evidence/c8_v65_sync_*` 和历史报告；未修改同步，也未声称
本次文件整理构成完整同步正确性证明。
