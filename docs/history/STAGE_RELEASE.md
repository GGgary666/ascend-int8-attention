# V65-p101 阶段交付

**选择 p101 作为当前综合最优、可复现的阶段基线**，标识 `v65-p101-stage1`，
Python package `0.65.1`。默认入口继续使用该版本。这里的“综合最优”依据现有
精度、广泛 shape、配平和稳定性证据；性能目标尚未全部完成。

## 决策与保留内容

| 内容 | 阶段决策 |
|---|---|
| Q/P 双残差、INT8 Cube QK/PV、FP32 softmax/累积 | 保留；沿用已通过的独立整数和 BF16 精度门槛 |
| `sharedkv-c8-v1` frozen-channel INT8 KV | 保留；KV 不存 residual，writer 和 attention 共用编码 |
| batch/query 容量 | 无 B8/T48 准入限制；按实际 metadata 和资源选择任务粒度 |
| 通用成本调度、两波任务配平 | 保留 p101；AIC/双 AIV 共用任务归属，不使用 shape 名单 |
| 受限 KV 驻留、紧凑中间结果和活跃行处理 | 保留；共用 L1 资源计划，可复用 KV 槽，超出资源继续 INT8 分块 |
| p102–p110 候选 | 不合入；局部收益伴随其他 shape 回退或证据不稳定，已明确撤回 |

p101 的 B31 配平收益有独立证据：graph 约快25%，最慢 AIC 约43.5→31.0µs。
后续 p110 虽将 B97 两次测至 BF16+5%以内，却使 S32、B31 变慢，因此保留完整的
p101，而不拼接各实验最快的局部路径。证据入口见[优化索引](OPTIMIZATION_20261029.md)。

## 交付结构

核心 `kernels/c8/` 已按调度、metadata、资源计划、Cube/Vector、gather和数值原语分工，
只有3个编译单元。保留父目录13个编译单元用于 cache writer 和实际需要的兼容 ABI；
它们不是废弃实验。各模块职责见 [kernel说明](c8_kernel.md)。

本次整理收敛版本命名、默认配置、文档和验收入口，压缩过期维护说明，将旧validation
产物移出核心。全部43份设备源码与p101一致，设备库和指令图不变，未为目录美化
调整编译单元或函数顺序。历史源代码、所有失败实验/profile完整保留在外部目录，
详见[归档索引](archive.json)。运行、构建和默认测试自包含。

## 已有精度、性能与边界

[阶段证据摘要](evidence/stage1_summary.json)保留来源及哈希，不把不同采集口径混算。

| 已有检查 | 结果 |
|---|---|
| 主矩阵331个不同case ID | 数值检查通过；最大 BF16 relative L2 0.01118177，最小 head cosine 0.99965868 |
| 原54、扩展24、边界23各两次 | 对应记录中均无 BF16+5% 回退 |
| 更广边界/容量合计331例 | 主记录8例超过 BF16+5%，最差19.77% |
| sparse91、metadata views12、同图任务切换24、padding20 | 已通过对应数值与guard检查 |

八个主记录回退包括极短窗口的大batch、宽mixed query、960-key资源边界，以及
C128 B97异构长度。后续独立诊断还观察到mixed129等上下文相关波动；不能将
“8/331”当作所有运行条件下的保证。相同二进制在不同库/捕获上下文仍有差异。
B97同图任务分析支持其部分回退发生于kernel内部，尚未证明全部根因。

测试使用少量真实激活重排扩展的几何负载；并非完整模型质量评测或端到端部署验收。
当前版本仅在 **Ascend910_9382 / DAV2201、CANN9.2** 及本机 torch_npu/vLLM Ascend
组合验证。不支持 context parallel；其他硬件/API组合需要独立验证。
Q/输出为BF16，H8/D512、C4/C128、SWA128、block32、C4 top-k1024属于核心ABI。
显式draft slots、R1、其他头数等输入仍使用所需的INT8兼容入口，不转为BF16 attention。

整数输出 `rtol=.008, atol=.0003`、LSE `2e-5`，BF16 relative L2 `<.04`、
min head cosine `>.999`。这些门槛未降低。原始同步审计候选全部保留，
通过数值检查不等于完成形式同步证明。

## 使用、冻结与复现

```bash
cd /path/to/historical-checkout
source /usr/local/Ascend/cann-9.2.0/set_env.sh
source configs/int8_stage1.env
python3 tools/check_release.py --include-build
```

配置默认开启 INT8 KV + INT8 attention、有序host queue、紧凑page view；已提供的
`SHAREDKV_SCALES`会保留，否则使用仓内校准文件。更换模型权重需重新校准。
各节点worker均需相同代码/scale配置；此文件不启动或更改服务，不更改并行预设。
不可叠加vLLM旧 `enable_dsv4_kv_int8` 原型。显式回滚仍可使用 `SHAREDKV_C8_DECODE=0`。

[release/manifest.json](../../release/upstream-v65-p101.json)记录完整源码、校准、测试样本及四份本地库
的身份。校验工具仅用Python标准库，不初始化NPU。源包可在具备相同依赖的环境重建；
`--include-build`核对的是此环境下冻结的精确产物，host库含本地RPATH，不宣称二进制可移植。

只修改核心时增量构建 `cmake --build build --target sharedkv_c8 -j3`。
全新源码目录的首次本地构建使用 `./build.sh`，仅生成本项目所需的两个设备库及host桥。
源码、冻结预编译产物和本轮回归记录另有独立交付包，路径列于[归档索引](archive.json)。

本次只执行必要的独立算子/入口回归及产物一致性检查，不重跑性能矩阵或端到端，
避免把同一设备镜像的计时波动误记为整理收益。验收结果见阶段证据摘要中的
`stage_validation`记录。后续新优化应作为独立候选，与此冻结版本比较后再决定升级。
