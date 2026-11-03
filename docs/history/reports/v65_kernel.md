> 历史验收记录（V65 转为默认之前）。本文件内的 `results/`、旧脚本及报告路径
> 相对于未附带的历史项目根目录；其机器位置已移除，归档标识见
> [历史索引](../archive.json)。其中 opt-in 描述由当前 README 的默认策略取代。
> 数值记录保留，当前目录不包含这些实验脚本。

# C8 v65：代码整理、动态调度与泛化验证

kernel、接入验收和双机端到端正式计时均已完成；端到端数字与局部回退的补充诊断
单列于 `E2E_C8_V65_REPORT.md`。本文件聚焦代码整理与单算子证据。

## 实现范围

维护入口为 `c8_attention.py`、`host/c8_launch.cpp` 和 `kernels/c8/`。
保留 Q/P 双残差、INT8 Cube QK/PV、FP32 softmax/累积、原有 frozen-channel
`sharedkv-c8-v1` KV 编码和 200-byte 参数 ABI。未增加 KV 残差缓存。

平台核验为 Ascend910_9382、DAV2201、CANN9.2、24 AIC/48 AIV。没有采用 A5
MicroAPI，也没有以目录名称代替设备/API 核验。对 AIC/AIV 混合 kernel 使用超过
物理组合数的 blockDim 方案，按已检索到的官方9.1混合调用约束保守排除，未在设备上尝试。
9.2在线对应页未能取得；本机9.2的正向兼容性证据来自24-block增量编译、实机及图重放验证。

支持边界为 H8/D512、C4/C128、SWA128、物理 block32、C4 top-k1024、T<=48、
metadata batch<=8。显式 draft SWA slots、较大 prefill/T96 和其他 ABI 保留原入口。
这些约束来自模型/缓存 ABI 与 buffer 容量，不是 benchmark case ID 或长度白名单。

主要变化：

- 将公共数学、gather、Cube/Vector 流水、调度和 host ABI 分开；实验性 row-factor、
  task-map 等未启用分支不进入维护核心。历史实现保留用于回退与结果复现。
- 不依赖 CPU 固定 query/长度数组，不回读 NPU metadata。图重放时根据当前活跃请求
  决定 G1/G2、N128/256/512 和有界 Split-KV，允许同一张图跨长度/批量边界。
- 固定 block/window/top-k 除法与 H8/G1 的 16 行 Cube、每 AIV 4 个 head，在受保护
  入口中表达为编译期常量，减少无效标量工作。
- 对超过24个 query 的异构两轮工作，把最长任务独占核，剩余短任务按相反成本顺序
  配对。只有保守最忙核成本上界严格改善时才换序，均匀/等成本情况保留线性分配。
  稳定成本排名只使用最多8个请求的局部标量信息，不创建调度 tensor 或附加 kernel。
- 保留独立的小容量 C4 代码镜像。仅调整大型函数顺序曾引发可测的 IB 回退，不能把
  文件合并本身当作优化目标。队列 host bridge 复制 launch 参数，保证异步 rebind 生命周期。

构建仅涉及 `sharedkv_c8` 三个 device translation units 和独立 C8 host bridge；未全量
编译 Ascend 算子。最终 device SHA256 见 `results/c8_v65_acceptance.json`。

## 参考与优化依据

先检查 vLLM Ascend 的 sharedkv SCFA、metadata 的成本配平和 arch22 indexer 实现，
再按 `ascendc-perf-optimize` / arch22 FlashAttention best practices 建模。
当前上游 SCFA blob 为 `3d35e2ab46e5d142e8bda8e92a944a767c25605c`，metadata blob 为
`5fdddd7eaf1896ab10ea5d60fe03cd1d5755bffb`。

- [vLLM Ascend sharedkv SCFA](https://github.com/vllm-project/vllm-ascend/blob/main/csrc/attention/sparse_attn_sharedkv/op_kernel/arch32/sparse_attn_sharedkv_scfa_kernel.h)
- [官方核函数调用与混合 blockDim 约束](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/maintenref/basicdataapi/atlasopapi_07_00236.html)
- 本地技能：`ascendc-perf-optimize`、`ascendc-performance-best-practices`、
  `ascendc-docs-search`、`ascendc-sync-audit`、`ops-profiling`、`ops-precision-standard`。

四层工作中，本算子不包含卡间通信，因此 Step2 不适用；Step1 对 QK/PV 的
M/N/K、L1/L0/UB/workspace 分别建模，Step3 处理任务/核间配平，Step4 处理固定
ABI 标量开销、代码镜像与 Vector/搬运等待。保留现有低阶 MMAD 流水和事件协议。
增加 N512/G2、调度扫描剪枝是负面实验，已归档并从候选路径移除。

## 同输入回归

证据为 `results/c8_v65_{known127,holdout64,confirm16}_final.json`，汇总为
`results/c8_v65_summary.json` / `results/c8_v65_comparison.csv`。
同一测试进程共享 Q/metadata、INT8 workspace/partial/output 地址，使用生产 KV
writer 字节、实际 page stride 和完整页表宽度，比较原版、v63、v64、整理版及 BF16。

| 集合 | 数量 | 说明 |
|---|---:|---|
| 原计划 | 127 | 先前32+95，包括其他 batch、长上下文、边界和 draft |
| 扩展集 | 64 | B1..8、非整齐长度、异构/partial query |
| 异构确认集 | 16 | 2..6 query/请求，最大133333上下文；256/512重复深度 |

207/207 通过原有 BF16 和额外 KV 量化误差门槛。33个改变分块的用例还通过对应
整数双残差 oracle，输出/LSE 共66份数组通过独立精度 skill；没有放宽阈值。
NumPy 无原生 BF16，独立检查先验证导出 FP32 的低16 bit 全零，再恢复 BF16 类型
元数据供原 skill 比较，未重新舍入数值、未改变容差。

按各 case 的实际 graph/eager 执行方式，207例本轮都快于 BF16，延迟比中位数0.7630，
最差0.9571。设备重复重放单列时最差1.0134，未出现>5%回退。
C4 的94例实际模式延迟比中位数0.7694，C128 的94例为0.7812；另外19例为保持原入口
的 ratio1 draft 路径，不能把这些 fallback 当作新 C8 实现的独立收益。
仅统计维护核心覆盖的188个C4/C128用例，实际模式相对BF16延迟中位数0.7751，
相对v64为0.9735，相对原版INT8为0.8819。这样不把19个未改变的draft入口计入核心收益。

相比 v64，实际模式中位数比0.9760，但不是每例都提高。两例短 C128 eager
(B1/q1、B1/q4) 分别增加约1.91/1.93 us，约8.6%/8.4%；仍比 BF16快约50%/45%。
设备计时共有5例比 v64 慢>5%。完整失败/回退清单保留在 CSV，不能只展示获益点。
v64 是需要 CPU 固定 schedule 的 standalone 对照，不能直接代替支持动态元数据的图入口。

额外验证：同一张图的14种 live batch/长度/空 batch/异构状态，NaN padding 和
poisoned workspace；32次 queued producer/rebind 与两个 stream；真实 vLLM 页视图、
cache writer、24次 queued attention、新指针与 graph replay；11种 ABI 边界回退。
CPU 编译实际调度代码的2万组输入检查任务覆盖、唯一性、成本上界及 AIC/AIV 一致性。

64/16集最初发现退化后用于修正调度，最终是回归集，不再称为未参与调优的独立 holdout。
真实历史激活行按几何扩展，不能等同于全新相关长模型前向；固定地址 graph 性能也不能
替代持续轮换 KV 的服务结果。

## msopprof 因果证据

最终6份匹配采集和紧邻调度版本的2份消融采集均通过完整性审计，频率均1800MHz。
3个最终 C8 capture 的所有 AIC 均有整数 Cube 指令，整数指令数等于总 Cube 指令数，
浮点 Cube 指令计数为0，见 `results/c8_v65_int8_cube_counters.json`。
源数据：`results/msopprof_c8_v65_pair_final*`、`results/msopprof_c8_v65_pair_ablation_before*`；
汇总：`results/c8_v65_profile_summary.json`。根据日志的实际 launch 顺序与同名 kernel
序号关联 case；CSV 导出时间不代表执行顺序。

C128 B8/31 query 异构长上下文，仅从紧邻的循环移位调度改为短任务配对：

| PMU指标 | 修改前 | 修改后 |
|---|---:|---:|
| 最忙 AIV 总计时/us | 46.172 | 40.381 |
| 该 AIV Vector/us | 7.898 | 5.477 |
| 所有核最大 Vector/us | 7.898 | 6.884 |
| 该 AIV 等待 QK 结果/us | 18.769 | 15.766 |
| 该 AIV 等待 PV 结果/us | 10.394 | 8.266 |
| 该 AIV Scalar/us | 7.455 | 8.040 |

配平增加了一些标量工作，但降低最忙核的 Vector 工作及结果等待。独立256/512深度
计时将该场景从比 BF16慢约10%变为更快。PMU 管线计数相互重叠，表中各项不可相加；
这不是逐指令时间线，也不用其 Task Duration 代替稳态性能门槛。

## 同步与交付边界

`results/c8_v65_sync_candidates_final.json` 原样保留16条红线、5条高级候选；
`results/c8_v65_sync_flow_final.json` 的 flow findings 为0。没有据此声称消除所有同步风险。
数量比早期减少来自删除不支持的死路径，不能当成修复事件协议的证据。
本轮没有删除活跃路径的 flag/barrier，也没有根据正则候选盲目插入可能不对称的信号。

新的调度只在 splits=1 时启用；Cube 与其两个 Vector partner 使用同一个映射，
workspace 继续由物理 core 索引。稳定成本排名把输出唯一分配给单任务或任务对；
Split-KV 的 producer/partial 地址和全核 barrier 可达性保留原协议。
原始候选及后续逐项复核仍是交付限制，动态/队列测试不构成形式化同步证明。

端到端结果在同目录的 `E2E_C8_V65_REPORT.md` 中单独给出。当前候选仍通过
`SHAREDKV_C8_DECODE=1` 启用；未将实验结果自动发布为全平台默认实现。

## 启动观测修正

首次 INT8 启动已形成567058-token缓存和图，但自定义模块 INFO 日志没有进入 vLLM
的日志树，导致严格路由证据检查失败。该尝试未发送计时请求，进程已清理，原始日志
保留在 `results/e2e_c8_v65_int8/`。只把 adapter 两处 logger 名改为
`vllm.sharedkv_c8`，kernel 二进制完全不变；真实 adapter/队列/graph 验证再次通过，
并确认日志可见。精确替换和新旧 SHA 记录在 `results/c8_v65_logger_fix.json`。
BF16 已完成的结果保留，INT8 重试目录为 `results/e2e_c8_v65_r1_int8/`。
