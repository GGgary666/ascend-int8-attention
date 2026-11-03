# V65 容量扩展（2026-10-28）

本页是容量扩展完成时的历史基线，设备库为下文的 `b1ba6051…`。
后续通用优化、当前设备库和剩余回退见 [当前优化记录](OPTIMIZATION_20261028.md)。

已解除 `batch <= 8`、`tokens <= 48` 的入口限制。符合现有 C4/C128 ABI 的
decode、prefill、较大 graph 均走 `kernels/c8`，不再因 B/T 超限转到旧 kernel。
保留 Q/P 双残差、INT8 Cube QK/PV 和 frozen-channel INT8 KV 编码。

## 实现

- `c8_attention.supported` 接受正 B/T，不再设固定容量门槛。仍要求 H8/D512、
  C4/C128、SWA128、block32、C4 top-k1024；R1、显式 draft SWA slots、其他 ABI
  仍使用兼容实现。显式 rollback 开关仍保留。没有宣称所有 ABI 已合并。
- 原调度的 `ends[8]/costs[8]` 已替换。最多两波任务只缓存非空请求，最多48项；
  更大的工作量使用24个物理核的循环负载计数。存储量不随 B/T 增长，且不再使用
  只能编码两项6-bit任务索引的重排表示。旧两波配平仍用于它适用的范围。
- C128 在同一核心内增加 G4/N256，按 G1/G2/G4 的峰值核负载选计划。
  G4 计入更大的 M/Vector 代价和请求尾部，仅在成本严格更低时选择。
  无 shape 名称、batch 白名单或 benchmark 查询表。
- 准备阶段先按实际选中的实现分配 scratch。归并区最多为12个活跃 query
  各预留4份，最大798720字节；大 graph 回放小 batch 仍能使用 split-KV。
  主 workspace 最大9056256字节。没有增加常驻 KV 数据或残差容量。
- 只增量构建本地 `sharedkv_c8` 的3个 TU；未编译 vendor 全部 Ascend 算子。

最终设备库 SHA256：
`b1ba605117bd732074c661ca3e332d529a2ff5328286a94871f5a2d303ffd249`。
200字节 host/device 参数 ABI 不变。

## 依据与平台

平台为 Ascend910_9382 / DAV2201，CANN9.2。参考本机 vLLM Ascend
`8fde6d4e335fe300e2b949aab0e31659cce0811d` 的
`csrc/attention/sparse_attn_sharedkv_metadata/op_kernel_aicpu/sparse_attn_sharedkv_metadata_aicpu.cpp`
中按请求前缀及任务成本分配工作的实现，以及 sharedkv SCFA 的预加载机制。
对应 host tiling 明确接受 ASCEND910B/ASCEND910_93；未从目录名推断架构。
没有引入新的 CANN API。G4 的 L0C128KiB、UB184KiB 上限由编译期断言检查。

采用 `ascendc-perf-optimize`、arch22 性能最佳实践、`ops-profiling`、
`ops-perf-profile-feature-extractor` 和同步审计技能的独立诊断流程。
性能验收沿用已确认的同输入 graph replay/NPU Event 协议；msopprof 用于诊断，
不把其采集耗时混入稳态计时，不把聚合 pipe 时间相加解释为总延迟。

## 验证范围

1. 22000组 CPU 调度检查：覆盖 B256/T8192 范围，任务映射无重复/遗漏，
   G1/G2/G4 峰值成本上界和 AIC/两路 AIV 计划一致。
2. 原有19个 NPU 场景通过。可安全调用冻结旧 V65 的16个场景，同输入同输出地址
   结果 bitexact；9轮交替 graph 计时，最差 +0.0676%，中位 +0.0040%。
3. 新增54个容量场景（主矩阵52例，加2例 G4 奇数尾组）：B9/16/23/32/48/49/64/128/256/512，q1/q5/q6、异构长度，
   query容量49/64/96/128/144/192/256/384/768/1024/4096/8192/16384；
   上下文129–135000。均断言实际进入 variant20/21/22，未因容量回退。
4. 所有54例使用真实 cache writer、622592字节 page stride、完整页表宽度。
   INT8 KV 写入与 CPU 编码逐位相同。小例全部 query 对独立整数残差 oracle；
   大例采样完整分组核验整数 oracle，同时对全部 query 核验 BF16 和内存保护区。
   BF16 对照最差 relative L2=0.010767，最低逐头 cosine=0.999824；既定门槛不变。
5. 同一 B128/T768 graph 的24个状态通过，包括大→小→空→大 batch、异构长度、
   `-1`/合成请求 padding、穿插零-query metadata，输出和 scratch 先填 NaN。
   捕获后的参数及地址不变，每个状态重放3次，保护区未被覆盖。
6. 默认/显式回退策略、有序队列生命周期、真实 vLLM adapter 的 B16/T96
   紧凑 page view、双 stream 和 graph capture/rebind 通过。

激活来自已有11层真实模型小样本；通过重采样扩展几何，使用冻结校准 scale。
不是重新截取完整长上下文模型前向，不能由此推断端到端模型质量。
BF16 对照使用同一 BF16 源数据，INT8 对照共用量化 KV、输入、输出和 workspace 地址。
BF16 cache 为连续页，INT8 使用真实紧凑 page stride；完整口径在测试源码中保留。

## 性能结论与边界

54例均比较最终候选、原兼容 INT8、BF16，并加入“只扩容、未加 G4”的 r2
冻结设备库做同地址消融。原有 B8/T48 冻结 V65 只在其安全范围内调用。

相对 r2，所有54例最差 +2.86%，没有超过5%的新增回退；G4 选中的较长 query
场景改善约9%–33%，T256–16384 的主要 prefill 场景改善约19%–33%。
但相对 BF16 仍有14/54例慢超过5%，相对原兼容 INT8 有3/54例慢超过5%。
因此，容量已解除不等于所有大 batch 都具备性能优势，不宣称已满足全 shape
“只有少数轻微回退”的长期性能目标。

完整逐例数据、54例对照表、msopprof 原始结果和未采用的尝试均保存在：

历史附件：`historical-artifacts/capacity_20261028`（未随仓库发布）

其中 `capacity_final.json` 保存主矩阵，`group_tail_final.json` 保存补充尾组；
`REPORT.md` 汇总完整计时/精度表格与 profile 分析。
请求边界缓存、单-query 双 KV 预加载和代码布局尝试未进入最终源码；前者出现
C4 回退，后者虽改善部分 G1 场景，却增加 G2 指令缓冲等待。原始失败记录保留。

同步算术主体和事件代码与扩容前一致；仅扩展同一实现的 G4 实例与调度。
静态扫描仍报告原有16个红线、5个高级候选，完整保留在 `sync_audit_final.json`；
flow analyzer 未新增候选。它们未被判为无效，也没有宣称形式化同步证明。

Node1已备份并同步相同源码/设备库。默认策略及 B23/q6 C4、T1024 C128
单算子复核通过。首次裸 SSH 的 BF16 对照因 tiling 未注册失败；补齐进程启动前的
custom OPP/driver 路径后通过，失败日志保留，未改 SDK 或编译 vendor。

端到端任务保持暂停。本轮没有启动服务或运行端到端评测。
