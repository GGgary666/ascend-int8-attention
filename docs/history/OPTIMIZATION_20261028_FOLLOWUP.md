# V65 通用优化续轮：b17

本轮降低了大batch短上下文的绝对耗时，但**仍未达到只有少数轻微 BF16 回退的目标**。
原54例两轮仍有11/12例比 BF16 慢超过5%；额外24例为8/7例，新23个边界例为20/20例。
没有进行端到端测试或Node1部署更新。

当前设备库 SHA256：
`6e0bcfea8774ee8d8f45ed104cf4c526bfca3beb1085e2261b059c01d66d361d`。
对照是本轮开始时冻结的 r33：
`bf1555047533cc4ab5a91d2c9919b6db266ade2d2954f2b3fd93613da5f6f2dd`。
V65接口和默认路由不变；b17仅是本轮候选标识。

## 保留的改动

`kernels/c8/task_batch.h` 为 C4/C128 共用短任务实现，按L1容量和真实压缩KV列数
选择每核保留的独立任务数，让多个任务共享Vector指令设置和阶段通知。每个请求
的矩阵乘法、scale和softmax仍独立，Q/P双残差、INT8 Cube QK/PV及C8 KV ABI均保留。
没有新增batch/query准入上限，没有按测试名称选择实现，也没有扩大host workspace。

C4短路径用arch22支持的Sort/Extract精确验证完整选择集合，保留重复索引的
multiplicity语义。批处理C4采用共同softmax/P编码，其量化顺序在独立oracle中
显式建模，不声称与旧分阶段P编码逐位相同。精度门槛没有放宽。

不常用的两波异构调度排序移出公共入口；六个旧C128 Vector计算体的地址、长度及
机器码与r33相同。b15/b16出现prefill回退后，撤回新增的公共入口对齐；b17的
prefill256两轮恢复为比r33快1.3%–1.9%，没有给prefill添加特殊分支。

## 两轮同输入结果

以下为耗时降低比例，基准是r33；BF16差距同时列出。

| 场景 | 比 r33 耗时降低 | 仍比 BF16 慢 |
| --- | ---: | ---: |
| C4 B256/q1/S513 | 26.4%–29.2% | 11.8%–14.3% |
| C4 B512/q1/S129 | 30.1%–31.4% | 48.2%–54.9% |
| C128 B64/q1/S8192 | 16.6%–17.1% | 9.4%–10.7% |
| C128 B128/q1/S4096 | 23.0%–24.4% | 19.3%–30.9% |
| C128 B256/q1/S513 | 26.8%–27.1% | 36.9%–40.7% |
| C128 B512/q1/S129 | 29.0%–29.6% | 38.0%–42.0% |

仍有新增r33回退：C128 B11/q1/S511两轮慢5.30%/5.08%，但仍比BF16快约15%；
C4 B9/q5第二轮慢5.27%，比BF16慢0.94%。所有结果均保留，没有取最好的一轮。
测试数据由真实激活行重采样形成，不能替代完整真实长上下文或端到端评估。

最终12份msopprof显示，C4 B512的Vector均值76.50→49.97µs，AIV MTE2/MTE3
从32.86/15.94→15.35/6.13µs；C128 B512的Vector从48.50→41.09µs。
但两者Vector仍分别约为BF16的4.04/3.35倍，q6分组路径也没有获得相应收益。
这些重叠计数不能相加为总耗时，不能据此声称某一等待已解释全部差距。

两轮101个shape及动态graph、稀疏边界、队列、adapter等检查通过。101例最大相对
L2为1.077%，最小head cosine为0.999714；整数残差/LSE门槛不变。
大案例整数oracle为完整量化组抽样，BF16门槛覆盖全部live输出。CPU调度验证包含
22,000个随机负载和84个边界负载。同步审计候选及人工复核完整保留，不声称形式化证明。

## 证据和复现

完整源码实验、失败记录、不可变r33对照和原始采集均位于外部实验目录，生产运行
及构建不依赖它们。当前仅增量编译`sharedkv_c8`，NPU任务串行使用device1。

- 完整结论、限制与剩余方向：`historical-artifacts/regressions_followup_20261028/REPORT.md`（未随仓库发布）
- 101例两轮全量计时：`historical-artifacts/regressions_followup_20261028/PERFORMANCE.md`（未随仓库发布）
- 最终msopprof及原始数据入口：`historical-artifacts/regressions_followup_20261028/PROFILES.md`（未随仓库发布）
- skill、vLLM arch22及CANN API依据：`historical-artifacts/regressions_followup_20261028/SOURCES.md`（未随仓库发布）
- 同步候选及数据生命周期审计：`historical-artifacts/regressions_followup_20261028/SYNC_AUDIT.md`（未随仓库发布）
- 复现命令：`historical-artifacts/regressions_followup_20261028/REPRODUCE.md`（未随仓库发布）

平台仅验证Ascend910_9382/DAV2201/arch22、CANN9.2；未验证其他架构的运行兼容性。
