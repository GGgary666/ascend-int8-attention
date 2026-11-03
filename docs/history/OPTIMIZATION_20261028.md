# V65 容量扩展后的通用优化

2026-10-28，本机维护仓保留本轮 r33 候选。设备库 SHA256：
`bf1555047533cc4ab5a91d2c9919b6db266ade2d2954f2b3fd93613da5f6f2dd`。

保持 Q/P 双残差、INT8 Cube QK/PV、C8 KV ABI，以及无固定 B/T 容量上限。
主要改动是常驻 scale/sinks、短 C128 联合 Q 编码及 softmax/P 编码、G4 尾组成本与
线性/交替波次比较、经完整 metadata 验证的直接索引、短 C4 N128/splits1。
所有选择来自活跃 workload、tile 容量和24个物理核，没有 case/batch 白名单。

短 C128 联合 softmax 改变 P 的量化时机。独立联合整数 oracle 和 BF16 检查均通过，
门槛不变；不宣称该分支与旧版本逐位相同。常驻 KV 和 host/device 参数 ABI 未增加。

同一二进制完成两轮 54+24 例，覆盖 B1–512、query容量9–16384、上下文129–135000。
相对本轮开始时冻结 V65：C128 B512/q1 约快26%，B128/q6 快12%–14%，
C4 B512/q1 快7%–9%，C4 B9/q1 快16%–21%。

**性能目标尚未全部达到：**原54例仍有11–12例慢于BF16超过5%；扩展24例还有7例。
mixed31 相对旧V65新增约5%–6%回退；C128 B67/q3 在两轮中分别慢5.87%/3.52%。
C4/C128 B512/q1 仍约为BF16的2.15–2.25倍/1.95倍，不能宣布全shape性能优势。
历史14/54与本轮同进程冻结对照16/54分开记录，没有跨时段择优比较。

全部78例正常输入通过全量BF16检查，最大relative L2=0.01076877，最低逐头cosine=0.999824。
原19例、42个动态graph状态、queue、adapter和默认策略通过；CPU调度广泛枚举通过。
12个合成稀疏边界用例对独立整数/FP64公式通过，但内部洞/空选择/超因果slot重复
与vendor BF16遍历语义不同，不称为BF16通过。同步审计候选全部保留。

最终10份msopprof证据确认主要收益在Vector、标量与搬运；mixed31剩余回退伴随
IB等待上升。profile计数不作为稳态计时，也不相加推断总延迟。

- 完整报告：`historical-artifacts/regressions_20261028/REPORT.md`（未随仓库发布）
- 78例两轮逐项性能：`historical-artifacts/regressions_20261028/PERFORMANCE.md`（未随仓库发布）
- msopprof摘要及原始路径：`historical-artifacts/regressions_20261028/PROFILES.md`（未随仓库发布）
- 同步候选与buffer协议：`historical-artifacts/regressions_20261028/SYNC_AUDIT.md`（未随仓库发布）
- 复现方式：`historical-artifacts/regressions_20261028/REPRODUCE.md`（未随仓库发布）

只增量编译本地C8目标，未重编vendor或host。未运行端到端，未启动服务，Node1仍是此前版本。
历史基线见 [CAPACITY.md](CAPACITY.md)；失败探索代码/数据均留在外部实验目录。
