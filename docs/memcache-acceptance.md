# MemCache 接入收尾验收

最终版本为 `0.65.1+memcache.10`，搭配注册时完整块传输计划的 vLLM-Ascend 补丁。功能与容量验收已完成；逐场景 TTFT、TPOT、吞吐全面不退化的目标仍未达到。

## 保留的关键改动

- 以完整对象状态、实际长度和 TP 共识保护恢复；失败接收报告完成并结束请求，取消等待实际传输终止。
- 减少同步接收前后的空调度、重复 DP 元数据通信及无用 SAS 元数据构造；不绕过原 scheduler 或页引用保护。
- 注册时按真实 compact tensor view 预计算完整块计划，276 个组件合并为 125 段；部分页保留分段路径。
- 保留任务队列 2、启动 capture helper 与 static/Super 配置；提前 shared MLP 和可选通信实验未合入。

p101、Q/P 双残差、INT8 Cube、FP32 累积、冻结 scale、45 个 native 源文件及编译顺序保持不变。

## 正确性与容量

插件 CPU 36 项、框架 CPU 832 项通过，另有 12 项原有跳过。32 worker 的实际包、协议及 native 身份一致。27 条完整 token 检查中，18 条匹配稳定结果，9 条匹配相同完整输入的独立无 MemCache 已观察变体；未用文本相似代替 token 一致。

真实模型字节、4096/8192 恢复边界、124 次延迟 PUT、取消、非主 TP 故障、五类 GET/info 故障和 PUT 失败重试通过。整理后 NPU 入口另外通过新进程首 GET、四种传输组合、完整 backing/guard、双 stream、graph replay，以及 R1/C4/C128 恢复后 attention。

测试 pool 副本原有 24 小时自动退出导致一次重启命中检查失败；失败记录保留。启动器改为等待明确停止信号后，重新建池、生产、保留池重启，首次 GET 在任何成功 PUT 之前命中 8192 token，完整输出匹配独立控制。

当前版本从空池逐级达到每池 32、64、128 GiB。最终实际对象占用分别为 130.307 和 130.131 GiB，均低于 70% 回收高水位。统计来自经已知对象校准的原生容量接口，payload 与 allocation padding 单列；未用 RSS、预留量或累计 PUT 替代。

## 性能与交付边界

性能采集与正式计时分开。profile 确认过同步接收前的冗余 dummy、重复 metadata 通信等问题；局部 API、kernel 区间存在重叠，不把耗时相加当作端到端收益。提前 shared MLP 的 prefill profile 较快，但 decode 图同时变化且完整矩阵存在回退，因此未采用。

同配置 BF16 对照中仍有 15 个主指标格未通过，详见 [逐场景数据](../release/memcache/fixed-scenario-performance.csv)。本次 324 请求压力结果及历史无 MemCache 参照见 [压力报告](memcache-pressure-results.md)；整体吞吐不能覆盖这些未通过项。

干净 upstream 的补丁应用、幂等和输入身份拒绝已验证；NPU 服务使用已验证的本地框架 native 构建。不能将 CPU patch 检查解释为验证了所有 upstream native 构建组合。

测试池与 Meta 已释放，原 argv、环境、工作目录恢复，两个 DP 的真实请求通过。源码、参数化部署、设备检查和脱敏报告已整理，尚未上传 GitHub。

证据：[验收摘要](../release/memcache/acceptance.json)、[profile 与选择依据](../release/memcache/optimization-evidence.json)。
