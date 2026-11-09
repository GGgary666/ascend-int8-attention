# 冻结压力数据回放

此工具使用外部数据集目录中的 `payloads.jsonl.gz`、`requests.jsonl` 和
`run_plan.json`。源码不包含原始提示词、身份数据或生成输出。输入格式对应本轮
324 请求的冻结 Thursday trace；不会改写输入长度、输出预算、原始 FIFO 或 DP 路由。

```bash
python -m pip install '.[benchmark]'
python -m benchmarks.memcache.read_dataset "$DATASET" --verify
python -m benchmarks.memcache.replay --dataset "$DATASET" \
  --precision int8 --run-id "$UNIQUE_RUN_ID" --base "$API_BASE" \
  --driver-lock "$VALIDATION_DRIVER_LOCK" --capacity-host "$META_HOST" \
  --output "$NEW_RESULT_DIRECTORY"
```

在计时前为两个精度运行同样的独立合成 warmup，保留首用编译记录。使用隔离测试池，
在每个精度的正式回放前清空本地缓存和测试池；回放期间不清缓存、不发送其他请求、
不开启 profiler、不运行构建或重 CPU 分析。不要清理其他服务使用的池。

每轮使用新的 run ID，避免跨轮复用。父请求必须完成后才允许相应子请求；原始队首
阻塞保留。仅当客户端和两个 DP 都确认没有运行或等待工作时跳过空闲时间；状态未知
时不跳过。任何父请求失败都会停止新提交并等待当前请求结束。

输出包括逐请求完整 token、结束原因、输入摘要、TTFT/TPOT、命中 token、提交顺序、
每 2 秒引擎指标和原生池占用。summary 检查完整输出、顺序、因果关系、两 DP 成功
计数、token 计数以及本地＋外部命中与 usage 一致；任一必要检查失败即退出非零。

有效 payload 使用原生 `used_bytes - fragmentation_bytes`，必须先通过已知大小对象
校准 fragmentation 的含义。成功 PUT/GET 字节是传输累计量，不能代替池内当前占用。

计时结果包含回放中首次遇到新 shape 时的开销。历史无 MemCache 报告只有在权重、
框架、编译、调度与输入口径都匹配时才能用于直接因果比较。TTFT、TPOT 与吞吐按
场景分别报告；总体吞吐提高不能覆盖某场景的退化。
