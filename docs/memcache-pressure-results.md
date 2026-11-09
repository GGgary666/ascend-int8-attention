# MemCache 压力回放结果

每组完整回放 324 请求、27,790,756 输入 token、723,291 输出 token。保留原始全局 FIFO、父子依赖和 DP0 322 / DP1 2 的固定路由，所有请求保留完整生成预算。

输入是已冻结的重建提示词；原始记录大多截断，不能称为原始全文逐字重放。两组使用新的cache salt，经过相同合成 shape warmup 后清空本地与测试池。计时期间未开启 profiler，也未并行运行独立构建或重 CPU 分析；推理触发的首次形状编译仍计入耗时。

| 当前对照 | 完成 | 总时长 s | 输出 tok/s | TTFT P50/P95 s | TPOT P50/P95 ms | 抢占 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16＋MemCache | 324 | 3627.88 | 199.37 | 2.555 / 25.885 | 18.453 / 97.843 | 2 |
| INT8＋MemCache | 324 | 3638.96 | 198.76 | 2.245 / 19.073 | 17.492 / 85.010 | 0 |

本地命中与 MemCache 命中单列，池实际占用来自原生容量接口。逐请求 CSV 不包含提示词、原始身份或完整生成内容；完整原始记录保留在本地测试目录。

| 当前对照 | 本地命中 token | 外部命中 token | 池0峰值 GiB | 池1峰值 GiB |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 11730944 | 11591680 | 26.476 | 26.632 |
| INT8 | 20606976 | 2904064 | 18.058 | 18.279 |

## 当前回放的分组延迟

下表为 INT8/BF16 比值，延迟越小越好。`long_history` 与 `short_history` 是 `continuation` 的子集，标签沿用冻结数据。`cold` 表示没有计划中的历史复用，其中仍可能命中共享模板，因此不等同于完全冷缓存。分组共享执行时间轴，吞吐只报告整轮值。

| 分组 | 请求数 | TTFT P50 | TTFT P95 | TPOT P50 | TPOT P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| cold | 44 | 0.9797 | 0.7142 | 1.0307 | 0.9171 |
| continuation | 280 | 0.8672 | 0.7534 | 0.9556 | 1.1496 |
| long_history | 173 | 0.8058 | 0.6714 | 0.9280 | 0.6569 |
| short_history | 107 | 0.8356 | 0.9909 | 0.9864 | 0.9664 |

本轮仍未通过的分组延迟项如下；完整均值、P50/P90/P95/P99、最大值见 [分组 CSV](../release/memcache/pressure-group-latency.csv)。两种精度各运行一次，这些数值是本轮观测，没有重复运行的置信区间，也不据此豁免未通过项。

| 分组 | 指标 | 统计量 | INT8 相对 BF16 |
| --- | --- | --- | ---: |
| continuation | tpot_ms | p95 | +14.96% |
| long_history | ttft_ms | p90 | +6.51% |
| short_history | tpot_ms | mean | +0.56% |
| short_history | tpot_ms | p90 | +30.97% |
| short_history | tpot_ms | p99 | +15.18% |
| cold | tpot_ms | p50 | +3.07% |

## 历史无 MemCache 参照

以下是 2026-11-03 原报告的实测值，使用较早框架与 static/Super 关闭配置。仅作为历史参照，不能把跨版本差异归因为 MemCache。

| 历史对照 | 总时长 s | 输出 tok/s | TTFT P50/P95 s | TPOT P50/P95 ms | 抢占 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 无 MemCache | 6196.36 | 116.73 | 25.434 / 162.910 | 27.620 / 189.536 | 304 |
| INT8 无 MemCache | 3675.09 | 196.81 | 2.226 / 21.987 | 21.033 / 100.159 | 0 |

## 性能结论边界

固定场景矩阵仍有未通过的 TTFT、TPOT 或吞吐项，不能宣称逐场景全面不慢于 BF16。本压力回放的总体结果不覆盖这些失败项。提前 shared MLP 实验因部分负载回退未合入最终版本。

| 固定矩阵场景 | 未通过指标 | INT8/BF16 比值 |
| --- | --- | ---: |
| 1024/1/cold | ttft | 1.350476 |
| 1024/1/cold | tpot | 1.037604 |
| 1024/1/cold | output_tokens_per_second | 0.921806 |
| 8704/1/cold | tpot | 1.000493 |
| 8704/1/cold | output_tokens_per_second | 0.998066 |
| 8704/1/remote_hit | ttft | 1.096441 |
| 32768/1/remote_hit | ttft | 1.041513 |
| 130000/1/remote_hit | tpot | 1.002090 |
| 8704/4/cold | tpot | 1.042985 |
| 8704/4/remote_hit | ttft | 1.101130 |
| 8704/4/remote_hit | tpot | 1.052897 |
| 8704/4/remote_hit | output_tokens_per_second | 0.948081 |
| 32768/4/remote_hit | ttft | 1.042479 |
| 32768/4/remote_hit | tpot | 1.089559 |
| 32768/4/remote_hit | output_tokens_per_second | 0.913931 |

数据：[完整汇总](../release/memcache/pressure-results.json)、[INT8 逐请求](../release/memcache/int8-pressure-requests.csv)、[BF16 逐请求](../release/memcache/bf16-pressure-requests.csv)、[INT8 原生占用](../release/memcache/int8-pressure-capacity.csv)、[BF16 原生占用](../release/memcache/bf16-pressure-capacity.csv)。
