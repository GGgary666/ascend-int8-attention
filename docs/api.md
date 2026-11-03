# Python API 契约

## SharedKVCase

`SharedKVCase` 是输入容器；attention 前已完成 RoPE。D 固定为 512。

| 字段 | 形状 / 类型 | 语义 |
| --- | --- | --- |
| `q` | BF16 `[T,H,512]` | 连续的 query；默认核心 H=8 |
| `ori` | INT8 `[P,B,1,512]` | original KV，最后两维连续；允许额外 page stride |
| `ori_block_table` | INT32 `[batch,logical_pages]` | logical page → physical page |
| `cu_q` | INT32 `[batch+1]` | 每个请求在 Q 中的累计 query 边界 |
| `seq_lens` | INT32 `[batch]` | 每请求当前有效 KV 长度 |
| `sink` | FP32 `[H]` | sink logit，参加 denominator，无 value contribution |
| `cmp` / `cmp_block_table` | 与 original 相同的 KV / table 形式 | 独立 compressed cache |
| `cmp_indices` | INT32 `[T,1,topk]` | C4 的 compressed logical token 索引，负值表示 padding |
| `ori_indices` | 可选 INT32 `[T,1,K]` | 显式 original physical slot，走所需兼容入口 |
| `ratio` | `1` / `4` / `128` | R1 / C4 / C128 |
| `window` | 默认 `128` | original sliding window |
| `softmax_scale` | Python float | 默认 `1/sqrt(512)` |

所有设备 tensor 必须位于同一个 NPU。`cu_q` / `seq_lens` 与 page table 的值需由
调度方保证有效；构造器校验 dtype、shape、stride，不逐项回读设备 metadata。

## PreparedAttention

```python
operation = PreparedAttention(
    case, original_scale, compressed_scale,
    implementation="auto", return_lse=False,
)
output = operation.run()
```

两个 scale 为冻结的连续 FP32 `[512]` tensor。输出 BF16 `[T,H,512]`；可通过
`output=` 提供预分配连续输出。`return_lse=True` 时通过 `operation.lse` 读取
FP32 `[T,H]`，`run()` 仍只返回 output。

`auto` 默认路由到 C8；`fallback` 显式使用保留的 INT8 兼容入口；`v65` 要求非空输入
满足 C8 ABI，否则报错。`SHAREDKV_C8_DECODE=0` 关闭自动 C8 路由，保持 INT8 cache
协议不变。H8/D512、C4/C128、SWA128、block32、C4 top-k1024 是默认核心边界。

构造与 scratch 分配发生在 graph capture 之外。对象持有输入、输出与 workspace
引用，`run()` 不进行 tensor 分配或设备数据回读；其生命周期需覆盖全部 graph replay。
更改 geometry 需重新 prepare；动态 metadata 的允许变化由所选 kernel 契约约束。

## PreparedCacheWrite

```python
writer = PreparedCacheWrite(cache, updates, slots, inverse_scale)
writer.run()  # 返回同一个 cache
```

| 参数 | 契约 |
| --- | --- |
| `cache` | INT8 `[pages,page_size,1,512]`，允许 page stride |
| `updates` | BF16 `[T,512]` 或 `[T,1,512]`，每行连续 |
| `slots` | INT32/INT64 `[T]` physical slot，或 `[T,2]` 的 `[page,offset]` |
| `inverse_scale` | 冻结、连续 FP32 `[512]`；使用 `CacheQuantization.inverse_scale` 提前生成 |

负值和越界目标被跳过。同一次写入中有效 slot 必须唯一；重复写同一位置没有确定的
并行顺序。writer 使用调用方当前 stream，可与 attention 在有序 host queue 中串联。

## CacheQuantization 与 reference

`CacheQuantization` 校验 scale、提供量化/反量化和离线校准。生产 scale 在填充 KV
之前冻结。`sharedkv_reference()` 是 FP64 浮点参考，`tiled_quantized_reference()`
是保留 Q/P residual 的 CPU 整数参考；均可从 `reference` 模块导入。

示例中的随机数据校准仅用于演示。模型推理应使用与实际权重及层名对应的校准文件。
