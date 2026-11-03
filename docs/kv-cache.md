# INT8 KV Cache 与校准

## 编码协议

协议名 `sharedkv-c8-v1`。每层的 original / compressed cache 各有一个冻结的
FP32 `[512]` scale；K 和 V 共用同一份 shared-KV 行。

```text
inverse_scale[c] = FP32(1 / FP64(scale[c]))
encoded[t,c] = clamp(round(FP32(kv[t,c]) * inverse_scale[c]), -127, 127)
decoded[t,c] = encoded[t,c] * scale[c]
```

采用对称量化、零 offset、nearest-even rounding。scale 在 cache population 前固定，
追加 token 时不重估 scale、不重新编码已有行；KV 不存 residual。
Q/P 双残差只存在于 attention 计算过程。

## 分页和写入

核心 cache 是 INT8 `[pages,32,1,512]`，一行 512 字节，一页有效 payload 16KiB。
允许 page stride 大于 payload；reader 和 writer 都使用真实 stride，不能把有间隔的
view 当作紧凑连续 tensor。

flat slot 按 `page = slot // page_size`、`offset = slot % page_size` 解析；也可传
`[page,offset]` pair。writer 跳过负值或越界 slot，调用方保证有效 slot 唯一。
page table 是 logical page → physical page；C4 compressed indices 是 logical token
索引，显式 draft original indices 则是 physical slot。

`SHAREDKV_COMPACT_PAGES=1` 启用现有 vLLM packed allocation / view 逻辑。
FP32 compressor 状态保持原类型，因此混合 KV pool 的总显存收益应按实际 group
组成计算，不能从单份 KV payload 直接推导“总容量翻倍”。

## 附带校准文件

`npu_sparse_attn_sharedkv_int8/data/scales_v1.pt` 是当前 DeepSeek V4 Pro 权重对应的
冻结参数，125 个 original/compressed scale key，margin 为 1.1。公开版本仅将
`provenance.model` 的机器路径替换为模型标识，125 组 scale 的原始 storage 字节
和其他字段保持不变。公开文件 SHA-256：
`54f6aad6582a65eb921ec65bdd103d2a1abe229c07d55c09f62440075d3d2e7c`。
原文件身份及 scale 数值摘要保存在 [publication.json](../release/publication.json)，
原冻结清单保持不变。

payload 包含 `protocol`、`scales`、`margin`、`observed_rows_across_ranks`、
`source_sha256` 与 `provenance`。`provenance` 保存模型 config / weight index
身份和校准请求记录的哈希；这不是对每一个大体积权重 shard 的重新校验。

scale key 与运行时 layer prefix 一致，如 `model.layers.0.self_attn.ori` 和
`model.layers.0.self_attn.cmp`。更换模型或权重后需要重新采集和验证；附带参数不作为
其他模型的通用量化配置。

## 重新采集的接口

这部分供设备空闲时单独安排，整理和 CPU 回归不会启动采集。
`calibration.CacheObserver` / plugin 的 `calibrate` 模式观察真实 post-RoPE BF16
cache updates，按层、cache kind 和 rank 收集 channel absolute maximum。

1. 在匹配的 BF16 vLLM 部署中设置 `SHAREDKV_INT8_MODE=calibrate` 与
   `SHAREDKV_CALIBRATION_DIR`，使用 `--enforce-eager`。
2. 服务 warmup 后创建采集目录中的 `collect` marker；每个 worker 需能看到自己的
   marker 和输出目录，然后运行代表性请求。
3. 移除 marker，再向各 rank 发送一次请求以刷新统计；各节点的 `rank_XX.pt` 需要
   汇总，避免相同文件名覆盖。
4. 对每个层/kind 逐 channel 合并各 rank maximum，按
   `scale = clamp_min(margin * maximum / 127, 1e-8)` 生成冻结参数，记录源文件哈希、
   请求来源、模型身份和观测行数。检查目标模型需要的全部 key，随后重新验证精度。

运行时 `load_scales()` 检查协议名、shape、dtype、有限性和正值。观测模式会在保存
checkpoint 时同步，不用于性能测量。当前附带 scale 保留原定版文件，未重新校准。
