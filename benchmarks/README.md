# 性能测量

`bench_attention.py` 复用定版 `tests.npu.test_capacity`，避免重写计时方法改变结论。
需要源码中的真实激活 fixtures、匹配的 BF16 vendor op，以及空闲的验证设备。
所有以下命令都使用 NPU；本次整理没有执行。

```bash
env -u SHAREDKV_INT8_MODE -u SHAREDKV_C8_DECODE SHAREDKV_HOST_QUEUE=1 \
  python3 -m benchmarks.bench_attention --output benchmark-results/capacity.json

env -u SHAREDKV_INT8_MODE -u SHAREDKV_C8_DECODE SHAREDKV_HOST_QUEUE=1 \
  python3 -m benchmarks.bench_attention --heldout --output benchmark-results/heldout.json
```

该协议同时保留 BF16、原 INT8 compatibility 和当前核心等角色；先独立数值检查，再
通过预热后的 NPU graph 按交替顺序计时，记录样本和中位数。默认全部角色 LSE OFF；
需要计时 LSE 时使用 `--time-lse`，使每个角色同时开启。

`--case ID` 过滤用例；`--control-device /path/to/libsharedkv_c8.so` 可提供冻结对照库。
输出文件需使用新路径，脚本拒绝覆盖现有结果。其余边界矩阵保留在 `tests/npu/`。

激活样本来自 11 个真实模型层，长上下文和大 batch 使用真实行重采样扩展几何。
这不保留完整原始长上下文 Q/KV 相关性，也不构成模型质量或端到端吞吐验收。

已有主矩阵 331 例中 8 例超过 BF16+5%，最差约 19.8%；同一 binary 在独立库和
捕获上下文中的计时差异仍未解决。不得减去 A/A 偏移、挑选有利样本，或将 eager /
simulator 时延替代 warm graph 测量。详见 [验证与限制](../docs/validation.md)。
