# vLLM Ascend 接入

plugin 沿用已经验证的 DeepSeek V4 / DSpark 接口。需要与原环境匹配的 vLLM Ascend
分支；[安装说明](../../docs/installation.md)列出软件版本限制。完整 vendor 库由环境
提供，本项目不复制 vendor 源码或重建全部 Ascend 自定义算子。

## 启用

先在目标服务环境安装并完整构建本项目。在所有节点、所有 worker 的启动环境中：

```bash
source /path/to/ascend-int8-attention/configs/int8_stage1.env
# 然后使用原本已验证的 vllm serve 模型参数启动。
```

`serve.sh MODEL [原有参数...]` 提供同样的配置封装。它仅在显式执行时启动服务；
本次整理未执行该脚本。TP/DP、expert parallel、模型路径、DSpark speculative 配置、
端口与 scheduler 参数由使用方原有部署命令提供。

已安装的 `vllm.general_plugins` 入口为：

```text
sharedkv_int8 = npu_sparse_attn_sharedkv_int8.vllm_plugin:register
```

| 环境变量 | 当前阶段值 | 作用 |
| --- | --- | --- |
| `SHAREDKV_INT8_MODE` | `int8` | 显式启用 INT8 adapter |
| `SHAREDKV_SCALES` | 冻结 scale 文件绝对路径 | 各层 original / compressed 编码 |
| `SHAREDKV_HOST_QUEUE` | `1` | 调用方 stream 上有序下发 |
| `SHAREDKV_COMPACT_PAGES` | `1` | 使用 packed page allocation 与真实 stride view |
| `SHAREDKV_C8_DECODE` | `1` | 默认 p101 C4/C128 路径 |

env 文件保留调用方已设置的 `SHAREDKV_SCALES`，否则指向仓内附带校准参数。
使用 wheel 独立部署时，也可从已安装包的 `data/scales_v1.pt` 取路径，然后自行设置
这些变量；根目录 `configs/` 不属于运行时 wheel。

## 集成范围

adapter 负责 INT8 cache allocation / views、cache updates、prefill / decode attention
和动态 metadata；plugin 将 DSpark 的 standard SWA draft-context writer 路由到同一
cache 编码。不能同时启用旧 `enable_dsv4_kv_int8` 原型；不支持 context parallel。

保持各节点源码、校准参数和配置一致。只看到模型加载成功不能证明 INT8 路由生效；
独立验收应检查 cache dtype、C4/C128 route 及 prefix reuse。

## 切换

`SHAREDKV_C8_DECODE=0` 只切换到保留的 INT8 兼容实现，cache 编码不变。
彻底关闭 plugin 需在服务启动前 unset `SHAREDKV_INT8_MODE`，并使用 BF16 配置重建
对应 cache / graph。开关不是运行中转换已有 cache 的接口。

设备空闲时可单独验证 adapter：

```bash
env -u SHAREDKV_INT8_MODE -u SHAREDKV_C8_DECODE SHAREDKV_HOST_QUEUE=1 \
  python3 -m tests.npu.test_adapter --compact-pages
```

该用例自己安装 adapter，因此必须 unset `SHAREDKV_INT8_MODE`，防止 plugin 重复安装。
最初目录整理时未执行此命令或端到端服务测试；后续回归见项目的验证记录。

## 安装迁移的输出对比

`tools/check_vllm_equivalence.py` 用固定的公开合成请求比较原安装与新安装，覆盖
两个 DP rank、短回答、代码、多轮上下文、超过 8192 tokens 的输入和 prefix 复用。
请求使用 `temperature=0`、`seed=42`，保存 prompt / generated token IDs、文本、
结束原因和逐 token log probability。服务需要支持 `return_token_ids` 和
`X-data-parallel-rank`。

在项目根目录执行；`VLLM_BASE_URL` 指向待检查服务。每轮使用原部署的缓存重置方式
清空 prefix cache，随后按脚本固定顺序执行请求。先在原安装采集两轮基线：

```bash
python3 tools/check_vllm_equivalence.py record --base-url "$VLLM_BASE_URL" \
  --output validation/serving/baseline-a.json
# 重置 prefix cache 后采集第二轮。
python3 tools/check_vllm_equivalence.py record --base-url "$VLLM_BASE_URL" \
  --output validation/serving/baseline-b.json
```

然后在隔离环境安装新 wheel，保持模型、并行方式、阶段变量和服务参数一致，切换服务，
重置 prefix cache，再采集候选结果并比较：

```bash
python3 tools/check_vllm_equivalence.py record --base-url "$VLLM_BASE_URL" \
  --output validation/serving/candidate.json
python3 tools/check_vllm_equivalence.py compare \
  --baseline validation/serving/baseline-a.json \
  --baseline-repeat validation/serving/baseline-b.json \
  --candidate validation/serving/candidate.json \
  --output validation/serving/output-diff.json
```

默认比较要求文本、输入 token IDs、输出 token IDs 和结束原因完全一致。
提供 `--baseline-repeat` 时，每个候选请求必须与预先采集的某一轮同项基线完全一致；
报告单独列出基线自身发生变化的请求。逐 token log probability 差值用于观察数值波动，
不作为生成文本的一致性替代。验证完成后恢复原有服务环境与启动配置。

若基线对完全相同的 prompt token IDs 在不同 DP rank / cache 阶段产生不同措辞，
可显式加入 `--allow-observed-prompt-variants`。此模式仍要求候选输出的完整 token IDs、
文本和结束原因与某条预先记录的相同输入基线完全一致；报告分别列出同项严格匹配数和
跨 rank / cache 阶段的已观测输出匹配数。新措辞或不同输入均不能通过此模式。

本检查用于安装迁移兼容性，不替代模型质量或性能评测。默认地址、机器路径及原始服务
配置不会写入结果文件；运行日志与原始结果仍应放在忽略的 `validation/` 目录中。
