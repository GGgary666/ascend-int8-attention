# 验证与已有证据

## MemCache 接入收尾（2026-11）

当前插件 `0.65.1+memcache.10` 的 CPU 36 项及 C++ 调度检查通过；
配套干净框架补丁的 CPU 832 项通过，另有 12 项原有跳过。
整理后的 NPU 字节往返、双 stream / graph、恢复后 attention、32 worker 身份、真实模型
失败隔离、重启首次 GET 和当前版本每池 32→64→128 GiB 实际占用验收通过。

27 条完整 token 检查中 18 条属于稳定结果，
9 条匹配同输入的独立无 MemCache 已观察变体。
两组压力回放各完成 324 请求；逐场景 TTFT、TPOT、吞吐全面不退化仍未通过。
详见 [收尾验收](memcache-acceptance.md) 和 [压力结果](memcache-pressure-results.md)。
以下各节保留对应历史日期的验证范围与结果。

## wheel 真实服务器回归（2026-11-03）

在两个节点隔离安装新 wheel，保留原有全局 Python 安装；保持既有 DeepSeek V4 Pro
权重、TP16 / DP2 / EP32、V1 runner、DSpark5、FULL_DECODE_ONLY 和 INT8 阶段配置。
逐个检查 32 个 worker 的已加载 native 库，确认实际使用新 wheel 中的库。

| 检查 | 结果 |
| --- | --- |
| NPU adapter，紧凑页与普通页 | R1/C4/C128、cache 写入、page guard、双 stream 队列和 graph replay 通过 |
| 原 INT8 安装控制组 | 提前采集两轮，各 16 条请求；同项 15/16 的生成 token 完全一致，长输入存在原有措辞波动 |
| 新 wheel 端到端 | 两轮共 32 条请求，覆盖两个 DP rank、短回答、代码、多轮上下文、9016-token prefill 与 prefix 复用 |
| 同项严格输出对比 | 30/32 条的 prompt IDs、generated IDs、文本和结束原因与某轮同项基线完全一致 |
| 其余两条长输入 | 输出 `006293`；与相同 prompt IDs 在原基线中已出现的输出完全一致，另一种原有措辞为完整英文句子 |
| prefix cache | 四条候选复用请求均命中 8192 个 prompt tokens |
| 设备库与校准数值 | 两个设备库与原安装的 SHA-256 相同；校准 tensor storage 字节保持不变 |
| 原服务恢复控制 | 两节点原 argv / 环境 / 工作目录与 32 个 worker 库路径恢复；另 16 条控制请求通过同一兼容性判据，原版本也复现短措辞 |

当前默认 CPU 检查为 20 项 Python 测试及 C++ 调度检查，全部通过；C++ 覆盖
22,000 组随机负载、84 组 uniform / boundary 负载和到 B512 的成本枚举。

结论为**安装迁移的基本输出兼容性通过**，不宣称 32/32 同项严格 token 一致。
单算子输出继续使用 `rtol=.008, atol=.0003`，LSE 使用 `rtol=atol=2e-5`；
未放宽数值精度门槛。模型质量和性能不在本次验收范围。

完整脱敏记录见 [serving-validation.json](../release/serving-validation.json)；
复现步骤见 [vLLM 输出对比](../examples/vllm/README.md#安装迁移的输出对比)。

## 最初目录整理（仅 CPU）

按用户要求，本次验证仅使用 CPU，不初始化 NPU、不执行设备用例、不启停服务。
原工程保持原样，新工程在独立目录构建。

| 检查 | 作用 |
| --- | --- |
| `tools/check_release.py` | 校验当前源码清单，并将 43 份设备源码 / 2 份 host 源码与原定版逐文件比较 |
| `--include-build` | 校验构建输入及四个库身份，并检查两个设备库与原定版完全相同 |
| `tools/test_cpu.sh` | Python 数学/编码/打包契约，加 C++ 调度边界测试 |
| `tools/check_publication.py` | 检查待发布源码、分发包和 `.pt` 元数据中的机器信息 |
| CPU example | R1/C4/C128 的独立整数残差 reference |
| wheel / sdist 解包 | 核对包内资源与 native 库，离开源码目录后验证 CPU import / reference |

本次 CPU 结果：16 项 Python 检查通过；C++ 调度检查覆盖 22,000 组随机活跃负载、
84 组 uniform / boundary 负载及到 B512 的 uniform-cost 枚举。

两个重新构建的设备库 SHA-256：

```text
libsharedkv_c8.so   3484f5fb08b26fde5b23880398cd4fd6f8980c63742fc31c428fd16cb881643e
libsharedkv_int8.so cd660affe8a4d9c96f6e2960a690209db048819de0302ed778e9ab4c8f75e6d0
```

host C++ 源码保持一致；链接参数改为通过 `$ORIGIN` 加载同目录设备库，因此 host 库
不再要求匹配原工程含绝对 RPATH 的哈希。新的构建身份记录在 `_native/manifest.json`。
本次检查的汇总位于 [整理验收记录](../release/organization-validation.json)。

## 可运行的 CPU 检查

```bash
python3 tools/check_release.py
bash tools/test_cpu.sh
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -m examples.reference_attention --ratio 128
```

默认 `pytest` 只收集 `tests/unit/`。CPU 用例覆盖 scale 合法性、rounding / clipping、
追加 token 不改变冻结编码、联合 softmax / sink、R1/C4/C128 整数残差与 FP64 参考、
源码和校准身份、关闭 plugin 时的包导入。C++ 用例验证保留的调度策略。

## NPU 回归入口

这些脚本是定版用例迁移后的入口，尚未在新目录重新执行；它们使用设备 1。
需要在用户其他测试结束后独立安排，不能作为 CPU 检查的一部分自动运行。

```bash
env -u SHAREDKV_INT8_MODE -u SHAREDKV_C8_DECODE SHAREDKV_HOST_QUEUE=1 \
  python3 -m tests.npu.test_policy
env -u SHAREDKV_INT8_MODE -u SHAREDKV_C8_DECODE SHAREDKV_HOST_QUEUE=1 \
  python3 -m tests.npu.test_attention --output validation/attention.json
env -u SHAREDKV_INT8_MODE -u SHAREDKV_C8_DECODE SHAREDKV_HOST_QUEUE=1 \
  python3 -m tests.npu.test_dynamic --output validation/dynamic.json
env -u SHAREDKV_INT8_MODE -u SHAREDKV_C8_DECODE SHAREDKV_HOST_QUEUE=1 \
  python3 -m tests.npu.test_queue
```

metadata views、sparse edge、retained capacity、packed query 和 task-batch 边界用例均
保留在 `tests/npu/`。adapter 的环境要求见 [接入文档](../examples/vllm/README.md)。

## 继承的定版验收

原冻结清单原样保存在 `release/upstream-v65-p101.json`；公开版校准文件的路径脱敏、
文件哈希变化及 scale 数值身份单独记录在 `release/publication.json`。历史摘要入口为
[stage1_summary.json](history/evidence/stage1_summary.json)。

| 原有证据 | 结论 |
| --- | --- |
| 331 个不同 case ID | 数值检查通过；最大 BF16 relative L2 0.01118177，最小 head cosine 0.99965868 |
| 主矩阵性能 | 8 例比 BF16 慢超过 5%，最差 19.77% |
| sparse 91、metadata views 12、动态切换 24、padding 20 | 对应原记录的数值 / guard 检查通过 |
| 后续诊断 | 仍有上下文相关计时波动，不将 8/331 解释为所有条件保证 |

保留原门槛：整数输出 `rtol=.008, atol=.0003`，LSE `2e-5`，BF16 relative L2
`<.04`，min head cosine `>.999`。没有因为目录整理降低门槛。

历史 `v65_kernel.md` / `v65_e2e.md` 属于对应时间点的实验记录；服务器路径已清理，
未附带的实验数据使用逻辑标识引用。本次未重新验证模型质量、完整端到端性能或新环境兼容性。
静态审计和数值通过也不等于完成形式同步证明。
