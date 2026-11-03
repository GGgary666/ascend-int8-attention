# Ascend INT8 Attention

面向 **DeepSeek V4 shared-KV sparse attention** 的 AscendC 实现，包含 INT8
attention、分页 INT8 KV Cache、冻结通道校准和 vLLM Ascend 接入。

当前算法基线：**`v65-p101-stage1`**，Python 包版本 **`0.65.1`**。
Q/P 保留双残差；QK/PV 使用真正的 INT8 Cube，softmax 与累积使用 FP32；
KV 使用 `sharedkv-c8-v1` 对称 per-channel INT8，不保存 KV residual。

## 支持范围

| 项目 | 当前支持 |
| --- | --- |
| 已验证设备 | Ascend910_9382 / DAV2201，24 AIC / 48 AIV |
| 软件基线 | CANN 9.2.0、PyTorch 2.10.0、torch_npu 2.10.0.post7；vLLM 需匹配本地验证分支 |
| 默认 attention | BF16 Q/输出，H8 / D512，C4 / C128，SWA128，block32，C4 top-k1024 |
| KV Cache | original/compressed 分别冻结 FP32[512] scale；分页、带 stride page view、增量 slot 写入 |
| batch / query | 无 B8/T48 准入上限；设备根据实际 metadata 与资源选择切分 |
| INT8 兼容入口 | R1、显式 draft slots、其他受支持 ABI；保留原 variant0..11 |
| 推理接入 | vLLM general plugin、有序 host queue、compact pages、DSpark cache writer |
| 边界 | 不支持 context parallel；其他芯片、模型与软件组合需另行验证 |

这是阶段定版：历史主矩阵 331 个 case 的数值检查通过，其中 8 个 case 比 BF16
慢超过 5%，最差约 19.8%。[验证记录](docs/validation.md)说明证据口径和已知限制。

新 wheel 已完成双节点、32 worker 的真实 vLLM 回归：32 条请求中 30 条与同项基线
文本 / token 完全一致，另 2 条长输入仅在原基线已出现的措辞之间变化；
R1/C4/C128 的真实 NPU adapter 检查通过。详见 [本轮回归](docs/validation.md)。

## 安装与构建

### 环境要求

| 组件 | 当前要求 / 已验证版本 |
| --- | --- |
| Python | >=3.11；本次整理使用 3.12.13 |
| CANN | 9.2.0，默认路径 `/usr/local/Ascend/cann-9.2.0` |
| PyTorch / torch_npu | `torch==2.10.0+cpu`、`torch_npu==2.10.0.post7` |
| vLLM / vLLM Ascend | 本地验证分支，版本标记分别为 `0.30.0+empty` / `0.30.0rc2` |
| 构建工具 | CMake >=3.20、Ninja、C++ 编译器、`setuptools>=68`、`wheel` |

vLLM 的两个版本号来自已验证的本地分支，同版本号的 PyPI 包不能保证具有相同的
DeepSeek V4 / DSpark 内部接口。接入时需匹配实际 API；分支 commit、关键文件哈希
和本地改动数量见 [environment.json](release/environment.json)。

### 从源码安装

在已经配置好上述 torch / torch_npu 和构建工具的目标 Python 环境中执行：

```bash
cd /path/to/ascend-int8-attention
source /usr/local/Ascend/cann-9.2.0/set_env.sh

# 安装 Python 包和 vLLM plugin 入口，保留环境中已有的依赖版本。
python3 -m pip install --no-deps --no-build-isolation -e .

# 完整构建设备库与 host bridge，并检查源码和构建产物的身份。
./build.sh
python3 tools/check_release.py --include-build
```

`pip install` 只安装 Python 包，NPU 路径还需要完整执行 `./build.sh`。四个本地库会
放在包内的 `npu_sparse_attn_sharedkv_int8/_native/`，运行时按包路径加载。
构建在 CPU 上完成，不执行设备 kernel；同一服务环境只安装一个项目副本，
服务正在使用的副本应在停服后再重建或替换。

CANN 安装在其他位置时，先 source 对应的 `set_env.sh`，再将 `ASCEND_HOME_PATH`
设为实际路径。默认构建并行度为 2，可通过 `SHAREDKV_BUILD_JOBS` 调整；
`SHAREDKV_CMAKE` 可指定 CMake 可执行文件。

### wheel 部署

在上述完整构建成功后打包，再将 wheel 安装到各目标节点的服务环境：

```bash
python3 -m pip wheel --no-deps --no-build-isolation . -w dist
# 将下面的路径替换为实际生成的 wheel 文件。
python3 -m pip install --no-deps /path/to/package.whl
```

目标环境需匹配构建时的 Python ABI、CPU 架构、CANN 和 torch_npu；环境不同时从源码
重新构建。未先完整构建的 wheel 只具有 CPU reference 功能。

## vLLM 部署配置

### 启用 INT8 Attention + KV Cache

先在各节点的服务 Python 环境中安装并完整构建本项目，然后在**所有节点、所有
worker 启动前**加载阶段配置：

```bash
source /path/to/ascend-int8-attention/configs/int8_stage1.env
# 接着执行原有 vllm serve 命令，保留模型、TP/DP、DSpark、端口和调度参数。
```

也可以把现有启动命令的 `vllm serve` 替换为
`bash /path/to/ascend-int8-attention/examples/vllm/serve.sh`，模型路径及其余参数保持原值。
安装包通过 `vllm.general_plugins` 注册 `sharedkv_int8`，由 vLLM 自动发现并加载。
在匹配的已验证分支上，通过插件运行时替换内部方法接入，无需额外修改源码文件。

阶段配置设置以下变量：

| 环境变量 | 阶段值 | 作用 |
| --- | --- | --- |
| `SHAREDKV_INT8_MODE` | `int8` | **开启本项目的 INT8 attention 和 INT8 KV Cache** |
| `SHAREDKV_SCALES` | 校准文件的绝对路径 | 各层 original / compressed cache 的冻结通道 scale |
| `SHAREDKV_C8_DECODE` | `1` | 默认 p101 C4/C128 attention 路径 |
| `SHAREDKV_COMPACT_PAGES` | `1` | 紧凑 KV page 分配与真实 stride view |
| `SHAREDKV_HOST_QUEUE` | `1` | 在调用方 stream 上有序下发 |

如果设置了 `VLLM_PLUGINS`，需要在已有插件名单中包含 `sharedkv_int8`；未设置时
由 vLLM 自动加载已发现的插件。`--attention_config.indexer_kv_dtype int8` 控制
indexer cache，`--quantization ascend` 控制模型量化，本项目的启用开关是
`SHAREDKV_INT8_MODE=int8`。

### 校准参数与多节点配置

env 文件默认使用仓内的 `npu_sparse_attn_sharedkv_int8/data/scales_v1.pt`，并保留
调用方已经设置的 `SHAREDKV_SCALES`。使用自己的校准参数时，在 source 前指定：

```bash
export SHAREDKV_SCALES=/absolute/path/to/your_scales.pt
source /path/to/ascend-int8-attention/configs/int8_stage1.env
```

附带参数对应当前 **DeepSeek-V4-Pro-0813-w4a8** 权重，使用 `sharedkv-c8-v1` 协议、
125 组 FP32 `[512]` scale 和 1.1 倍范围裕量。更换模型或权重后需重新校准并验证精度；
采集与冻结方法见 [KV Cache 与校准](docs/kv-cache.md)。

各节点需使用一致的源码、校准文件内容和阶段配置，并保证 worker 能访问对应的绝对路径。
关闭旧 `enable_dsv4_kv_int8` 原型；当前要求 prefill / decode context parallel size
均为 1。

仅使用 wheel 部署时，根目录 `configs/` 不在运行时包内，可直接设置同一组变量。
下面从已安装包中取得随附校准文件路径：

```bash
export SHAREDKV_INT8_MODE=int8
export SHAREDKV_SCALES="$(python3 -c 'from pathlib import Path; import npu_sparse_attn_sharedkv_int8 as p; print(Path(p.__file__).resolve().parent / "data" / "scales_v1.pt")')"
export SHAREDKV_C8_DECODE=1
export SHAREDKV_COMPACT_PAGES=1
export SHAREDKV_HOST_QUEUE=1
```

### 确认生效与关闭

服务执行到相应路径后，日志中的 `Standalone INT8 sharedKV cache` 应显示
`dtype=torch.int8`，`C8 decode route` 表示默认 attention 路由生效；
`Compact INT8 sharedKV` 表示启用了紧凑 page 分配。仅模型加载成功不能证明 INT8
已生效，独立验收还需检查实际请求、cache dtype 和 prefix reuse。

恢复 BF16 时，在启动原有 BF16 服务前执行：

```bash
unset SHAREDKV_INT8_MODE SHAREDKV_SCALES SHAREDKV_C8_DECODE
unset SHAREDKV_COMPACT_PAGES SHAREDKV_HOST_QUEUE
# 使用原有 BF16 启动命令重启服务，重新创建 KV Cache 和 graph。
```

恢复时直接使用原有 BF16 命令；`examples/vllm/serve.sh` 会重新加载 INT8 配置。
`SHAREDKV_C8_DECODE=0` 只切换到保留的 INT8 兼容实现；cache 编码保持 INT8。
这些配置在进程启动前生效，切换时需要重建 cache / graph。

## CPU 示例与 Python 入口

CPU 示例无需 NPU、torch_npu 或 vLLM。在项目根目录执行：

```bash
python3 -m pip install --no-build-isolation -e '.[test]'
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -m examples.reference_attention --ratio 4
bash tools/test_cpu.sh
```

`TORCH_DEVICE_BACKEND_AUTOLOAD=0` 阻止 PyTorch 自动加载设备插件；CPU 检查脚本会
自行设置它。显式运行 `examples.attention`、`examples.cache_write`、`tests.npu.*`、
benchmark 或 vLLM 服务才使用 NPU。最初目录整理仅运行 CPU 检查，后续真实服务器
回归及差异说明见 [验证记录](docs/validation.md)。

公开 Python 入口保留原包名：

```python
from npu_sparse_attn_sharedkv_int8 import CacheQuantization, SharedKVCase
from npu_sparse_attn_sharedkv_int8 import PreparedAttention, PreparedCacheWrite
```

后两个类需要 torch_npu 和已构建的设备库。完整示例见
[attention](examples/attention.py)、[cache writer](examples/cache_write.py)
和 [API 契约](docs/api.md)。

## 工程结构

```text
ascend-int8-attention/
├── npu_sparse_attn_sharedkv_int8/  # Python API、KV 编码、vLLM 适配
│   ├── data/scales_v1.pt         # 当前模型的冻结校准参数
│   └── _native/                 # 本地构建产物；源码中不提交
├── csrc/
│   ├── kernels/c8/              # p101 默认 C4/C128 内核
│   ├── kernels/                 # KV writer 与所需 INT8 兼容内核
│   └── host/                    # 有序 torch_npu launch bridge
├── tests/
│   ├── unit/                    # 默认 pytest：仅 CPU
│   ├── cpp/                     # 调度策略的 CPU C++ 检查
│   ├── npu/                     # 显式执行的设备与图回放回归
│   ├── support/                 # 独立 oracle、BF16 对照与几何构造
│   └── fixtures/                # 11 层真实激活小样本
├── benchmarks/                  # 保留配对图计时协议的性能入口
├── examples/                    # CPU、attention、cache write、vLLM
├── configs/                     # 阶段环境配置
├── docs/                        # 安装、接口、架构、量化与验证
├── release/                     # 原始冻结清单、迁移映射、当前源码身份
└── tools/                       # 构建、完整性校验、CPU 测试
```

目录职责参考 [FlashInfer](https://github.com/flashinfer-ai/flashinfer) 和
[FlashAttention](https://github.com/Dao-AILab/flash-attention) 对 Python 包、`csrc`、
tests、examples、benchmarks 的组织方式。算法与实现来自当前已确定的 Ascend 工程。

## 发布前检查

```bash
python3 tools/check_publication.py
python3 tools/check_release.py
```

发布检查覆盖源码、历史文档和 `.pt` 文件内的序列化元数据，也可把待发布的 wheel
或源码包路径作为参数传入。历史服务器路径已替换为逻辑标识，校准文件仅清理模型路径
元数据，125 组 scale 数值保持不变。检查范围及身份记录见 [发布说明](docs/publication.md)。

## 文档

| 文档 | 内容 |
| --- | --- |
| [安装与构建](docs/installation.md) | 依赖、增量构建、wheel / sdist、迁移 |
| [API](docs/api.md) | attention、KV writer、metadata、返回值与内存生命周期 |
| [架构](docs/architecture.md) | QK → softmax → PV、调度、兼容路径与 host queue |
| [KV Cache 与校准](docs/kv-cache.md) | 编码、slot/page、compact pages、校准流程 |
| [vLLM 接入](examples/vllm/README.md) | 显式启用、每节点配置、DSpark、回滚 |
| [测试与证据](docs/validation.md) | CPU / NPU 分层、精度门槛、已有性能限制 |
| [benchmark](benchmarks/README.md) | 配对图计时、LSE、公平比较与原始样本 |
| [迁移映射](docs/layout.md) | 原目录到新目录的映射与字节一致性 |
| [贡献说明](CONTRIBUTING.md) | 修改边界与验证要求 |
| [发布检查](docs/publication.md) | 路径脱敏、二进制元数据、分发包扫描与校准身份 |

源码整理、测试和构建不依赖工作区的兄弟目录。`docs/history/` 保存脱敏后的历史证据；
其中 `historical-artifacts/` 是未随仓库发布的附件标识，当前操作入口以本 README
和上述文档为准。
项目许可状态见 [NOTICE](NOTICE.md)。
