# 安装与构建

## 环境

CPU reference 和 CPU 单测仅需要 Python >=3.11、PyTorch 2.10；本次整理使用 Python
3.12。设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` 可阻止 PyTorch 自动发现本机安装的
设备插件，保证 CPU 检查不导入 torch_npu。

NPU 路径沿用原验证组合：Ascend910_9382 / DAV2201、CANN 9.2.0、
torch `2.10.0+cpu`、torch_npu `2.10.0.post7`、vLLM `0.30.0+empty`、
vllm-ascend `0.30.0rc2`。vLLM 两项是现有本地分支的版本标记，不能据此保证
PyPI 同版本具备相同的 DeepSeek V4 / DSpark 内部接口。

[environment.json](../release/environment.json) 记录现有分支 commit 与关键 API 文件哈希。
其中 vLLM Ascend 有 9 个本地 tracked 改动，含旧 INT8 prototype；这些并非本项目新要求的
补丁，不应自动恢复或叠加启用。接入仍以匹配的实际 API 和关闭旧 prototype 为前提。

## 从源码使用

在已配置原有 torch / torch_npu 的环境中，避免安装命令更换依赖：

```bash
python3 -m pip install --no-deps --no-build-isolation -e .
source /usr/local/Ascend/cann-9.2.0/set_env.sh
./build.sh
python3 tools/check_release.py --include-build
```

同一环境仅安装本项目的一个副本。安装路径影响 vLLM plugin 发现；服务正在使用的
副本不应在运行中重建或覆盖。最初目录整理只构建新目录；后续真实服务回归在各节点
隔离安装 wheel，通过切换服务比较输出，保留原有全局安装。结果见
[验证记录](validation.md)。

`build.sh` 仅构建两个设备目标和两个小型 host bridge。默认并行度为 2，可使用
`SHAREDKV_BUILD_JOBS` 调整，`SHAREDKV_CMAKE` 可指定 CMake。CANN 路径优先读取
`ASCEND_HOME_PATH`，也可用 CMake 的 `ASCEND_CANN_PACKAGE_PATH` 指定。

## 增量构建

```bash
# 设备侧，不构建 host bridge。
./build.sh --device-only

# 只编译默认 C8 核心，再将核心库和 bridge 更新到包内。
cmake --build build --target sharedkv_c8 -j2
python3 tools/build_host.py --component core
```

完整构建的四个库放在 `npu_sparse_attn_sharedkv_int8/_native/`；运行时按包路径
加载，不依赖当前工作目录或兄弟工程。host bridge 用 `$ORIGIN` 找到同目录设备库。
`_native/manifest.json` 记录构建输入与库的 SHA-256。

`--device-only` 只更新 `build/`，不会让已安装的包切换到新设备库。需要运行新库时，
显式执行 `tools/build_host.py` 完成 host 构建和 staging。

## wheel 与源码包

```bash
# 先完整构建，再生成带四个本地库的 wheel。
./build.sh
python3 -m pip wheel --no-deps --no-build-isolation . -w dist

# 将生成的 wheel 安装到临时目录，只运行 CPU reference，不更改当前安装。
python3 tools/check_distribution.py dist/npu_sparse_attn_sharedkv_int8-0.65.1-cp312-cp312-linux_aarch64.whl

# 包含设备源码、测试样本和文档的源码交付包。
python3 setup.py sdist --dist-dir dist
```

包含 native 库的 wheel 使用 Python ABI / OS / CPU 架构标签，不标记为通用纯 Python
wheel。安装到相同 CANN / torch_npu ABI 的环境；其他环境从源码重新构建。
未执行 `build.sh` 时生成的 wheel 只具有 CPU reference 功能。

sdist 保留构建所需源码、校准参数、测试与文档；本地 `_native` 和 build 缓存不应
作为源码依赖。源码及 wheel 的独立解包检查记录在 [validation](validation.md)。
