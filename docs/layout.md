# 目录迁移与来源

来源是已经确定的 `npu_sparse_attn_sharedkv_int8` / `v65-p101-stage1`，不是早期
`dpskv4_sparse_attn_int8` 融合原型。新工程使用完整独立文件，未以软链接或 import
依赖原工程、归档、release 副本或 vendor 副本。

| 原位置 | 新位置 |
| --- | --- |
| 顶层 Python 模块 | `npu_sparse_attn_sharedkv_int8/` |
| `kernels/` | `csrc/kernels/` |
| `host/` | `csrc/host/` |
| `tests/test_*.py` | `tests/npu/test_*.py` |
| `tests/test_schedule.cpp`、stub | `tests/cpp/` |
| `tests/support/`、`tests/fixtures/` | 保留原测试资源职责 |
| `data/scales_v1.pt` | `npu_sparse_attn_sharedkv_int8/data/scales_v1.pt` |
| 原 `docs/` | `docs/history/`，保留历史结论、数值与原附件哈希，机器路径已脱敏 |
| 原 `release/manifest.json` | `release/upstream-v65-p101.json`，字节不变 |
| 原构建目录 | 新工程独立 `build/`；四库 staging 到包内 `_native/` |

43 份设备源码和两份 host C++ 源码逐字节保留，设备编译单元顺序保持原样。
Python 改动包括包导出、相对 import 和资源加载位置；测试适配 import、资源路径、
C++ include 路径，以及公开校准文件的数值身份核验。校准文件仅将模型路径元数据
替换为模型标识，scale 数值不变。默认路由、数值算法、cache 格式和 vLLM adapter
行为沿用定版。

host bridge 的链接位置改为 `$ORIGIN`；未复制原库冒充新构建。本次从源码重新编译，
两个设备库的哈希均精确匹配原定版。

`release/import-map.json` 列出逐文件迁移映射，`release/source_manifest.json` 固定
本目录的源文件身份。前者说明来源，后者用于后续检测意外变化；不覆盖原始冻结记录。

未纳入本目录的内容包括 build 缓存、完整 profiling 数据、服务日志、旧实验候选、
全部 vendor 源码和其他工作负载。当前功能、样本回归和构建无需这些工作区目录。

公开历史文档中的 `historical-artifacts/` 仅标识未附带的历史附件，不是可访问的本地
目录。发布脱敏范围和校准身份说明见 [发布检查](publication.md)。
