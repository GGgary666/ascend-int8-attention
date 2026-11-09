# 贡献与维护

当前基线是 `v65-p101-stage1`。提交应解释具体问题、行为变化和验证结果。
版本选择和已有性能限制见 [验证记录](docs/validation.md)。

默认检查在 CPU 上执行：

```bash
python3 -m pip install --no-build-isolation -e '.[test,benchmark]'
bash tools/test_cpu.sh
python3 tools/check_release.py
```

`benchmark` 额外依赖用于本地 mock HTTP 回放契约测试，不需要 NPU 或真实服务。
CI 安装这些依赖，避免将回放顺序和失败路径测试因缺少依赖而跳过。

`tests/npu/`、benchmark 和 vLLM 启动均需要显式安排空闲设备。用户正在使用 NPU
进行其他测试时，不执行这些入口，不运行 import 后即访问设备的脚本。

维护设备代码时保留 Q/P residual、INT8 Cube QK/PV、FP32 softmax / 累积和冻结
`sharedkv-c8-v1` 编码。不要删除 parent kernel：它们包含必要的 writer 和 INT8 ABI。
默认 C8 与 adapter 共用路由；保持 page stride、output / LSE、host queue 和 ABI 一致。

目录整理不要调整设备编译单元顺序或函数布局。新性能方案在独立候选中验证，保留
全部配对原始样本；不拼接已撤回 p102–p110 的局部最优路径。不能为使意外改动通过
校验而重写 `upstream-v65-p101.json`。有意修改当前源码时应明确更新当前交付身份，
同时保留原基线及来源映射。

不要覆盖正在被服务使用的 `_native` 库，也不要重建全部 vendor Ascend kernel。
仅改 C8 时用针对 `sharedkv_c8` 的增量构建，再更新包内构建产物。
