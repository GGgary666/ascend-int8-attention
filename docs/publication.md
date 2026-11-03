# 发布检查

在项目根目录检查待提交的源码：

```bash
python3 tools/check_publication.py
python3 tools/check_release.py
bash tools/test_cpu.sh
```

`check_publication.py` 仅使用 Python 标准库，检查机器路径、IP 地址、内部域名、
邮箱和常见凭据格式。它读取 `.pt` 的 pickle 字符串元数据，不反序列化对象，
不导入 torch_npu、不初始化设备。检查失败会返回非零退出码，报告文件位置及信息类型，
不重复输出匹配到的私密值。默认 CPU 检查已包含此扫描。

源码扫描排除本地构建缓存、`_native/`、`dist/`、安装元数据和运行结果；这些目录
已在 `.gitignore` 中排除。通用的 CANN 安装位置及 `/path/to/` 示例占位符保留。
发布 wheel 或源码包时，另行检查实际包内容，包括嵌套的 `.pt` 元数据：

```bash
python3 tools/check_publication.py /path/to/package.whl
python3 tools/check_publication.py /path/to/source.tar.gz
```

## 本次清理

- 历史报告、归档索引和证据 JSON 中的机器绝对路径改为逻辑标识；文件系统 inode / device
  标识已移除。`historical-artifacts/` 表示未随仓库发布的历史附件，不作为本地访问路径。
- 历史数值结果与原始附件 SHA-256 保留。未发布附件的旧链接改为文字引用。
- `scales_v1.pt` 仅修改 `provenance.model`，保存模型标识，移除本机位置；125 组 scale
  的 storage 字节及其他 payload 字段保持不变。
- 43 份设备源码、2 份 host C++ 源码和原始 `upstream-v65-p101.json` 保持不变。

校准文件的原始哈希、公开文件哈希及 scale 数值摘要见
[publication.json](../release/publication.json)。当前源文件身份以
[source_manifest.json](../release/source_manifest.json) 为准；原始冻结清单不用于
冒充脱敏后文件的字节身份。
