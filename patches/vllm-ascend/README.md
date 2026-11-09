# 配套 vLLM-Ascend 补丁

基线 commit：`8fde6d4e335fe300e2b949aab0e31659cce0811d`。
vLLM commit：`ced6857afa0ea7b2e3f0846a62e1394e90f15607`。
插件版本：`0.65.1+memcache.10`。版本字符串不能代替实际源码身份。

`0001-compact-int8-memcache.patch` 包含 18 个文件的变化，包括 CPU 契约测试。
`identity.json` 记录补丁以及每个文件应用前后的 SHA-256；所有输入文件与上述干净
upstream commit 一致。本补丁不包含本地分支的其他旧实验或 native 源码修改。

在隔离框架副本中执行，不能修改活动服务正在导入的源码：

```bash
python tools/memcache/apply_framework_patch.py --framework "$FRAMEWORK"
python tools/memcache/apply_framework_patch.py --framework "$FRAMEWORK" --apply
```

工具先核验全部输入和 patch，再运行 `git apply --check`；完整应用后的重复执行是
幂等的。部分应用或其他内容不匹配会拒绝继续。无需修改全局 vLLM 源码：配套插件在
经摘要核验的 engine loop 上安装同步接收进度 hook，其他模型/connector 保留原路径。

完整 CPU 测试在框架目录运行：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH="$FRAMEWORK" \
  python -m pytest -q tests/ut/distributed/ascend_store
```

设备验收使用已经核验的框架 native 构建。本轮实际服务环境继承了原本地框架构建及
其未启用的旧 INT8 原型；补丁文件本身与干净 upstream 对应，不能把 CPU patch 应用
检查解释为已经重新验证所有 upstream 构建组合。其他 native 构建仍需运行同一组
设备、模型、失败和性能验收。

协议清单在应用最终补丁后生成。运行时会验证 protocol source 摘要；不能在生成清单
后继续修改这些文件，也不要在安装失败时删除版本检查。旧算法与设备库身份不变。
