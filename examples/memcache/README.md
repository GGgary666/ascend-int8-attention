# 双节点 MemCache 部署

先阅读 [支持范围与协议](../../docs/memcache.md)。在两节点匹配已验证的 CANN、Python、
torch_npu、vLLM、vLLM-Ascend 和 MemCache 环境；为本次部署建立独立目录。
所有路径和节点地址由调用者提供。原参考部署目录不需要改动。

1. 检查 `/etc/ascend_install.info`、`/usr/local/Ascend/driver/version.info`。
   如果平台容器未挂载，从对应宿主机复制真实文件，不应自行构造内容或覆盖不匹配版本。
2. 在隔离框架副本中按 `patches/vllm-ascend/README.md` 应用补丁；安装并构建本插件。
3. 对冻结权重计算内容清单，再生成协议清单。两节点使用同一份清单，并完整核验内容。

```bash
python tools/memcache/freeze_model.py --model "$MODEL" --output "$MODEL_MANIFEST"
python tools/generate_memcache_manifest.py \
  --model-content-manifest "$MODEL_MANIFEST" \
  --connector-contract "$FRAMEWORK/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/cache_compatibility.py" \
  --scales "$PLUGIN/npu_sparse_attn_sharedkv_int8/data/scales_v1.pt" \
  --output "$CACHE_MANIFEST"
```

这里 `PLUGIN` 是隔离安装或源码目录，内部含 `npu_sparse_attn_sharedkv_int8/` 和已构建
的四个 native 库。不要在服务运行时重新构建此目录。完整权重摘要只在部署准备阶段
计算，不在启动 worker 或每次请求时计算。

每节点生成配置，`NODE_RANK` 分别为 0、1，`LOCAL_IP` 使用该节点通信地址。
`DP_ADDRESS`、`META_HOST` 两节点一致。`DEPLOYMENT` 必须是新目录。

```bash
python tools/memcache/prepare_deployment.py \
  --model "$MODEL" --manifest "$CACHE_MANIFEST" \
  --framework "$FRAMEWORK" --plugin "$PLUGIN" --output "$DEPLOYMENT" \
  --meta-host "$META_HOST" --dp-address "$DP_ADDRESS" \
  --local-ip "$LOCAL_IP" --interface "$INTERFACE" --node-rank "$NODE_RANK"
python tools/memcache/preflight.py --deployment "$DEPLOYMENT" --output "$PREFLIGHT"
```

配置固定为每池 256 GiB、client 0GB、device_sdma、DP2/TP16/EP、135000 context、
8192 chunk、16 sequences、DSpark5、compact INT8。默认 API 仅监听 localhost；需要
外部访问时显式指定 `--listen-host`。仅在测试环境通过 `--validation-endpoints`
启用前缀清理等调试端点。

在源 shell 中加载 CANN / ATB 后启动。Meta 只运行一份，pool 每节点一份，model
每节点一份。以下命令均在前台运行，可交给已有服务管理器记录 PID、日志和退出状态。

```bash
source /usr/local/Ascend/cann-9.2.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1
python tools/memcache/run.py meta --deployment "$DEPLOYMENT"
# 两节点分别执行，使用独立终端或服务单元：
python tools/memcache/run.py pool --deployment "$DEPLOYMENT"
python tools/memcache/run.py model --deployment "$DEPLOYMENT"
```

记录实际 model 父进程 PID。服务 ready 后，每节点检查当前 16 个 worker，再将两份
preflight 与两份 worker 报告放到验收节点进行对比。

```bash
python tools/memcache/audit_workers.py --deployment "$DEPLOYMENT" \
  --parent-pid "$MODEL_PID" --output "$WORKER_REPORT"
python tools/memcache/compare_nodes.py --preflight "$PREFLIGHT_0" "$PREFLIGHT_1" \
  --workers "$WORKERS_0" "$WORKERS_1" --output "$NODE_CHECK"
```

`preflight --metadata-only` 仅适用于先前已经完整核验且保持冻结的权重；该结果明确标注
未重算权重内容，不能代替首次完整双节点预检。实际部署清单与报告含本地路径，保留在
部署目录，不提交到 Git。

## 设备验收

在模型 worker 停止、设备空闲且测试池运行时执行。加载环境并使 `PYTHONPATH` 指向
本项目和配套框架，`MMC_LOCAL_CONFIG_PATH` 指向 client 配置。指定独立输出目录。

```bash
export SHAREDKV_HOST_QUEUE=1 TASK_QUEUE_ENABLE=2
python -m tests.npu.memcache.test_roundtrip --mode write --device 0 --model "$MODEL" --output "$BYTE_OUTPUT"
python -m tests.npu.memcache.test_roundtrip --mode read --device 0 --model "$MODEL" --output "$BYTE_OUTPUT"
python -m tests.npu.memcache.test_transfer_plan --device 0 --participants 1 \
  --model "$MODEL" --sync-dir "$PLAN_OUTPUT/sync" --output "$PLAN_OUTPUT"
python -m tests.npu.memcache.test_immediate_attention --device 0 --output "$ATTENTION_OUTPUT"
```

write/read 使用两个独立进程，read 的首个操作就是 GET。检查完整 backing 的有效字节、
其他 group、padding、双 stream 与 graph replay；attention 检查继续使用既有数值门槛。
设备测试会删除自己创建的随机前缀对象，不执行全池清空。

## 容量、回放与恢复

容量用 `/api/v1/capacity/usage` 和 `/api/v1/capacity/segment_remaining` 原生接口采集。
分别记录 pool 总量、实际 used、allocation padding、有效 payload 和成功/失败 I/O。
按实际 used 32→64→128 GiB 每池逐档推进，并在每档做跨 DP 读取与完整输出核对。
累计 PUT、RSS 和预留容量都不等于实际对象占用；每池保持低于 70% 回收高水位。

真实压力回放使用外部提供的数据集；原始提示词、身份信息和输出不随源码发布。报告需
保留输入摘要、原始 FIFO、父子请求依赖、DP 路由、完整输出预算、逐请求指标和失败记录。
历史无 MemCache 结果单列版本与配置差异，不能替代当前对照。

切换前由服务管理器保存原 argv、环境、工作目录和 PID 身份。结束测试后，先停止并
等待所有 model worker 和 I/O 退出，再释放测试 pool、停止 Meta，恢复原服务配置。
恢复后检查健康状态并对两个 DP 各发真实请求。避免使用模糊的 `pkill` 影响其他服务。
