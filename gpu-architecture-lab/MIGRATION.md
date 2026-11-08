# CUDA 编程实验室合并说明

`cuda-programming-lab` 的编程语义、源码和历史证据已并入本实验室。统一入口为 [README.md](README.md)，体系内容见 [ARCHITECTURE.md](ARCHITECTURE.md)。不再维护两个平行的内存、同步与异步目录。

## 路径对应

下表旧路径相对于原 `cuda-programming-lab/`，新路径相对于 `gpu-architecture-lab/`：

| 旧路径 | 新路径 | 处理方式 |
| --- | --- | --- |
| `00_environment/` | [00_machine](00_machine/README.md) | 原为空占位；环境、能力与错误检查并入机器主题 |
| `01_programming_model/` | [01_execution](01_execution/README.md) | 原为空占位；host/device、索引和执行映射并入执行主题 |
| `02_language_features/01_setmaxnreg/` | [03_instruction/01_setmaxnreg](03_instruction/01_setmaxnreg/README.md) | 完整移动；语言限定符、设备函数与编译链接由上层指令主题承接 |
| `03_memory/01_cache_operators/` | [02_memory/01_cache_operators](02_memory/01_cache_operators/README.md) | 完整移动 |
| `03_memory/02_eviction_priority/` | [02_memory/02_eviction_priority](02_memory/02_eviction_priority/README.md) | 完整移动 |
| `03_memory/03_load_store_intrinsics/` | [02_memory/03_load_store_intrinsics](02_memory/03_load_store_intrinsics/README.md) | 完整移动 |
| `04_synchronization/01_memory_model/` | [04_sync/01_memory_model](04_sync/01_memory_model/README.md) | 完整移动 |
| `04_synchronization/02_async_barriers/` | [04_sync/02_async_barriers](04_sync/02_async_barriers/README.md) | 完整移动 |
| `04_synchronization/03_thread_scopes/` | [04_sync/03_thread_scopes](04_sync/03_thread_scopes/README.md) | 完整移动 |
| `04_synchronization/04_warp_functions/` | [04_sync/04_warp_functions](04_sync/04_warp_functions/README.md) | 完整移动；执行与基础原语章节交叉引用，不复制实现 |
| `05_streams_events/01_async_execution/` | [07_async/01_async_execution](07_async/01_async_execution/README.md) | 完整移动 |
| `06_advanced_features/` | [04_sync](04_sync/README.md)、[06_gemm](06_gemm/README.md)、[07_async](07_async/README.md) | 原为空占位；cluster、异步拷贝、Graph 与 Tensor Core 按机制归类；动态并行仍为待扩展主题 |
| `07_correctness_performance/` | [实验规范](EXPERIMENT_PROTOCOL.md)、[tools](tools/README.md)、[体系结构基础](ARCHITECTURE.md) | 原为空占位；正确性、误差、计时和工具验证贯穿所有主题 |
| `results/` | [各专题的配置日志](RECORD_LAYOUT.md) | 历史文本按授权归并到 `results/<variant>.log`；日期目录、旧源码/脚本快照与旧二进制已删除 |

原顶层与内存、同步、streams/events 的索引已整合进新导航；删除旧的重复索引和空占位目录。旧目录不保留兼容 symlink，外部书签或自定义命令需按此表更新。

## 当前构建与记录约定

此前曾使用单入口与配置级聚合日志；本轮按最新要求废止该规则，完整拆分原有 9 个入口为 51 个独立动作脚本。

- 当前入口为各专题的 `commands/<action>/run.sh`，每个动作直接列出实际命令，不转调万能分派器。旧入口已移除。
- 源码仍位于专题，产物共享 `build/<variant>/`，脚本和日志同处 `commands/<action>/`。日志为 `<variant>.log`；同动作同配置重跑覆盖，其他动作不受影响。
- `compile` 显式生成程序与必要 PTX/cubin。运行、静态检查、sanitizer、JIT 和调试只读取已有产物，缺失即提示先编译；源码、工具或编译选项改变后手动重编译。
- 不做哈希缓存、自动快照、时间戳归档或 history。正负编译契约探针保留，每次实际执行并检查诊断。公共层仅负责低层执行、日志、锁和产物检查。
- 本轮不改写或删除已有历史日志/报告，不运行 GPU 实验。旧 `HISTORICAL IMPORT` 仍只说明原记录；旧绝对路径不可直接重放，新日志不能证明旧数值。完整规则见 [RECORD_LAYOUT.md](RECORD_LAYOUT.md)，规划仍见 [学习路径](LEARNING_PATH.md)。

## 迁移后使用示例

从工作区根目录进入已有缓存实验：

```bash
cd gpu-architecture-lab/02_memory/01_cache_operators
CUDA_ARCH=sm_89 bash commands/compile/run.sh
CUDA_ARCH=sm_89 bash commands/static/run.sh
CUDA_ARCH=sm_89 bash commands/run/run.sh
```

这些命令仅展示入口，不表示本次已经执行 GPU 实验。运行前先读专题 README 并确认设备/工具链；`sm_89` 是示例而非默认保证，须替换为本机支持的目标。该配置产物在本专题 `build/sm_89-release/`，三个动作分别记录到 `commands/compile/sm_89-release.log`、`commands/static/sm_89-release.log` 和 `commands/run/sm_89-release.log`；重复动作仅覆盖自己的日志。warp 的 release 多动作复用与 gdb 独立 debug 构建示例见 [RECORD_LAYOUT.md](RECORD_LAYOUT.md)。
