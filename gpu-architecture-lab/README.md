# CUDA 编程与 GPU 体系结构实验室

本目录合并 CUDA 编程特性与 GPU 微架构学习：从 **编程语义 → 硬件映射 → 正确性 → 性能模型 → 实验证据** 理解执行、存储、指令、同步和异步流水线。不把 CUDA API 名称直接等同于某个硬件单元，也不把一次运行结果当作跨架构保证。

**当前状态：9 个已有可执行专题与微架构规划并存。** 下列专题从 `cuda-programming-lab` 迁入，源码、文档、独立动作脚本、共享构建产物与日志均位于各专题内；其他表中标为规划的项目并未因此完成。此次文档整合没有重新进行 GPU 实测。专题 `README.md` 说明实现与命令，`RESULTS.md` / `DEBUGGING.md` 说明各次验证范围与限制；“有代码”不等于所有路径都在当前机器上运行过。

## 从哪里开始

1. 读 [体系结构基础](ARCHITECTURE.md)：建立执行/存储层次、roofline、延迟与吞吐模型，理解正确性与异步的不同层次。
2. 读 [学习路径](LEARNING_PATH.md)，先完成 [机器与环境](00_machine/README.md) 和 [编程模型](01_execution/README.md)，再选择已有专题或新的单变量实验。
3. 每次运行遵循 [实验记录规范](EXPERIMENT_PROTOCOL.md)，按 [工具说明](tools/README.md) 区分功能、静态检查、sanitizer 与 profiler。
4. **当前新增：先规划的 [五类 SASS 指令图谱](03_instruction/02_sass_atlas/README.md)**。从小复制、地址/位运算、分支、shared/原子/同步到异步搬运/TMA，复用已有特性源码；[目录与案例规划](03_instruction/02_sass_atlas/PLAN.md) 明确缺口和分阶段交付。本轮仅文档与源码盘点，不新增已验证专题，不开展大规模性能测试；已有动作脚本复用，其存在不代表本轮执行通过。

## 主题地图

| 目录 | 核心问题与承接内容 | 当前状态 |
| --- | --- | --- |
| [00_machine](00_machine/README.md) | 系统拓扑、设备能力、驱动/Toolkit、构建目标、错误检查 | 教学与机器清单规划 |
| [01_execution](01_execution/README.md) | host/device、kernel 启动、grid/block/thread、warp、occupancy、cluster 执行模型 | 教学与微基准规划；引用已有 warp/cluster 专题 |
| [02_memory](02_memory/README.md) | 地址空间、生命周期、缓存策略、带宽、coalescing、shared bank | 3 个已有专题 + 微基准规划 |
| [03_instruction](03_instruction/README.md) | 语言限定符、设备函数、编译链接、寄存器；五类 SASS 案例图谱；依赖链、ILP | 1 个已有专题 + 五类指令分析规划；性能微基准另行保留 |
| [04_sync](04_sync/README.md) | 原子性、order/scope、warp 通信、barrier、缓冲区所有权 | 4 个已有专题 + 同步成本规划 |
| [05_primitives](05_primitives/README.md) | vector add、transpose、reduction、scan 的组合与校验 | 综合算法规划；引用已有 warp/CUB 示例 |
| [06_gemm](06_gemm/README.md) | blocking、复用、roofline、异步 tile、Tensor Core | GEMM 与 Tensor Core 规划 |
| [07_async](07_async/README.md) | stream/event、host/device 重叠、Graph、kernel 内异步搬运 | 1 个已有专题；关联 barrier/pipeline，保留性能扩展规划 |
| [tools](tools/README.md) | 底层执行、日志、产物检查与测量职责 | 9 个专题、51 个独立动作脚本；无万能分派入口 |
| [专题记录布局](RECORD_LAYOUT.md) | `build/<variant>/` 与 `commands/<action>/<variant>.log` | 源码/构建共享，动作脚本与日志同目录；历史证据不覆盖 |

## 9 个已有可执行专题

| 入口 | 学习内容 | 需要特别区分的边界 |
| --- | --- | --- |
| [setmaxnreg](03_instruction/01_setmaxnreg/README.md) | warpgroup 寄存器配额、spill、静态与动态验证 | 历史 `sm_89` 只运行基线；动态路径要求匹配的受支持目标，不能将 `sm_90` 等同于 `sm_90a` |
| [Cache Operators](02_memory/01_cache_operators/README.md) | PTX 缓存提示、依赖地址环、streaming、mapped memory | 提示不建立同步；逻辑带宽不是 DRAM 流量 |
| [Eviction Priority](02_memory/02_eviction_priority/README.md) | L1 热点保护与干扰访问 | 仅本例 L1 提示；周期数不是命中率 |
| [Load/Store Intrinsics](02_memory/03_load_store_intrinsics/README.md) | C++ 重载、只读生命周期、类型/对齐与指令映射 | 类型和功能验证不是性能排名 |
| [Memory Model](04_sync/01_memory_model/README.md) | release/acquire、fence、scope、消息发布 | system GPU-only 不等于 CPU/GPU 并发测试 |
| [Async Barriers](04_sync/02_async_barriers/README.md) | arrive/wait、phase、drop、completion、双缓冲 | 普通路径与 SM90 transaction 路径分开；工具报告见专题记录 |
| [Thread Scopes](04_sync/03_thread_scopes/README.md) | scoped atomics、pipeline、carveout、system/cluster 条件路径 | scope 不等于缓存层级；cluster/system 各自能力门控 |
| [Warp Functions](04_sync/04_warp_functions/README.md) | mask、shuffle/vote/match/reduce、CUB、共享内存交接 | 寄存器 collective 不自动发布内存；可选 API 与 JIT 单独记录 |
| [Async Execution](07_async/01_async_execution/README.md) | 默认流、events、回调、拷贝流水线、Graph、错误定位 | API 返回、操作完成、真实重叠分别验证 |

## 专题阅读

- [深度解读：Using CUDA Warp-Level Primitives](04_sync/04_warp_functions/USING_CUDA_WARP_LEVEL_PRIMITIVES.md)：结合 NVIDIA 博客与 CUDA 指南，分析 warp 参与集合、寄存器交换、内存同步和动态分组；包含来源说明与后续实验建议。本次为文档分析，不是新增 GPU 实测。

## 构建与迁移约定

**主入口是具体专题的 README，动作入口统一为 `commands/<action>/run.sh`。** 每个动作是独立可读脚本，不转调万能入口。进入专题后先核对参数、默认架构与工具能力，不假设所有专题都有同样动作。编译成功、静态检查通过、运行通过和工具跳过分别记录；历史 `sm_89` 不是全实验室的默认支持承诺。

```bash
# 在所选专题目录，先编译，再选择一个需要的验证动作
CUDA_ARCH=sm_89 bash commands/compile/run.sh
CUDA_ARCH=sm_89 bash commands/run/run.sh
```

`compile` 显式生成共享 `build/<variant>/` 中的程序和必要 PTX/cubin；其余执行/检查动作只复用已有产物，缺失时提示先编译，不隐式构建。更改源码、头文件、编译选项或工具链后需手动重新编译。无哈希缓存、自动快照、时间戳归档或 history。日志写在脚本同目录 `commands/<action>/<variant>.log`，同动作同配置重跑覆盖，不再聚合到 results。特殊编译配置和正负编译探针见 [RECORD_LAYOUT.md](RECORD_LAYOUT.md)。

既有 `results/` 日志及历史报告保持原样；其中 `HISTORICAL IMPORT` 不是新入口的实测。本轮迁移没有运行 GPU。旧数值继续指向原证据，不换链到新动作日志。迁移路径见 [MIGRATION.md](MIGRATION.md)；原空框架由现有主题与实验规范承接，不为目录齐全新增空主题。

## 与 `cuda-sdk-lab` 的分工

| 已有内容 | 在这里的用途 |
| --- | --- |
| [编译追踪](../cuda-sdk-lab/experiments/compile/README.md)与 [`cuobjdump` 教程](../cuda-sdk-lab/experiments/binary/README.md) | 复用 `.cu → PTX → cubin → SASS` 的检查方法 |
| [转置源码](../cuda-sdk-lab/examples/vector_transpose.cu)与 [当前运行动作说明](../cuda-sdk-lab/experiments/execution/README.md) | 仅作为实现与操作方法参考；原小规模运行记录已不可用，不能继续作为实测证据，当前 README 也不是替代证据 |
| [SDK 学习总图](../cuda-sdk-lab/LEARNING_FRAMEWORK.md) | Toolkit、Runtime/Driver、编译与工具背景 |

每个实验回答：**规范保证什么？代码与输入是什么？PTX/SASS 生成什么？动态工具测到什么？解释还有哪些边界？** 工具不可用时写“未测量”并保留原因，不能用理论值或旧记录补写本次指标。
