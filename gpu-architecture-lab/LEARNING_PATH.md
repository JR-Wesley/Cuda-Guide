# 学习路径：从机器到微架构

本路径将 CUDA 编程语义与 GPU 微架构实验并行组织。先读 [体系结构基础](ARCHITECTURE.md)，再建立机器信息、编程模型与正确性基础。9 个迁入专题已有源码和历史记录；原 M0～M9 的系统性微基准与综合算法计划仍保留，不能因相关专题已有代码就把整个里程碑标成完成。此次整合没有新实测。

## 已有专题阅读线

| 顺序 | 阅读与动手入口 | 与微架构主线的连接 |
| --- | --- | --- |
| P0 | [机器与环境](00_machine/README.md)、[执行与编程模型](01_execution/README.md) | 查询真实能力，理解 launch、索引、边界与错误检查；不是已有统一环境探测程序 |
| P1 | [语言与设备函数](03_instruction/README.md)、[Load/Store Intrinsics](02_memory/03_load_store_intrinsics/README.md) | 类型、限定符、生命周期、编译资源与生成指令 |
| P2 | [Cache Operators](02_memory/01_cache_operators/README.md)、[Eviction Priority](02_memory/02_eviction_priority/README.md) | 接入 M1 的地址/缓存模型；不替代完整 stride、带宽和计数器扫描 |
| P3 | [Memory Model](04_sync/01_memory_model/README.md)、[Warp Functions](04_sync/04_warp_functions/README.md) | 接入 M2/M4/M5：参与集合、发布、复用与作用域 |
| P4 | [Async Barriers](04_sync/02_async_barriers/README.md)、[Thread Scopes](04_sync/03_thread_scopes/README.md) | 从正确协议进入 kernel 内 pipeline；SM90、cluster、system 路径单独判断支持 |
| P5 | [Async Execution](07_async/01_async_execution/README.md) | host 侧 stream/event、默认流与 Graph 可先于 GEMM 学习；接入 M8 的时间线证据 |
| P6 | [setmaxnreg](03_instruction/01_setmaxnreg/README.md) | M3 的高级寄存器案例；先区分基线、交叉编译、动态目标与未验证项 |

每项先读专题支持条件，再按本机能力选择动作；没有统一构建入口。专题功能 PASS、静态检查、历史性能和当前运行状态分别记录。

## 当前优先补充：五类指令分析线（不跑性能扫描）

[指令图谱](03_instruction/02_sass_atlas/README.md) 与下面的 M0～M9 性能主线并列，不要求先完成整套带宽或延迟扫描。它以源码 → PTX → SASS 的逐段解释为主，默认只编译、静态阅读和少量正确性核对；当前完成的是 [案例/文件架构规划](03_instruction/02_sass_atlas/PLAN.md)，不是已完成实测。

1. 单操作与索引复制：`LDG/STG`、线程坐标、地址计算、尾部谓词。
2. 整数/位/字节运算：`IMAD/LEA/IADD3/SHF/LOP3/PRMT` 的数据流。
3. 条件、循环与设备调用：`BRA/BRX/EXIT/CALL`、谓词与控制流；接受优化后没有目标 opcode。
4. shared、local、原子与 warp 交换：`LDS/STS/LDL/STL`、原子家族、`SHFL`；结合 `BAR/MEMBAR/WARPSYNC` 的正确性语义。
5. 异步搬运：`LDGSTS/LDGDEPBAR/DEPBAR` 候选、mbarrier、SM90 bulk 与待补 tensor-map TMA；编译/静态/运行按目标分别记录。

先读取每课必需的环境和能力，不沿用历史 `sm_89` 假设。复用现有专题独立动作入口，先核对依赖与目标配置再动手；本轮未执行这些脚本。分析卡将**功能语义、执行管线、延迟、吞吐**分开，后两项默认未测量，不为填写表格强行增加计时实验。

## 路线总览

| 顺序 | 里程碑 | 对应目录 | 关键问题 | 完成证据 |
| --- | --- | --- | --- | --- |
| M0 | Machine Explorer | `00_machine` | 这台机器的 CPU、GPU、驱动、Toolkit、SM/内存资源是什么？ | 原始设备/工具输出、版本与拓扑限制说明、机器逻辑图。 |
| M1 | Global Memory Explorer | `02_memory` | 连续、跨步、不同工作集的访存性能为何不同？ | 正确性、计时原始记录、固定条件的 CSV、可用时的 NCU 事务/缓存指标。 |
| M2 | Warp Explorer | `01_execution` | 分支布局和 warp 内通信怎样影响活跃 lane 与指令执行？ | divergence 对照、shuffle/ballot 结果、PTX/SASS、分析器记录。 |
| M3 | Latency Hiding Explorer | `01_execution` + `03_instruction` | 依赖链、独立指令、驻留 warp 和寄存器资源如何共同影响吞吐？ | 固定工作量的参数扫描，静态资源与 eligible/active warp 等指标对照。 |
| M4 | Shared Memory Lab | `02_memory` + `04_sync` | bank 映射、padding 和 barrier 如何影响正确性及吞吐？ | `[32][32]` 与 `[32][33]` 对照、SASS、可用时的 bank conflict 指标。 |
| M5 | Atomic & Sync Lab | `04_sync` | 冲突、同步和内存顺序的成本及正确性边界是什么？ | 正确性检查、不同冲突度的对照、sanitizer 原始报告。 |
| M6 | Reduction | `05_primitives` | 原子、共享内存、warp shuffle 怎样逐步组合成高效归约？ | 每一版都保留，使用相同输入、误差条件和测量方法对比。 |
| M7 | GEMM | `06_gemm` | blocking 与数据复用为何有效，何时受访存或计算限制？ | 从朴素版逐步演化的实现及正确性、SASS、性能证据。 |
| M8 | Async Pipeline | `07_async` | 传输与计算能否重叠，重叠程度受什么限制？ | 同工作量的顺序/流水线对照与时间线。 |
| M9 | Tensor Core | `06_gemm` | 专用计算单元改变了哪些数据类型、布局与吞吐条件？ | 与前一 GEMM 版本的同精度定义对照。 |
| 全程 | PTX/SASS/Profiler | `tools` + 各实验 | 源码如何映射到指令，指令与实际性能是否一致？ | 每个实验引用编译命令、静态报告和运行记录。 |

微基准主线先通过 M0、M1，再扩展 M2～M5；M6 的 reduction 是第一个综合算法，GEMM 与 Tensor Core 放在基础机制之后。已有 warp/CUB 示例是 M2/M6 的素材，不等于完成整套 reduction/scan 优化；已有 Graph capture 属于 M8，cluster 在执行与同步主题中学习，Tensor Core 保留在 M9，不另建空的“高级特性”目录。`occupancy`、静态指令数或事件计时任何一项都不足以独自证明瓶颈。[Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)解释了硬件指标采集与占用率的解释边界。

## 首批待实施的微基准计划：M0 + M1

### M0：认识本机

计划采集 `lscpu`、`lspci -vv`、`nvidia-smi -q`、`nvidia-smi topo -m` 和 CUDA Runtime 设备属性；记录 GPU 名称、compute capability、SM 数、warp 大小、线程与寄存器限制、共享内存、L2、显存与异步引擎等。将每个值标注来源：**API 返回、系统工具输出、官方规格或推断**。若通过 WSL 使用 GPU，其 PCIe/拓扑视图可能与物理连接不同，不能仅凭一条虚拟化输出下结论；本机当前环境需重新查询。

完成条件：画出 CPU → 互连 → GPU → SM/L1/shared → L2 → DRAM 的**逻辑图**，并说明哪些连接或容量只是模型、哪些由本机命令直接观测。可按 [SDK 当前环境检查说明](../cuda-sdk-lab/experiments/environment/README.md)采集工具与设备版本。原链接的环境记录已不可用，不能继续用于版本对照或实测证据；当前 README 是操作说明，不是环境测量结果，也不能代替完整的机器清单。

### M1a：连续访问的基线

从最小复制/读取写入实验建立正确性与时间基线；扫描工作集大小，分别讨论“应用处理的有效字节数”和硬件实际搬运的字节数。输出时间、有效带宽、输入大小与重复次数；缓存命中和 DRAM 流量必须由可用的硬件指标支持，不能把 `有效字节 / 时间` 直接称为 DRAM 带宽。基线测量应固定目标架构、编译选项、时钟环境、预热与计时范围。

### M1b：stride/coalescing

在相同有效工作量下扫描 stride `1, 2, 4, 8, 16, 32, 64, 128`；先画一个 warp 的地址表，预测访问跨度和可能的事务变化，再记录时间及指标。设计输入时保证每次访问都在范围内、结果可校验，并注明改变 stride 是否同时改变工作集或缓存复用。将“访问模式变化”与“数据量变化”区分开，才能解释曲线。[CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)讨论了 global memory coalescing。

### M1c：读出报告再决定下一步

优先核对 kernel 正确性和计时稳定性；Nsight Systems 用于检查启动、拷贝与 kernel 的时间线；Nsight Compute 可用时再看访存事务、缓存和吞吐。采集方法见 [SDK 当前 Nsight Compute 动作说明](../cuda-sdk-lab/experiments/ncu/README.md)。原链接的采集记录已不可用，不再据此引用 `SKIPPED` 或硬件指标获取状态；当前 README 不是新的采集证据，工具可用性和采集结果须重新验证。完整 M1 扫描的硬件指标仍需独立采集。可以先写预测与收集命令，但**不能把 M1 标为已完成实测**。

## 每个里程碑的通过标准

1. 至少有一个单变量对照，源码和输入能复现，输出经过正确性校验。
2. 把预测写在看 profiler 结果之前；报告保留与预测相符和不符的证据。
3. 原始命令、stdout/stderr、退出码、设备/工具版本与明确的编译配置可核对；日志位于各动作脚本同目录。
4. 同一结论明确区分源码/地址推断、PTX/SASS 静态证据、Profiler 动态证据。
5. 报告未采集的指标、测量噪声和受限环境；不能用空白或 `0` 表示“工具不可用”。

具体字段和写法见 [实验记录规范](EXPERIMENT_PROTOCOL.md)。旧的环境、编程模型、语言特性、高级特性和正确性学习目标分别由 00/01/03、04/06/07 与全程规范承接；目录迁移及历史证据路径见 [MIGRATION.md](MIGRATION.md)。
