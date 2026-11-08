# NCU 学习材料与本地编译入口

这里既有 Nsight Compute（NCU，GPU 性能分析工具）的课程笔记，也有 CUTE、CUTLASS 和矩阵乘法练习。教程面向会使用函数和数组的大一学生，不要求事先掌握 GPU 调度或性能分析。**Kernel（内核）**是交给 GPU 上许多线程执行的函数：例如两个数组逐项相乘，让不同线程分别计算不同下标的结果。先区分三件事：**编译**生成程序，**运行**观察程序结果，**采集**让分析器收集指标；编译成功不能替代正确性或性能验证。

## 怎样使用 lec 和 lab

`lecNN.md` 负责解释概念与原理；`labNN.md` 负责带你用具体代码检验这些原理。第一次学习按课号推进，不需要先读完所有讲义，也不要一开始运行完整采集矩阵。

每课 lab 按下面的学习过程组织：

1. **先知道为什么做。** 阅读开头的问题和学习目标，明确本课要解释哪种现象，补齐必要术语。
2. **先用小输入预测。** 对照入口与共享 kernel，手算输出、线程地址或依赖关系。先弄清程序算什么，不急着测性能。
3. **编译和正确性检查分开做。** 按文档给出的目录执行底层命令，先确认结果正确，再扩大输入；修改常量后显式重新编译。
4. **一次只改变一个主要条件。** 每个步骤都说明目的、要操作的代码或命令、观察点，以及结果不符合预期时先查什么。
5. **把观察接回原理。** 不只抄指标数字：用地址、资源预算、依赖链或数据流解释变化，同时说明哪些结论还不能得出。
6. **完成理解检查。** 先预测一个小修改会造成什么变化，再核对解释。能运行命令不等于已经理解实验。

文档中的“预期”是待检验的推导，不是实测结论。各课末尾集中说明已有编译记录、未验证范围及旧证据边界；阅读步骤时不必反复跳进历史日志。底层命令帮助理解动作，已有独立脚本只是复现入口，不替代实验解释。

## 12 课课程目录

初次学习按 `lec01 → lab01 → lec02 → lab02 → …` 阅读；讲义中的配套专题在相应位置再展开，不需要先读完所有总指南。第三课介绍硬件分层与静态容量，第四课才研究实际驻留和指令调度。课程顺序与实验主题以 [配套实验总纲](README_experiment_curriculum.md) 为准；[系统学习指南](ncu_systematic_learning_guide.md)提供知识地图，[硬件参考](ncu_guide.md)用于按需回查。理论讲义统一命名 `lecNN.md`，配套实验统一命名 `labNN.md`（两位课号）。

| 课程目录 | 理论讲义 | 实验文档 | 本课入口源码 | 命令与日志路径 |
| --- | --- | --- | --- | --- |
| [01_ncu_basics](01_ncu_basics/) | [lec01.md](01_ncu_basics/lec01.md) | [lab01.md](01_ncu_basics/lab01.md) | [main.cu](01_ncu_basics/kernels/main.cu) | [commands/](01_ncu_basics/commands/) · [compile.log](01_ncu_basics/log/compile.log) · [run.log](01_ncu_basics/log/run.log) |
| [02_replay_and_measurement](02_replay_and_measurement/) | [lec02.md](02_replay_and_measurement/lec02.md) | [lab02.md](02_replay_and_measurement/lab02.md) | [main.cu](02_replay_and_measurement/kernels/main.cu) | [commands/](02_replay_and_measurement/commands/) · [compile.log](02_replay_and_measurement/log/compile.log) · [run.log](02_replay_and_measurement/log/run.log) |
| [03_occupancy](03_occupancy/) | [lec03.md](03_occupancy/lec03.md) | [lab03.md](03_occupancy/lab03.md) | [main.cu](03_occupancy/kernels/main.cu) | [commands/](03_occupancy/commands/) · [compile.log](03_occupancy/log/compile.log) · [run.log](03_occupancy/log/run.log) |
| [04_warp_scheduling](04_warp_scheduling/) | [lec04.md](04_warp_scheduling/lec04.md) | [lab04.md](04_warp_scheduling/lab04.md) | [main.cu](04_warp_scheduling/kernels/main.cu) | [commands/](04_warp_scheduling/commands/) · [compile.log](04_warp_scheduling/log/compile.log) · [run.log](04_warp_scheduling/log/run.log) |
| [05_global_memory](05_global_memory/) | [lec05.md](05_global_memory/lec05.md) | [lab05.md](05_global_memory/lab05.md) | [main.cu](05_global_memory/kernels/main.cu) · [vector_access.cuh](05_global_memory/kernels/vector_access.cuh) | [commands/](05_global_memory/commands/) · [compile.log](05_global_memory/log/compile.log) · [run.log](05_global_memory/log/run.log) |
| [06_shared_memory_and_transpose](06_shared_memory_and_transpose/) | [lec06.md](06_shared_memory_and_transpose/lec06.md) | [lab06.md](06_shared_memory_and_transpose/lab06.md) | [main.cu](06_shared_memory_and_transpose/kernels/main.cu) | [commands/](06_shared_memory_and_transpose/commands/) · [compile.log](06_shared_memory_and_transpose/log/compile.log) · [run.log](06_shared_memory_and_transpose/log/run.log) |
| [07_compute_pipeline](07_compute_pipeline/) | [lec07.md](07_compute_pipeline/lec07.md) | [lab07.md](07_compute_pipeline/lab07.md) | [main.cu](07_compute_pipeline/kernels/main.cu) | [commands/](07_compute_pipeline/commands/) · [compile.log](07_compute_pipeline/log/compile.log) · [run.log](07_compute_pipeline/log/run.log) |
| [08_roofline_and_performance_models](08_roofline_and_performance_models/) | [lec08.md](08_roofline_and_performance_models/lec08.md) | [lab08.md](08_roofline_and_performance_models/lab08.md) | [main.cu](08_roofline_and_performance_models/kernels/main.cu) | [commands/](08_roofline_and_performance_models/commands/) · [compile.log](08_roofline_and_performance_models/log/compile.log) · [run.log](08_roofline_and_performance_models/log/run.log) |
| [09_ptx_sass_and_source](09_ptx_sass_and_source/) | [lec09.md](09_ptx_sass_and_source/lec09.md) | [lab09.md](09_ptx_sass_and_source/lab09.md) | [main.cu](09_ptx_sass_and_source/kernels/main.cu) | [commands/](09_ptx_sass_and_source/commands/) · [compile.log](09_ptx_sass_and_source/log/compile.log) · [run.log](09_ptx_sass_and_source/log/run.log) |
| [10_advanced_profiling](10_advanced_profiling/) | [lec10.md](10_advanced_profiling/lec10.md) | [lab10.md](10_advanced_profiling/lab10.md) | [main.cu](10_advanced_profiling/kernels/main.cu) | [commands/](10_advanced_profiling/commands/) · [compile.log](10_advanced_profiling/log/compile.log) · [run.log](10_advanced_profiling/log/run.log) |
| [11_multi_gpu_communication_async_engines](11_multi_gpu_communication_async_engines/) | [lec11.md](11_multi_gpu_communication_async_engines/lec11.md) | [lab11.md](11_multi_gpu_communication_async_engines/lab11.md) | [main.cu](11_multi_gpu_communication_async_engines/kernels/main.cu) | [commands/](11_multi_gpu_communication_async_engines/commands/) · [compile.log](11_multi_gpu_communication_async_engines/log/compile.log) · [run.log](11_multi_gpu_communication_async_engines/log/run.log) |
| [12_end_to_end_performance_attribution](12_end_to_end_performance_attribution/) | [lec12.md](12_end_to_end_performance_attribution/lec12.md) | [lab12.md](12_end_to_end_performance_attribution/lab12.md) | [main.cu](12_end_to_end_performance_attribution/kernels/main.cu) | [commands/](12_end_to_end_performance_attribution/commands/) · [compile.log](12_end_to_end_performance_attribution/log/compile.log) · [run.log](12_end_to_end_performance_attribution/log/run.log) |

### 每课统一骨架

```text
NN_<课程主题>/
├── lecNN.md              # 已有理论讲义，保留正文
├── labNN.md              # 唯一配套实验文档
├── kernels/
│   └── main.cu           # 本课的实验入口
├── commands/
│   ├── compile.sh        # 显式编译，CUDA_ARCH 必填
│   ├── run.sh            # 普通运行，不隐式编译
│   └── profile.sh        # 本课 NCU 采集，不隐式编译
├── log/
│   ├── compile.log       # 新入口执行后才生成，各动作日志含命令和退出码
│   ├── run.log
│   ├── profile.log
│   ├── profile.ncu-rep   # 成功采集后生成；拒绝覆盖已有报告
│   └── legacy_commands/ # 原 commands 非脚本证据，保留原相对动作路径
└── build/                # 各动作共用 build/lessonNN
```

12 课均已提供 `labNN.md` 实验设计和 `kernels/main.cu` 入口。先读各课顶部的少量常量，再读所选共享头中的 kernel 与 `run_*` 主机函数；每次只选择一种算法和变体，不自动扫描参数或跑完整实验矩阵。除第一课兼容可选 N 参数外，通过修改明确的常量进行对照，不引入通用参数系统。

共享向量接口已改为 `template<int Stride>`；Lab 5 另有本课五版本入口。当前验证范围与历史证据边界统一见[页末说明](#本次验证范围)，不能用旧日志验证改后的源码。

**改前版本的历史验证：** 当时 12 课均已执行 `CUDA_ARCH=sm_89 bash commands/compile.sh`，nvcc 13.4、C++17、`-O3 -lineinfo` 编译成功（第 10 课另加 `-ldl`）；随后执行 `bash commands/run.sh`，第 01–10、12 课 PASS/exit 0，第 11 课因单卡返回 `SKIPPED: need GPU0 and GPU1; found 1 device(s)`/exit 77，不是 PASS。证据为各课 `log/compile.log`、`log/run.log`，入口见上表。环境记录为 RTX 4060 Laptop、cc 8.9；`nvidia-smi` 驱动查询为 596.21，SMI 页面用户态版本为 615.78.08，分开记录，不混称同一版本。

上述历史验证只覆盖当时的入口配置，未测试全部参数；当时 `profile.sh` 仅通过 `bash -n`，未执行 NCU、NSYS、Python workers 或 SASS 检查，无性能结论。各课 `log/compile.log`、`log/run.log` 原样保留，仅属改前记录；`log/legacy_commands/compile/source_check.log` 也只代表历史 compile-only。修改源码、共享头或选项后须显式重编，不能用这些旧日志验证本次模板接口。

每课 `commands/` 现在只保留三个独立脚本，没有动作子目录或万能分派器。第一课原 env/basic/launch/memory/scheduler/metrics 等入口已移除；环境查询仍可按 lab 手工执行，其他专题按对应课程学习。旧 `commands` 中全部非脚本证据已移动到同课 `log/legacy_commands/<原相对路径>`，内容与原始命令不改；`build/` 和 `runs/` 中历史产物保持原地。第一课旧 Vector Add 结果**不对应当前 Multiply**，旧证据链接见 [RESULTS.md](01_ncu_basics/RESULTS.md)。

### 三个独立脚本怎样使用

先读本课 lab 的底层命令，再在对应课程目录执行（也可从任意目录传入脚本的实际路径）：

```bash
# 以第 01 课为例；sm_89 仅为架构示例，必须显式指定实际支持的目标
CUDA_ARCH=sm_89 bash commands/compile.sh
bash commands/run.sh 16
bash commands/profile.sh 1048576
ncu --import log/profile.ncu-rep --page details
```

第 02–12 课的三个脚本均无位置参数；修改入口常量后重新 compile，再 run、profile。第 01 课 run 的可选 N 默认 16，profile 默认 1048576，范围由程序检查。只有 compile 要求 `CUDA_ARCH`，固定 C++17、`-O3 -lineinfo`，第 10 课额外链接 `-ldl`；不猜架构、不自动安装依赖。compile 编译前移除旧 `build/lessonNN`，然后直接调用 nvcc，失败由 `set -e` 停止；run/profile 缺可执行产物会提示先编译，不自动构建，也不判断已有二进制是否对应新源码。脚本只展示实验动作，不校验参数数量、架构格式、编译后产物或报告内容；架构支持、计算结果和采集完整性按工具输出及实验步骤检查。

每个动作把展开的命令、stdout/stderr 与最终 `exit_code` 写到本课 `log/compile.log`、`log/run.log`、`log/profile.log` 并显示在终端；未执行不生成新日志。同动作重跑覆盖自己的新日志，不触碰 `legacy_commands/`，不要同课并发执行。要保留独立实验，重跑前把日志和报告一起改成有意义的配置/运行名；profile 拒绝覆盖已有 `log/profile.ncu-rep`，NCU 失败即停止；是否选中目标及报告内容由实验者检查。直接执行教学命令时也须记录命令、配置、输出与退出码，配置命名报告存于本课 `log/`，不能把脚本固定日志当成全部对照历史。

profile 使用本课已有 kernel 的函数基名过滤器；有多个算法的课程列出本课候选名称，但只运行源码所选分支，不扫描实验矩阵。除第 10 课外都跳过 10 次预热、采 1 次，显式固定 kernel replay、cache all、clock base，支持情况以实际工具输出为准。修改 warmup 后也要核对 skip。第 10 课同时使用 Profiler API 和 NVTX `measure/` 过滤，不再 `-s 10`；API/NVTX 分开对照及 Python worker 仍按 lab 的明确命令执行。第 11 课 profile 先普通执行一次能力/正确性预检，原样保留 `SKIPPED/77`，只有退出 0 才启动 NCU；预检不替代单独的普通基线。

| 课 | profile 默认采集内容 |
| --- | --- |
| 01、02 | basic；02 的 cache/clock/replay 单变量对照仍见 lab02 |
| 03 | LaunchStats、SpeedOfLight、Occupancy |
| 04 | 上述三组及 SchedulerStats、WarpStateStats |
| 05 | LaunchStats、SpeedOfLight、MemoryWorkloadAnalysis；过滤五版候选函数基名，只采源码所选版本，完整命令见 [lab05](05_global_memory/lab05.md) |
| 06、12 | 上述三组及 Occupancy；12 的 C/D 专项采集仍见 lab12 |
| 07 | LaunchStats、SpeedOfLight、ComputeWorkloadAnalysis、InstructionStats、SchedulerStats |
| 08 | LaunchStats、SpeedOfLight、ComputeWorkloadAnalysis、MemoryWorkloadAnalysis |
| 09 | LaunchStats、SpeedOfLight、SourceCounters、InstructionStats、MemoryWorkloadAnalysis |
| 10 | basic，限定 Profiler API 与 NVTX measure 范围 |
| 11 | basic，目标为 GPU1 的 copy_kernel；不支持时跳过 |

### 四个共享基础 Kernel

| ID | 唯一算法实现 | 最小对照 |
| --- | --- | --- |
| A | [kernels/vector_multiply.cuh](kernels/vector_multiply.cuh)；Lab 5 扩展见 [vector_access.cuh](05_global_memory/kernels/vector_access.cuh) | 编译期 stride；Lab 5 五种地址模式见下表，每次只选一个并重编 |
| B | [kernels/matrix_transpose.cuh](kernels/matrix_transpose.cuh) | naive / tiled / padded；后两者只改 shared 行宽 |
| C | [kernels/reduction.cuh](kernels/reduction.cuh) | shared tree / warp shuffle；同样的 GPU partial sums |
| D | [kernels/fma.cuh](kernels/fma.cuh) | 1/4/8 条独立链；固定每线程总 FMA；增加链数观察寄存器压力 |

向量乘法算法保持两读一写。当前共享接口为 `template<int Stride>`：入口用 `constexpr int stride=1` 调用 `run_vector<stride>(n, block)`，runner 启动 `vector_mul<Stride><<<...>>>(..., n)`，kernel 无运行时 stride 参数。host 保留原初始化与全量检查，不增加通用框架。编译期常量使地址计算可被常量化，但未经编译不保证 SASS 形式；改变 stride 也会影响输入足迹、缓存关系和可能的指令生成，不能称为“只改变内存行为”。

算法与 host 流程通常共享实现，不复制到每课。Lab 5 已增加本课轻量 host 和地址模式，复用共享 kernel、计时及 RAII，接口与验证状态见 [lab05](05_global_memory/lab05.md)。[common.cuh](kernels/common.cuh) 已提供 CUDA 错误检查、`DeviceBuffer<T>` / `CudaEvent` RAII 和 Event 批量计时；四个共享 runner 及第 10 课已采用设备缓冲区 RAII，`time_kernel` 使用事件 RAII。RAII 只管理资源生命周期，数据拷贝、kernel launch 与 CPU 全量校验仍显式书写。共享计时保持 10 次预热、同步、20 次计时启动并返回每次平均毫秒，分配、拷贝与校验在计时外，批量区间仍可能含提交间隙。传入的 `[&] { kernel<<<...>>>(...); }` 只是启动 kernel 的 lambda，不是实验框架；第 10 课仍无独立 Event 计时，第 11 课多 GPU 保留显式分配、设备切换与释放，不引入设备所有权框架。

### 每课只改什么、看什么

下表均为**待验证实验设计，不是实测结论**。完整固定条件、反例与命令见各课 `labNN.md`。

| 课 | 源码选择/微调 | 后续观察重点 |
| --- | --- | --- |
| 01 | A，小 N 核对；程序固定，换报告层级 | LaunchStats、SpeedOfLight、Raw 名称/单位 |
| 02 | A 不变；分开改变 Cache/Clock/Replay 采集配置 | 普通 Event 基线、Duration、pass 数与缓存流量 |
| 03 | A 的 block=128/256/512；D 作为寄存器压力补例 | 每线程寄存器数、各项驻留资源限制与理论 Occupancy；动态调度留到第 04 课 |
| 04 | D 的 chains=1/4/8；C 的 shuffle 开关 | SchedulerStats、WarpStateStats；依赖与同步 |
| 05 | A 的本课专用 host：access 选择连续/偏移/跨步读/跨步写/仅 b 广播；stride/offset 编译期选择，每次单独重编 | a/b/c sector 预测与待测表；分开 load/store、逻辑量/唯一输入/足迹，不以性能优化为主 |
| 06 | B 的 variant=0/1/2，固定 N 与 block | Global sectors、Shared wavefront/bank conflict |
| 07 | D 固定总 FMA，仅改 chains | Issue Activity、FP32 pipeline、指令数 |
| 08 | 分别读 A/D/B，不把不同算法耗时直接比加速 | FLOPs/Bytes 下界、分层流量和 Roofline |
| 09 | D 增加链数；B/C 补充指令映射 | Source/SASS、寄存器/local memory；不预设 Spill |
| 10 | A 不变；Profiler API / NVTX / worker 数 | 是否选中目标进程与范围；本课不再 `-s 10` |
| 11 | 同一个 copy kernel：GPU1 读本地或 GPU0 | P2P 方向、GPU1 访存及系统观测边界 |
| 12 | B 三版本完整归因，再用 C 或 D 检验方法 | 预测—正确性—独立计时—指标—反例 |

第 10 课有可选标准库 Python 启动层；CUDA Graph 暂不实现。第 11 课是独立选修，代码先检查两张 GPU 及 GPU1→GPU0 访问能力，不支持时返回 `SKIPPED/77`。它是 SM 执行的 copy kernel，不是 Copy Engine/TMA 示例，也不保证走 NVLink。

## 其他示例与维护边界

`basics/`、`CUTE/`、`cutlass/`、`mma/` 是既有独立示例，不并入本次 12 课骨架，保持其原有源码和独立 compile 入口。理论讲义中的示例命令或“独立下载文件”文字不代表对应程序已经落地；以实际源码、脚本和本页状态说明为准。

本次检查的本地示例没有嵌套 `.git`，`ncu/` 也不属于一个外层 Git 仓库。CMake 引用的 `/home/mikasa/hpc/cutlass` 则是独立 NVIDIA/CUTLASS 上游仓库，已有本地修改与构建目录，本次仅确认边界，未改写或构建它。

## 七个编译入口

原七个 `build.sh` **都只执行 CMake 配置与编译**，没有运行、NCU 采集或依赖安装动作。现在迁移为下列单动作入口；旧脚本已移除，不保留转调包装器，也不凭空增加 run/profiling 框架。

| 本地专题 | 新入口 | CMake 声明的目标名 | 当前架构限制 |
| --- | --- | --- | --- |
| CUTE layout | [compile](CUTE/layout/commands/compile/run.sh) | `layout` | 显式传入 `CUDA_ARCH`，须满足本机工具链与源码要求 |
| CUTE SGEMM | [compile](CUTE/sgemm/commands/compile/run.sh) | `sgemm` | 同上 |
| CUTLASS GEMM | [compile](cutlass/gemm/commands/compile/run.sh) | `turing_tensorop_gemm` | 当前 CMake 写死 `sm_89`，入口仅接受显式 `CUDA_ARCH=sm_89` |
| CUTLASS hello | [compile](cutlass/hello/commands/compile/run.sh) | `hello-cutlass` | 同上；虽只有 C++ 输入，现有 CMake 仍启用 CUDA 语言 |
| CUTLASS minigemm | [compile](cutlass/minigemm/commands/compile/run.sh) | `minigemm` | 同上；当前源码为空，不应期待成功链接 |
| CUTLASS ref_gemm | [compile](cutlass/ref_gemm/commands/compile/run.sh) | `ref_gemm` | 显式传入 `CUDA_ARCH`，须满足本机工具链与源码要求 |
| MMA | [compile](mma/commands/compile/run.sh) | `mma_gemm` | 当前 CMake 写死 `sm_89`，入口仅接受显式 `CUDA_ARCH=sm_89` |

这些是配置文件声明的目标名，**不是本次已生成或验证通过的程序**。

### 1. 先检查前提与已有约束

需要 Bash、CMake、Make、CUDA Toolkit，以及默认使用的 `clang++`（可用 `CXX` 显式指定其他已有编译器）。六个 CUTE/CUTLASS 练习仍引用 CMake 中固定的 CUTLASS include 路径；MMA 仍依赖现有 LibTorch、CUDA 和 lld 配置。入口不下载、不安装、不自动修复依赖，也不编译第三方工程。

上述七个专题的 `CMakeLists.txt` 与 CUDA/C++/Python 计算源码保持不变，以下既有问题仍需另行处理：

- `cutlass/gemm`、`hello`、`minigemm` 的 CMake 含形如 `&><...` 的异常 Debug 表达式，CUDA Release 参数中还有 `-Iineinfo`，不能把它当作已正确启用 `-lineinfo`。本次没有运行 CMake 来验证具体报错。
- 上述三个专题和 `mma` 仍写死 `-arch=sm_89`，因此新入口拒绝其他架构，避免日志中的请求架构与实际参数冲突。这不是整个工作区的默认架构，也不意味着设备或 Toolkit 一定支持它。
- `cutlass/minigemm/minigemm.cpp` 为空，没有 `main`；入口迁移不会补出计算实现。
- `mma` 原 Torch 前缀为 `/home/mikasa/.venvs/cuda_ops/lib/python3.13/site-packages/torch/share/cmake`。现在可通过 `TORCH_PATH` 覆盖为本机已有安装的 CMake 前缀；没有安装依赖的副作用。

### 2. 理解编译命令，再执行独立入口

以 `CUTE/layout` 为例，从 `ncu/CUTE/layout/` 执行。下面的 `89` **仅示例**，必须按设备、Toolkit 和本专题支持范围选择：

```bash
cmake -S . -B build/sm_89/Release -G 'Unix Makefiles' \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build/sm_89/Release --config Release --parallel 2
```

`-S` 指定当前源码目录，`-B` 指定独立构建目录；`CMAKE_BUILD_TYPE` 在配置时选择优化/调试设置，`--parallel 2` 限制并行构建任务数。配置与编译属于同一个 compile 动作，不包含执行生成程序。

实际建议使用本目录的动作脚本执行并记录这些命令：

```bash
CUDA_ARCH=sm_89 bash commands/compile/run.sh
# 需要切换配置时单独执行；并非每个专题的 Debug 配置都已可用
CUDA_ARCH=sm_89 bash commands/compile/run.sh Debug
```

- `CUDA_ARCH` 必填，脚本把 `sm_89` 转成 CMake 所需的 `89`，不自动查询或猜测架构。
- 可选位置参数为 `Release`（默认）或 `Debug`，分别传给配置和构建步骤。既有 CMake 的 Debug 可能仍采用优化加行号信息，不保证启用 CUDA `-G`。
- `JOBS` 控制并行任务数，默认 2；`CXX` 默认 `clang++`。MMA 另有 `TORCH_PATH`。
- 脚本按自身位置定位专题，不要求调用者先进入 build 目录；其他六个专题使用各自同名入口。

### 3. 观察日志，不把历史产物当成新结果

每个专题的 compile 动作只写自己的 `commands/compile/output.log`，记录展开的命令、参数、输出和最终 `exit_code`。同一动作重跑会覆盖自己的日志（包括切换 Release/Debug），其他专题日志不受影响；同一动作不要并发执行。尚未执行的动作不会生成日志。

共享产物位于各专题的 `build/<CUDA_ARCH>/<Release或Debug>/`，不放进 commands，也不复制源码。旧 `build/` 根目录中的缓存、二进制和既有 `log/`、`runs/` 等证据保持原样；新入口不会清理它们。没有时间戳目录、history、哈希、源码快照或自动归档。

先查看编译日志末尾的退出码：配置失败就停止，不继续构建；编译失败不能因旧二进制仍存在而认定有效。修改源码、头文件或参数后要显式重新 compile。以后进行普通运行或 NCU 采集时应使用同配置已有产物，并分别设置自己的动作入口与日志，不能在运行中偷偷重新编译。

## 本次验证范围

上述七个 CMake 编译入口在此前迁移时仅做脚本语法、引用和文档路径的静态检查，没有执行 CMake 或 CUDA 程序；本次文档修订未执行这些脚本。既有笔记中的数字、Python 源码中的常量以及旧 build 产物均不是本次验证的证据。

此前模板源码、脚本与文档修改仅通过静态文本检查和 36 个脚本的 `bash -n` 检查；未编译、运行、执行实验动作或 sanitizer，未生成新实验日志。上文及各课 `log/compile.log`、`log/run.log` 仅保留改前版本的验证记录，其中第 11 课为单卡跳过，不是当前模板接口的验证。Lab 5 五版源码与脚本已实现，目前仅完成静态检查，尚未 CUDA 编译、运行或采集；详情见 [lab05](05_global_memory/lab05.md)。

**历史 sanitizer 结果（仅限改前版本）：** 当时的临时入口覆盖 vector（N=257、block=128、运行时 stride=8）、transpose（三变体、N=65）、reduction（两变体、N=257）和 FMA（Chains=1/4/8、N=65、block=128、total=256），九组结果均 PASS；`compute-sanitizer --tool memcheck --error-exitcode 1 --leak-check full` 报告 0 errors、0 bytes leaked。临时入口已移除，这些历史结果不能验证当前 `template<int Stride>` 源码与调用，sanitizer 下的时间也不是性能结果。第一课 [RESULTS.md](01_ncu_basics/RESULTS.md) 的历史 Vector Add 正文与旧 compile-only 证据同样不用于验证当前源码；原始日志不改写。CUTE/CUTLASS/MMA 等七个编译入口及其项目保持不变。
