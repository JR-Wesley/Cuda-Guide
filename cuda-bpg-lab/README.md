# CUDA BPG 实验室：从正确结果到可信优化

**BPG = Best Practices Guide**，即 NVIDIA 的 [CUDA C++ 最佳实践指南（13.1 版）](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-c-best-practices-guide/index.html)。这个目录是配套的**本地学习与实验项目**，不是 NVIDIA 官方仓库，也不是安装 CUDA Toolkit 的工具说明。这里用很小的 CUDA 程序练习一个问题：**先保证算对，再弄清楚测到了什么，最后才讨论为什么快或慢。**

**先备知识**：C/C++ 数组和下标、编译与运行的区别；知道 CPU（主机）和 GPU（设备）各有内存即可。第一次接触线程、warp（通常由 32 个线程组成的执行组）、共享内存时，以对应阶段的例子学习，不要求先读完整本指南。项目范围为单 GPU、传统 CUDA Core 的 FP32 等实验，目标架构为 SM70–SM89；不包含 Tensor Core/SM90+ 等专题。历史实测在 RTX 4060 Laptop / 原生 SM89 路径上，不是所有设备的性能承诺。

## 第一次来：只做一件事

从工作区的 `cuda-bpg-lab/` 目录开始。下面的 `89` 只是示例，**先查自己的 GPU 计算能力**；`CUDA_ARCH=sm_89` 是脚本的架构格式，直接用 CMake 时写 `-DCMAKE_CUDA_ARCHITECTURES=89`。没有 GPU 时只读示例，或先运行 `python3 commands/host-build/run.py`，再运行 `python3 commands/host-test/run.py`；不要把主机测试当作 GPU 校验。

1. **看环境**：`nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader` 和 `nvcc --version`。`nvidia-smi` 显示驱动侧信息，`nvcc` 是编译器；两者版本不必相同。可选 `python3 commands/env/run.py` 把各工具的结果记录到 `commands/env/log/`。先看日志：个别工具探测失败不代表整个 `env` 返回失败。
2. **从能手算的输入开始**：打开 [Stage 1 的 N=8 / N=257 实操](labs/01_measurement/README.md)，先预测哪几个线程会写结果，再按其中的 CMake 命令**只构建** `vector_add`。CMake 把 `.cu` 源码编成可执行文件；构建成功不等于算对。
3. **运行并核对**：按照 Stage 1 的小输入命令看 `PASS`，再用 `CUDA_ARCH=sm_89 python3 commands/verify/run.py vector_add` 检查更多边界。这里 `--n` 是元素数，改变它会影响线程尾部和数据规模；失败先看输出或本次 `commands/<action>/log/` 的 `commands.jsonl`/对应日志。
4. **只改变一个变量**：理解 Event 与端到端计时范围后，用 `CUDA_ARCH=sm_89 python3 commands/run/run.py vector_add --variant scalar --n 4096 --warmup 5 --repeats 21 --iterations 1`；保持其他参数相同，再将 `--iterations` 改为 `100`，对比两次 `samples.csv`（重跑会覆盖 `commands/run/log/`，先阅读或自行保留第一份结果）。`--iterations` 是每批启动次数，不是 GPU 线程数；时间变化不能直接证明缓存或 DRAM 流量。

每步的**预期结果**、实际命令、要读的源码位置和不符合预期的检查顺序都在 [Stage 1 指南](labs/01_measurement/README.md)。不要第一次就运行全部 15 个程序或所有 profiler。

## 七阶段怎样连接

每阶段先读本阶段 README 的入门小步骤，再读实现和 [analysis.md](labs/01_measurement/analysis.md) 看证据与反例，最后独立完成 `exercises.md` 再读 `answers.md`。下一阶段复用上一阶段的**问题和测量方法**，不是照抄其数字。

| 顺序 | 从上一阶段继承什么 → 本阶段新增的问题 | 先打开 |
| --- | --- | --- |
| 1 测量 | 从零开始：结果对吗？Event 时间和端到端时间有什么区别？ | [向量加法与 Copy](labs/01_measurement/README.md) |
| 2 全局内存 | 固定结果与计时方式 → 相邻线程读写的地址如何变化？ | [连续、跨步、错位 Copy](labs/02_global_memory/README.md) |
| 3 存储层次 | 已能画地址图 → 全局/共享/local 与缓存如何影响观测？ | [转置、工作集与 local](labs/03_memory_hierarchy/README.md) |
| 4 执行配置 | 已知数据路径 → block、寄存器和 occupancy（可驻留线程比例上限）如何取舍？ | [归约与寄存器压力](labs/04_execution_config/README.md) |
| 5 指令控制 | 已知资源约束 → 编译器实际生成哪些 PTX/SASS 指令与分支？ | [整数除法、展开与分支](labs/05_instructions_control/README.md) |
| 6 异步执行 | 已区分单次计时 → 两条工作流是否真的重叠、依赖是否正确？ | [传输、流水线与并发](labs/06_async_execution/README.md) |
| 7 综合 | 固定工作、正确性和测量协议 → 五步 GEMM 优化何时有效？ | [传统 FP32 GEMM](labs/07_gemm/README.md) |

全路线与章节对照见 [课程地图](docs/roadmap.md)。[测量规范](docs/methodology.md)给出计时区间、CSV 统计和正确性边界；[分析方法](docs/analysis-workflow.md)教你把“猜想 → 对照 → 观察 → 修正”写清楚。它们供查阅，不要求先读完才开始第一步。

## 操作按功能拆开

在项目根目录按需运行一个入口；例中 `vector_add` 可以替换为 [各阶段中的目标名](docs/roadmap.md)。各脚本内部只执行本动作；[入口—源码—日志关系图](docs/tool-map.md)说明它们如何连接。共享的 [记录器](scripts/session.py)、[构建辅助](scripts/build_support.py)和 [验证辅助](scripts/verification.py)不是学习 CUDA kernel 的前置条件。

| 我想做什么 | 命令示例 | 证据边界 |
| --- | --- | --- |
| 只查看环境 | `python3 commands/env/run.py` | 探测结果逐项记录；个别失败须看日志 |
| 只构建主机测试 | `python3 commands/host-build/run.py` | 只编译，无需 CUDA |
| 仅主机工具测试 | `python3 commands/host-test/run.py` | 不验证 GPU 计算 |
| 构建一个目标 | `CUDA_ARCH=sm_89 python3 commands/build/run.py vector_add` | 只构建；主机/help 测试在 verify 中执行 |
| 更多 GPU 边界/CLI 校验 | `CUDA_ARCH=sm_89 python3 commands/verify/run.py vector_add` | GPU CTest、CSV 及非法参数检查 |
| 普通运行与计时 | `CUDA_ARCH=sm_89 python3 commands/run/run.py vector_add --n 257` | 程序校验 + 当前运行的计时 |
| 静态编译结果 | `CUDA_ARCH=sm_89 python3 commands/static/run.py vector_add` | PTX/SASS/资源，不是运行指标 |
| 内存/同步诊断 | `CUDA_ARCH=sm_89 python3 commands/sanitizer/run.py vector_add` | 仅本次输入与工具覆盖范围 |
| CUDA 时间线 | `CUDA_ARCH=sm_89 python3 commands/nsys/run.py vector_add --n 1048576` | 插桩计时与普通基线分开 |
| 单 kernel 指标 | `CUDA_ARCH=sm_89 python3 commands/ncu/run.py vector_add --n 1048576` | 当前环境缺 `ncu` 时为 Not Tested |

`run/nsys/ncu` 后面的 `--n` 等是程序参数。必须显式设置实际 `CUDA_ARCH=sm_XX`，不自动探测。先用 `commands/build/run.py` 编译所需目标；其余动作只读取共享 `build/sm_XX/`，缺产物则提示先编译，不隐式构建。修改源码或编译配置后要主动重编译；不做自制哈希/缓存判定，CMake 仅在显式构建动作中增量工作。

每个动作日志固定在 **`commands/<action>/log/`**，包含 `invocation.json`、`commands.jsonl`、原始输出、`status.txt`，以及该动作的 CSV 或分析器报告。同动作重跑会覆盖整个 log（含最新失败），不同动作互不覆盖；源码仍只在 `labs/`、`common/`，不复制源码或二进制，不新增时间/随机目录、快照、哈希或 history。原 `results/` 完整保留为历史证据。

本次迁移未运行 GPU；下方实测均为迁移前记录，不代表当前入口已复验。

本地 CMake 需要 CMake ≥ 3.24、C++17 编译器和可支持目标架构的 CUDA Toolkit；Python 入口只使用标准库。若仅想看到最底层的构建命令，可按 [Stage 1](labs/01_measurement/README.md) 先单目标执行 `cmake -S . -B build/sm_89 -DCMAKE_CUDA_ARCHITECTURES=89 -DBPG_LABS=01_measurement`，再 `cmake --build build/sm_89 --target vector_add`，然后回头看独立脚本如何附加取证。

以前的 `lab.sh` / `scripts/lab.py` 是已移除的混合入口；旧记录里的原始命令及源码快照仍描述**当时的**运行，不是当前命令。历史覆盖和限制见 [进展](docs/progress.md)、[交付审计](docs/completion-audit.md)：其中 GPU CTest、Sanitizer 通过数来自旧证据，Nsight Compute 硬件计数器仍未测量。

**迁移前入口的历史局部回归（2026-11-06，SM89）**：[`host_test.py` 的记录](results/20261106T143755Z_host-test_6ji_dtt7/status.txt)、[`run.py vector_add` 的记录](results/20261106T143807Z_run_3ohuyh6p/status.txt)、[`static.py vector_add` 的记录](results/20261106T143817Z_static_w2r9gr2v/status.txt)、[`verify.py vector_add` 的记录](results/20261106T143821Z_verify_8x7sqytl/status.txt)均通过；此外使用各阶段 README 的小输入，[S2 global_copy](results/20261106T145027Z_run_sjlr2bub/status.txt)、[S3 transpose](results/20261106T145029Z_run_e9q0kl5f/status.txt)、[S4 reduction](results/20261106T145031Z_run_trx_l0zw/status.txt)、[S5 integer_division](results/20261106T145032Z_run_scnh_tzl/status.txt)、[S6 host_transfer](results/20261106T145033Z_run_km_nem6m/status.txt)、[S7 gemm](results/20261106T145034Z_run_em18zi2v/status.txt)各运行一个版本并通过程序校验。这**不是**其他 14 个程序的完整 `verify`/Sanitizer/Profile 复验，短输入的时间也不是性能排名。
