# 实验 01｜从 16 个乘法到第一份 NCU 报告

## 为什么先做这件小事

假如程序只写了一句 `c[i]=a[i]*b[i]`，我们能否说 GPU 执行了多少工作、运行了多久、显存读了多少字节？这其实是三个问题：源码给出**逻辑工作量**，计时给出一个指定范围的时间，硬件计数器才提供特定内存层次的流量。本课用能手算的乘法，把这三者分开，再学习如何读一份 Nsight Compute（简称 NCU，GPU 内核性能分析器）报告。

完成后应能回答：程序算什么？报告选中了哪次启动？一个数字属于启动属性、动态计数还是派生比率？配套阅读 [lec01](lec01.md) 第 1–7 节；其中 Vector Add 是理论例子，**当前入口实际做 Multiply（乘法）**。


## 1. 在纸上预测：线程和数组元素如何对应

先把要运行的程序当成一个普通数组计算来理解。它接收两个浮点数组 a、b，输出一个长度为 N 的数组 c；第 i 个答案只由第 i 对输入相乘得到，不需要其他线程的结果。例如 N=4、连续读取时，输入 a 为 `[0, 0.125, 0.25, 0.375]`，b 为 `[2, 2, 2, 2]`，输出便是 `[0, 0.25, 0.5, 0.75]`。这里不是把整个数组相乘成一个数，而是得到四个各自独立的乘积。

GPU 上的函数称为 **kernel（内核）**，一次提交执行称为 **launch（启动）**。这次启动中许多**线程**分别执行函数，各有自己的编号；线程分为 **block（块）**，多个块组成 **grid（网格）**。本例一个有效线程算一个 c[i]，先由块号和块内线程号求出自己的全局编号，再判断是否越过 N。CPU 一侧负责造出这两个输入数组，分配 GPU 数组并上传数据，按足够覆盖 N 项的块数启动 kernel，最后下载 c 并逐项与 CPU 预期比较。我们选择它作为第一例，是因为答案和每线程工作都能手算；一旦报告中出现复杂数字，仍能回到“每项读两个数、乘一次、写一个数”这个不变事实，而不必同时排查复杂算法。

**目的与代码：** 打开 [入口](kernels/main.cu)，确认 `default_n=16`、`block=256`、`stride=1`；再读 [vector_multiply.cuh](../kernels/vector_multiply.cuh) 的 `vector_mul` 和初始化循环。本步只阅读，不运行。

```cpp
i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) c[i] = a[i * stride] * b[i * stride];
```

代码中的 `blockIdx.x` 是块号，`blockDim.x` 是每块线程数，`threadIdx.x` 是块内线程号。例如每块 256 线程时，块 1 的线程 0 负责 i=256。`stride` 是相邻有效输入位置之间的元素间隔；本课固定为 1，即连续读取。输入有效位置为 `a[i*stride]=(i%1024)*0.125f`、`b[i*stride]=2`。`float` 是本例占 4 字节的单精度浮点数，`f` 表示单精度常量。

| 块号 | 块内线程号 | i | N=16 时的行为 |
|---:|---:|---:|---|
| 0 | 0 | 0 | `0 * 2 = 0` |
| 0 | 1 | 1 | `0.125 * 2 = 0.25` |
| 0 | 15 | 15 | `1.875 * 2 = 3.75` |
| 0 | 16 | 16 | 条件不成立，不读写数组 |

**预期与解释：** C 应为 `0, 0.25, …, 3.75`。网格大小用整数式 `(n-1)/block+1` 向上取整；N=257 时需要两块，第二块只有线程 0 有效，`c[256]=64`。不是“每块都必须有 256 个有效输出”。

**理解检查：** N=513 要几块？为什么不能删 `i<n`？答案是 3 块；尾块其余线程会越界。若预测不同，先检查索引从 0 开始以及整数除法，而不是猜测 GPU 调度顺序。

## 2. 顺着 main 读完数据往返，再显式编译

**Host / Device** 分别指 CPU 一侧与 GPU 一侧。指针保存地址，图中的 `d_a` 等指向 GPU 数组，不能由 CPU 当普通数组直接读写。**异步提交**表示 CPU 交出 GPU 任务后可继续运行，提交返回不保证计算完成；**同步**是在需要时建立完成顺序或等待，例如下载并检查结果前必须确保计算已完成。

**目的与代码：** 读 `run_vector`，按以下数据流标出 CPU/GPU 边界：

```text
CPU 初始化 a、b → GPU 分配 d_a、d_b、d_c → 上传 a、b
→ 启动 vector_mul → 下载 c → CPU 核对全部 N 项 → 释放 GPU 内存
```

`cudaMalloc` 分配 GPU 空间，`cudaMemcpy` 在两侧复制，`cudaFree` 释放。共享 [common.cuh](../kernels/common.cuh) 的 `CUDA_CHECK` 在调用失败时打印位置和错误并退出；不能把错误输出当正常测量。

**目录与命令：** 下文均为 Bash，在 `ncu/01_ncu_basics/` 执行。先独立确认环境，不修改系统配置：

```bash
nvidia-smi
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
nvcc --version
ncu --version
mkdir -p build log
nvcc -std=c++17 -O3 -lineinfo -arch=sm_89 kernels/main.cu -o build/lesson01
```

CUDA Toolkit 是包含编译器、头文件和库的开发工具集；`nvcc` 把 CUDA 源码编译为程序，`ncu` 负责性能采集。`nvidia-smi` 查看设备/驱动；查询字段若不支持，用设备型号查官方 compute capability（计算能力，即架构能力版本）。`sm_89` **只是计算能力 8.9 的示例**，执行前按设备与 Toolkit 支持替换。`-std=c++17` 选 C++ 版本，`-O3` 优化，`-lineinfo` 保留源码映射而不是开启设备调试，`-arch` 选目标架构，`-o` 指输出文件；`mkdir -p` 创建目录且允许目录已存在。

**预期与异常：** 编译应成功生成 `build/lesson01`，但编译不是正确性验证。找不到共享头先检查是否位于本课目录、共享文件是否完整；不支持架构先检查 nvcc 版本和设备，不能把旧二进制的存在当本次编译成功。改源码、共享头或编译选项后都要重编。

## 3. 先核对小输入，再建立普通运行基线

小输入能看清答案，却不足以代表 GPU 连续处理大量数据的能力：N=16 时只启动一个块，而且大部分线程不做乘法；启动和提交的固定成本可能比有用计算更显眼。因此先用 16 项检查数值，再用 257 项检查尾块，最后才增大 N 观察时间。这个顺序把“程序是否正确”和“规模是否适合测量”分开，避免拿一个错误程序的较短时间当作优化成果。

**目的：** 不通过分析器，先证明当前程序算对，并认识打印时间的边界。仍在本课目录执行：

```bash
./build/lesson01 16
./build/lesson01 257
./build/lesson01 1048576
./build/lesson01 1048576
./build/lesson01 1048576
```

唯一可选参数是 N，合法范围 `1..67108864`。本组只改 N，固定算法、block、stride 和编译条件。

**预期观察：** 每次检查 `PASS`、`checked=N`、`mismatches=0`、grid/block；N≤16 才打印每个结果，257 不打印 `c[256]`，但会全量核对。这里的数都是预测，不是本次运行结果。程序用精确相等检查，是因为这些二进制分数及乘 2 的结果可精确表示，不能推广到任意浮点计算。

**如何解释时间：** **预热（warmup）**是正式计时前先运行几次，让初始化、缓存和运行频率等状态有机会稳定，不保证它们一定固定。**CUDA Event** 是按顺序插入 GPU 工作队列的时间标记；等结束标记完成，再读两标记之间的时差。`time_kernel` 先启动 10 次预热并同步，再在两个 Event 之间提交 20 次启动，返回区间毫秒数除以 20。例如区间是 0.4 ms，批量平均就是 `0.4/20=0.02 ms/次`；ms 是千分之一秒。`event_average_ms` 不含分配、上传/下载、CPU 核对和预热，可能含提交间隙；它不是 CPU API 调用时间，也不保证等于孤立单次 kernel 的时间。记录三次大输入的实际数值与范围，不填预设性能。

**从源码推导工作量：** 每次启动做 N 次乘法、逻辑读写 `12N` 字节（读两个 4B、写一个 4B）。每输入数组容量为 `storage=(N-1)*stride+1`；`allocated_bytes=8*storage+4*N` 是 CPU 或 GPU **一侧**三数组的总容量。N=1048576、stride=1 时每侧 12 MiB，`1 MiB=2^20` 字节。容量表示占了多大空间，逻辑流量表示一次启动按算法读写多少数据；两者数值恰好相同不表示意义相同，也都不是实测 DRAM（片外显存）流量。

**异常检查：** 出现 FAIL 先看第一处索引/数值错误及 CUDA 报错，不继续性能分析；时间过短或为零，先查规模、打印精度与计时范围，不除以零算带宽。

## 4. 只采一次匹配启动，先看配置再看性能

普通运行已经给出了重复调用的时间，但它没有告诉我们 GPU 的各部分在忙什么。现在才引入分析器：应用会多次启动同一个乘法，而我们希望先研究其中一个明确的对象，不把整段应用、一次启动和工具重放混为一谈。先验证报告的网格是否确实覆盖这次的 N，再读耗时，才能保证讨论的是刚才核对过的计算。

**Metric** 是单个指标，如耗时；**Section** 是围绕一个问题组织的指标组，如访存分析；**Set** 是多个指标组的预设集合。**Replay（重放）**是分析器为收齐指标重复执行工作，每一轮采集叫 **pass**。例如程序的第 11 次启动可能被工具执行两轮，仍只是一份目标启动报告，并不是应用循环多了两次。

**目的：** 把应用的 30 次启动与 NCU 的采集对象分开。先确认本机有哪些指标组：

```bash
ncu --list-sets
ncu --list-sections
ncu --set basic --kernel-name-base function -k vector_mul \
    -s 10 -c 1 --replay-mode kernel --cache-control all --clock-control base \
    -o log/profile.ncu-rep ./build/lesson01 1048576
ncu --import log/profile.ncu-rep --page details
ncu --import log/profile.ncu-rep --page raw --csv
```

目录仍为本课目录。`--set basic` 选基础集合；`--kernel-name-base function -k vector_mul` 按函数基础名筛选；`-s 10 -c 1` 跳过 10 次匹配启动，只采第 11 次，**应用仍提交 30 次**。`--replay-mode kernel` 允许单 kernel 重放；`--cache-control all` 请求工具清缓存；`--clock-control base` 请求基础时钟控制，是否支持要看实际提示。`-o` 导出，`--import` 读取已有报告；`--page details` 看分组解释，`raw --csv` 看原始名称、单位与数值的表格。

**寄存器**是线程保存当前中间值的快速小存储，由编译器分配；例如保留一次乘法的输入与结果。寄存器不是显存数组，也不能按源码变量数直接计数。**吞吐（Throughput）**是每秒或每周期处理的工作量；报告中的峰值百分比是相对于工具所用硬件能力参照的比率，不是程序答案中有效计算的百分比。**Duration** 就是该采集对象的持续时间。

**阅读顺序与理解：**

1. `LaunchStats`：先核对函数和 grid=4096、block=256。寄存器数是编译后资源属性，不能从 C++ 变量个数猜。
2. `SpeedOfLight`：看 Duration 和吞吐概览。低百分比只是线索，不自动证明“计算不足”或“显存瓶颈”。
3. Raw：找到 `gpu__time_duration.sum`，抄下单位和定义；`.sum` 是该指标定义的聚合，不是把 replay 的所有耗时相加。

**异常检查：** 没报告先查工具退出信息、计数器权限和过滤器；无权限不等于程序算错，也不授权更改系统权限。旧报告仍在不代表本次成功。采集前确认没有同名文件，不强制覆盖；需要保留时先明确另存。

## 5. 增加一个观察维度，而不是立刻下瓶颈结论

从每线程行为已经能算出逻辑流量，但线程发出的读取不一定每次都到显存取数：中间缓存可能保留数据，硬件也可能按比一个 float 更大的单位传输。因此“做了 N 次乘法”和“显存传了多少字节”不是同一道算术题。基础报告若提示内存活动值得关注，下一步应补充各存储层次的流量证据，而不是仅凭乘法少就断言它受显存带宽限制。

**目的：** 检验“逻辑字节数就是显存字节数”是否成立。固定步骤 4 的全部条件，仅把采集内容改成访存组，并用不同文件名：

```bash
ncu --section MemoryWorkloadAnalysis --kernel-name-base function -k vector_mul \
    -s 10 -c 1 --replay-mode kernel --cache-control all --clock-control base \
    -o log/vector_mul_memory.ncu-rep ./build/lesson01 1048576
ncu --import log/vector_mul_memory.ncu-rep --page details
```

**预期观察：** 寻找 L2 缓存（片上复用数据的存储层）和 DRAM（显存）读写流量；不预言它们必然等于 `12N`。**事务粒度**指硬件按多大的数据单位处理一次请求/传输；若看到 **Sector**，本课程相关 L1/L2 指标通常以一个 32 字节扇区计数，仍须查指标定义。例如 8 个相邻 float 恰在同一对齐 32 字节区域内（区域起始地址是 32 的整数倍），可能涉及同一个 Sector，而不是 8 个 Sector。缓存、粒度和写入路径都会改变硬件层次上的流量。若要计算实测流量/时间，必须使用**同一报告、对应作用域**的数据，即分子字节数和分母秒数针对同一受测工作与条件，不拼接另一轮 Event 时间。结果单位为 B/s；若写 GB/s，则按 `1 GB=10^9 B` 换算。

**异常与练习：** Section 不支持时先看 `--list-sections`，具体指标用 `ncu --query-metrics` 查询，不把缺失填零。用一句话分别解释“分配 12 MiB”“逻辑读写 12 MiB”“报告 DRAM 字节数”；不能互换才算理解。

## 6. 理解底层命令后，再用已有独立脚本

以下入口已存在，仍从本课目录调用；编译、运行、采集互相独立：

```bash
CUDA_ARCH=sm_89 bash commands/compile.sh
bash commands/run.sh 16
bash commands/profile.sh 1048576
```

`CUDA_ARCH=...` 仅在 compile 时必填，执行前按实际设备与 Toolkit 支持替换；compile 固定 C++17、`-O3 -lineinfo`，先移除旧 `build/lesson01`，编译失败即停止。run/profile 缺可执行产物会提示先编译，不隐式构建；已有产物不等于与新源码一致。run 默认 N=16，profile 默认 N=1048576，唯一可选参数是 N。脚本按自身位置定位本课，分别写 `log/compile.log`、`log/run.log`、`log/profile.log`，含展开的命令、输出与 `exit_code`；同动作重跑覆盖新日志，保留独立实验前须明确另存。profile 使用步骤 4 的 basic、vector_mul、skip 10/count 1 和 replay/cache/clock 配置，报告固定为 `log/profile.ncu-rep`，已存在则拒绝覆盖，NCU 失败即停止，报告内容由实验者检查。步骤 5 的扩展仅保留手工命令，不增加 memory/scheduler 等跨课动作脚本；细分实验见第 05 课。普通基线与 NCU 下 Event 时间务必分开。

**本课验收：** 不看答案说明 `-s 11` 选第几次启动，并解释为什么无法由此推算 pass 数；答案是第 12 次，pass 数由实际指标组合、架构和工具决定。

## 当前验证状态与旧证据边界

**当前状态：** 本次 stride 模板源码仅通过静态文本检查，相关脚本 `bash -n` 通过；未编译运行，旧日志不能验证当前模板源码与调用。

**改前版本的历史验证：** 当时已执行 `CUDA_ARCH=sm_89 bash commands/compile.sh`（nvcc 13.4、C++17、`-O3 -lineinfo -arch=sm_89`），见[编译日志](log/compile.log)，exit 0；随后执行 `bash commands/run.sh`，见[运行日志](log/run.log)，PASS/exit 0。当时只验证所选入口配置，未测试全部参数；profile 当时仅通过 `bash -n`，未执行 NCU、NSYS、workers 或 SASS 检查，无性能结论。[旧 compile-only 日志](log/legacy_commands/compile/source_check.log)仅作历史证据。

[RESULTS.md](RESULTS.md)、旧 VectorAdd 日志、`build/vector_add`、`build/vector_add_basic.ncu-repz`、`log/legacy_commands/basic/report.ncu-rep` 和 `runs/` 都是历史证据，不能验证当前 Multiply。本文未改写这些记录。环境细节可参考 [ENVIRONMENT.md](ENVIRONMENT.md)，实际执行仍须重新确认环境。
