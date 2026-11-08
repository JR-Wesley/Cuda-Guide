# 实验 02｜算法没改，为什么测量读数会改变？

## 本课的问题：分清“程序变慢”和“测量变贵”

用秒表量一次跑步，与要求运动员重复跑几次、每次记录不同项目，所付出的总时间不同。**NCU（Nsight Compute）**是 GPU 函数的性能分析器，GPU 函数叫 **kernel**。**硬件计数器**是累计事件的统计机制，例如记录显存读取字节；一个 **指标（metric）**就是这样的观测量或由它算出的比率。硬件未必能同时统计全部所需事件，因此 NCU 可能把一次目标 kernel 分成多个采集 **pass（轮次）**，每轮重新执行来收集一部分。本课不优化乘法，只改变**观测方式**，判断读数改变来自哪里。

本课对应 [lec02](lec02.md) 第 1–7 节：重复执行、Replay、Cache/Clock Control 和短 kernel。lec 后半的 Empty/Copy/Accumulate、Range 示例是理论扩展，**不是当前入口支持的模式**。

建议先做 [实验 01](../01_ncu_basics/lab01.md) 的索引与全量核对；下面也会回顾当前计算和计时范围，可独立跟读。**Replay（重放）**指工具为采指标而重新执行工作，与应用自己循环启动要分开。

## 1. 先证明每次启动做的是同一件事

本课继续计算逐元素乘法，而不是引入一个新算法。输入是两个数组 a、b，输出 c 与它们按项对应；N=4 时，`[0, 0.125, 0.25, 0.375]` 分别乘以 2，得到 `[0, 0.25, 0.5, 0.75]`。**线程（thread）**是执行函数的一份任务，有自己的编号和局部数据；本课一个线程用全局编号 i 找到自己的两个输入，只写 c[i]，越过 N 的线程不访问数组。线程之间没有交换数据，也不读取 c 的旧值，所以再次启动仍是在重算同一个答案，而不是在上次答案上继续迭代。

**Host / Device** 是 CPU / GPU 两侧；CPU 负责初始化、上传、提交启动、下载和核对。线程先组成 **block（块）**，多个块组成一次启动的 **grid（网格）**；每块 256 线程时，N=257 要两块，因为一块装不下。一次提交 kernel 执行称 **launch（启动）**，CPU 提交通常**异步**返回，即返回时 GPU 不一定算完，使用结果前需要等待相关工作完成（同步）。输入在这些启动之间没有被修改。这个简单而确定的行为正适合研究测量机制：改变分析器选项以后，如果成本或流量读数改变，我们可以先排除“应用本来就让下一次计算不同”的解释。它也限定了实验能力：这个例子能帮助认识重复采集，却不能检验一个会修改自身输入的程序是否被正确恢复。

**目的与代码：** 读 [main.cu](kernels/main.cu) 的 `n=16, block=256, stride=1`，再读共享 [vector_mul / run_vector](../kernels/vector_multiply.cuh)。本课没有命令行参数，不接受 `--mode`、`--iters` 或 N；改变输入必须改常量后重编。

全局索引是 `i=blockIdx.x*blockDim.x+threadIdx.x`：依次为块号、每块线程数和块内线程号；`i<n` 防止尾块越界。`stride` 是相邻有效输入之间的元素间隔，本课固定 1 即连续读取。

**纸上预测：** 一个块的前 16 个线程有效；`a[i]=(i%1024)/8`、`b[i]=2`，所以输出为 `0, 0.25, …, 3.75`。每次是覆盖写 `c[i]`，不是累加；运行 30 次不会把答案乘 30。

**原理：** CPU 初始化 → 上传固定输入 → 重复启动 → 下载 → 检查全部 N 项。目标内核不依赖其他 GPU、CPU 线程或外部网络的实时响应，因此适合作为单 kernel replay 的入门对象。但是，它没有 `c+=a` 这样的输入状态依赖，**不能用本例输出不变来验证分析器的状态恢复能力**。

**检查：** 把 N 预测为 257，grid 应为 2；若不同，回看 `(n-1)/block+1`。这里只做推导，不把预测写成实测。

## 2. 小输入核对，再单独放大 N 建基线

**目的：** 先检查正确性，再避免用 16 个元素判断吞吐。以下 Bash 命令均在 `ncu/02_replay_and_measurement/` 执行。

```bash
nvidia-smi
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
nvcc --version
ncu --version
mkdir -p build log
nvcc -std=c++17 -O3 -lineinfo -arch=sm_89 kernels/main.cu -o build/lesson02
./build/lesson02
```

`nvidia-smi` 查询 GPU/驱动；CUDA Toolkit 是提供编译器、头文件与库的开发工具集，`nvcc` 是 CUDA 编译驱动；两个 `--version` 分别查询编译器和 NCU 版本。`sm_89` 仅为示例，先按实际 compute capability（设备架构能力）与 Toolkit 支持替换；查询字段不支持时按型号核对官方规格。`-std=c++17` 选语言，`-O3` 优化，`-lineinfo` 保留源码映射而不是调试编译，`-arch` 选架构，`-o` 指输出；`mkdir -p` 确保构建目录存在。

**预期观察：** `PASS`、`checked=16`、`mismatches=0` 和逐项输出。若编译失败，不运行旧产物；若核对失败，先看首个 CUDA 错误和错误索引，不采性能。

通过后只把入口 `n` 改为 `1048576`，block/stride 不变，**重新执行同一 nvcc 命令**，再普通运行：

```bash
./build/lesson02
./build/lesson02
./build/lesson02
```

**记录与解释：** 记录三次 `event_average_ms`，排序后取中间值为中位数，两端为最小值、最大值；例如 0.02、0.03、0.09 ms 的中位数是 0.03 ms，不能只保留最快值。**预热（warmup）**是计时前先运行工作，让初始化、缓存与运行频率等状态有机会稳定，不保证状态固定。共享 [time_kernel](../kernels/common.cuh) 先预热 10 次并同步，再用 CUDA Event 包围 20 次启动并除以 20。**Event** 是插入 GPU 有序工作队列的时间标记，等结束标记完成后取时差；若区间 0.4 ms，平均是 `0.4/20=0.02 ms/次`，1 ms 是千分之一秒。时间不含分配、拷贝和 CPU 核对，可含提交间隙。每项读两个 4 字节 float、写一个，所以每次逻辑流量为 `12N` 字节。N=1048576 时三数组每侧容量为 12 MiB（`1 MiB=2^20` 字节）；分配容量、按算法计算的读写量和 DRAM（显存）实测字节数是三个概念，不能因为前两者数值相等就当成第三者。

**异常检查：** 时间为零或波动大先查输入是否重编、时间精度、后台工作与环境。不要直接把批量平均称为单次孤立 kernel 延迟。

## 3. 对照一：只增加观测内容

有了不受分析器干预的基线，现在让同一份乘法接受两种详细程度的观察。GPU 用于记录事件的硬件计数器有限，一组指标未必能在一次执行里同时收齐；工具可能重复执行目标，每轮收集其中一部分。因此首先提出的假设应是“记录更多内容可能让整条采集命令更久”，而不是“乘法本身多做了工作”。同时记录命令总时间和报告内核时间，才能区分这两种解释。

**Duration** 是报告所选 GPU 工作的时长；**wall time** 是外部秒表量整条命令从开始到结束的经过时间，包括初始化、等待和导出报告。**Section** 把围绕一个问题的指标组织成组，**Set** 是多个组的预设集合。例如增加访存组会多看流量，但并未给乘法增加算术。

**目的：** 检验“多测一些指标”是否增加采集成本，而不是假设算法变慢。保持 N、二进制、输入、目标启动、缓存/时钟策略不变。

先查询本机支持，再逐条执行（仍在本课目录）：

```bash
ncu --list-sets
ncu --list-sections
ncu --query-metrics
 time ncu --set basic --kernel-name-base function -k vector_mul -s 10 -c 1 \
    --replay-mode kernel --cache-control all --clock-control base \
    -o log/vector_mul_basic.ncu-rep ./build/lesson02
 time ncu --set basic --section MemoryWorkloadAnalysis \
    --kernel-name-base function -k vector_mul -s 10 -c 1 \
    --replay-mode kernel --cache-control all --clock-control base \
    -o log/vector_mul_memory_all.ncu-rep ./build/lesson02
ncu --import log/vector_mul_memory_all.ncu-rep --page details
```

`time` 是 Bash 的整条命令计时，关注其 `real`（经过时间），不是 kernel Duration。`--set` 选指标组集合，`--section` 增加访存组；`--kernel-name-base function -k vector_mul` 按函数基础名筛选，`-s 10 -c 1` 选择第 11 次匹配启动。应用仍有 10 次预热和 20 次计时启动；pass 数不能从 30 推出。`--replay-mode kernel` 按需重放目标内核；`--cache-control all` 请求工具在采集 pass 前清缓存，`--clock-control base` 请求基础时钟控制，支持情况看实际提示。`-o` 导出报告，`--import` 只读已有报告，`--page details` 显示分组详情。

**要看什么：** 对两次分别记录工具实际显示的 pass 数、`time` 的 real、报告 Duration。更多指标可能需要更多 pass，但不保证一定增加，必须看实际输出。若 real 增长而 Duration 接近，支持的是“采集成本增加”，不是“乘法算法变慢”。NCU 内打印的 Event 时间受到采集干预，不能替代步骤 2 的普通运行基线。

**异常检查：** 无报告先查函数匹配、版本与计数器权限；不支持 Section 就记录未采集原因，不填零。已有同名报告要先明确另存，不强制覆盖，也不能误读旧文件为新结果。

## 4. 对照二：只改变工具是否主动清缓存

上一个对照研究观察的数量，现在研究观察开始时的数据状态。应用会反复读取同一 a、b，之前启动留下的数据可能仍在缓存中；工具若在采集前主动清缓存，就可能改变目标启动需要从显存重新取得多少输入。于是，即使指令和答案都不变，访存路径仍可能不同。为了把这个假设单独拿出来检查，这里保留较完整的访存指标，只改变清缓存请求，不能同时退回 basic 集合。

**目的：** 固定步骤 3 第二条的所有条件，仅将 `all` 改成 `none`，报告换名以保留对照：

```bash
 time ncu --set basic --section MemoryWorkloadAnalysis \
    --kernel-name-base function -k vector_mul -s 10 -c 1 \
    --replay-mode kernel --cache-control none --clock-control base \
    -o log/vector_mul_memory_none.ncu-rep ./build/lesson02
ncu --import log/vector_mul_memory_none.ncu-rep --page details
```

**L2** 是服务多个 GPU 计算单元的二级缓存；**命中**表示查询时在这一层找到了数据。**Working set（工作集）**是计算涉及的数据范围，例如这里 a、b、c 三个数组，而不只是 a。**load** 是读取数据的指令。认识这些词后再做预测。

**先预测：** 相同输入被多次访问，如果数据能被缓存复用，none 可能减少 DRAM 读取；但 N=1048576 的完整三数组容量为 12 MiB，必须和实际设备 L2 容量、访问行为一起判断，不能宣布它在所有 GPU 都是“小工作集”。

**如何理解：** `all` 请求分析器清缓存，不保证每个 load 都来自 DRAM；`none` 是不主动清缓存，不是关闭缓存，也不保证全命中。联合查看 L2 命中相关指标、DRAM 读写字节和 Duration，抄清单位；命中率是相应命中事件数除以查询事件数，要核对是按请求还是按 Sector（缓存数据块，本课程相关指标通常为 32 字节）统计，不能混用分子分母。逻辑 `12N` 与硬件字节数分开。不能拿这轮字节数除以另一轮普通 Event 时间声称实测带宽。

**异常与下一步：** 没差异先检查控制请求是否生效、实际工作集和流量，而不是判定“缓存没用”。扩展时可选另一个 N，但要先重编并固定它，重新做该 N 的 all/none 一对；不要一边换 N 一边换策略后直接比较归因。

## 5. 对照三：只改变时钟控制请求

**时钟频率（clock）**表示硬件每秒经历多少周期，Hz 是周期/秒，1 GHz 是十亿周期/秒。计算侧与显存侧频率还可能不同。缓存状态之外，同样数量的工作也可能因硬件运行频率不同而耗时不同。应用预热可能影响设备状态，但“已经预热”不等于“频率已经固定”。所以这一步回到清缓存的基线，只改变工具的时钟控制请求，检验时间变化是否伴随可观察的时钟差异；不能承接上一步的 none 缓存配置，否则一次改变了两个条件。请求未生效时，这一对照的解释能力也相应受限。

**目的：** 以步骤 3 的 memory/all 为基线，固定其他选项，只改 `base` 为 `none`：

```bash
 time ncu --set basic --section MemoryWorkloadAnalysis \
    --kernel-name-base function -k vector_mul -s 10 -c 1 \
    --replay-mode kernel --cache-control all --clock-control none \
    -o log/vector_mul_clock_none.ncu-rep ./build/lesson02
ncu --import log/vector_mul_clock_none.ncu-rep --page details
```

**观察与解释：** `base` 请求基础时钟控制，`none` 请求工具不干预；后者不是锁定某个自然频率。记录实际警告、可获得的时钟信息、Duration 与重复采集的变异，不能把一次差异全部归因于锁频。

**异常检查：** 设备不支持或权限不足时如实记下，不修改系统时钟、驱动或计数器策略来“跑通”。温度、功率和其他用户任务仍可能影响结果。

## 6. 对照四：只改变重放单位

前面都允许工具围绕一个 kernel 收集指标，但一次应用执行还有分配、上传、预热、下载和核对。这些 Host（CPU 一侧）工作不属于乘法 Duration，却属于整条命令的成本。若为了每个采集 pass 都重启完整应用，就可能把这些准备与收尾也重复一遍。因此最后把重放单位从 kernel 换成应用，并重新固定 all/base：我们观察的是“为了得到报告重复了多大范围的工作”，不是比较两个乘法算法的速度。

**目的：** 回到 memory/all/base 基线，仅改 kernel replay 为 application replay：

```bash
 time ncu --set basic --section MemoryWorkloadAnalysis \
    --kernel-name-base function -k vector_mul -s 10 -c 1 \
    --replay-mode application --cache-control all --clock-control base \
    -o log/vector_mul_application.ncu-rep ./build/lesson02
ncu --import log/vector_mul_application.ncu-rep --page details
```

**原理与观察：** kernel 模式按需重复目标内核；application 模式按需重新启动完整应用，每轮都重新分配、初始化、预热、核对并退出。固定输入和启动序列有利于跨轮匹配，但实际是否匹配成功仍须看工具输出。只需要一个 pass 时不一定能观察到多次应用启动，不预设启动次数或耗时倍数。

**异常检查：** 匹配失败先查源码/二进制是否一致、启动配置和顺序，不放宽条件后假装同一对象。当前源码没有 profiler start/stop（通知工具开始/停止观察的接口）或 NVTX（给代码范围加名称的标记接口），不直接套用 lec 的 `--profile-from-start off` 或 Range 实验命令。

## 7. 汇总：写观察，不替工具编造因果

**退出码**是命令结束时返回的状态，通常 0 表示成功，但还须检查输出中的警告/错误和报告是否有效。表内 real、Duration 各自注明单位（如秒、微秒），不同采集不能混拼流量与时间。

每一对比较填写以下字段，未执行写“未测量”，不能填零：

| 配置 | 唯一改变项 | 实际 pass | 命令 real | 报告 Duration | L2 / DRAM 证据 | 警告/退出码 |
|---|---|---|---|---|---|---|
| basic 与 memory | 指标内容 | 待测 | 待测 | 待测 | 待测 | 待测 |
| all 与 none | 清缓存请求 | 待测 | 待测 | 待测 | 待测 | 待测 |
| base 与 none | 时钟请求 | 待测 | 待测 | 待测 | 待测 | 待测 |
| kernel 与 application | 重放单位 | 待测 | 待测 | 待测 | 待测 | 待测 |

理解检查：pass 数增加但 Duration 接近，是否证明算法变慢？不是。all/none 的 Duration 不同，是否已经证明 L2 导致？也不是，还需流量、命中和控制条件证据。每条结论至少写“观察事实 → 候选解释 → 还缺什么证据”。

## 当前验证状态、入口与记录边界

**当前状态：** 本次 stride 模板源码仅通过静态文本检查，相关脚本 `bash -n` 通过；未编译运行，旧日志不能验证当前模板源码与调用。

**改前版本的历史验证：** 当时已执行 `CUDA_ARCH=sm_89 bash commands/compile.sh`（nvcc 13.4、C++17、`-O3 -lineinfo -arch=sm_89`），见[编译日志](log/compile.log)，exit 0；随后执行 `bash commands/run.sh`，见[运行日志](log/run.log)，PASS/exit 0。当时只验证所选入口配置，未测试全部参数；profile 当时仅通过 `bash -n`，未执行 NCU、NSYS、workers 或 SASS 检查，无性能结论。[旧 compile-only 日志](log/legacy_commands/compile/source_check.log)仅作历史证据。

理解底层命令后，在本课目录依次显式执行：

```bash
CUDA_ARCH=sm_89 bash commands/compile.sh
bash commands/run.sh
bash commands/profile.sh
```

三个脚本均无位置参数，按自身位置定位本课。只有 compile 要求显式 `CUDA_ARCH`（架构须按设备替换），固定 C++17、`-O3 -lineinfo`，生成 `build/lesson02`；编译前移除旧目标，失败即停止。run/profile 缺产物提示先编译，不隐式构建；改常量或共享头后须重编。profile 固定 basic、vector_mul、kernel replay/cache all/clock base、skip 10/count 1，作为测量对照起点；后续改变指标、缓存、时钟或 replay 时仍执行本文对应的明确命令，不引入配置分派器。

日志为本课 `log/compile.log`、`log/run.log`、`log/profile.log`，记录展开的命令、输出与最终 `exit_code`；重跑覆盖各自新日志，需保留对照时先另存日志及报告。profile 拒绝覆盖已有 `log/profile.ncu-rep`，NCU 失败即停止，报告内容由实验者检查。旧证据原样移至 `log/legacy_commands/<原相对路径>`，原始命令不改；旧 compile-only 只证明当时编译，**不代表改后源码或新脚本**。