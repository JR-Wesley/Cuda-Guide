# 实验 04｜有 Warp 驻留，为什么仍然没有指令可发？

## 从一个必须等待的乘加说起

先看一个数：**FMA（融合乘加）**把 `x*a+b` 作为一个操作计算，只做一次舍入，`fmaf` 是单精度版本。考虑 `x=fmaf(x,a,b)` 连做 8 次：下一次需要上一次的 x，所以不能把这 8 次当成彼此独立的任务。若有四个互不依赖的 x，等待其中一个结果时就可能安排另一个。这是本课的核心问题：**限制性能的是驻留数量、结果依赖，还是执行管线已经很忙？**

先简短回顾执行关系：kernel 是 GPU 上由许多线程执行的函数；线程组成 block（块），本次所有块组成 grid；硬件 SM 接收并执行块，块内每 32 个线程组成一个 warp。warp 的执行状态和资源已分配且尚未完成，叫**驻留/Active**；**Occupancy** 是驻留 warp 数占硬件上限的比例，不是计算忙碌率。上一课的静态资源预算回答 SM 能容纳多少 warp，本课再问其中多少 warp 的下一条指令真的准备好了。驻留 warp 可能在等前一条计算的结果，也可能已经就绪、只是本周期没有被选中；这两种情况需要的解释不同。我们先改变一个线程内部的独立计算数量，检查能否填补等待空档，再用需要线程协作的求和认识另一类等待。目标是从程序的数据依赖提出可检查的假设。**发射**是调度器把某个 warp 的下一条指令交给执行路径，**Stall** 是该指令因条件未满足而暂不能发射；一个 warp 在等时，别的 warp 仍可能在工作，所以最大的 Stall 百分比不能直接当成全部耗时的来源。

本课用固定总工作量的 FMA 链实验连接 [lec04](lec04.md) 第 2–5 节，再用归约补例认识块内同步。前文已简短回顾第三课的执行单位，下面随例子解释新增概念，不需要先学所有 Stall 指标。

## 1. 画出一个线程：最小输入先预测依赖，不猜性能

`float` 是 32 位浮点类型，占 4 字节（B）；`out[i]` 表示输出数组第 i 个元素，下标从 0 开始。N 是有效输出数。先说明这个计算实际产生什么。FMA 链是对一个数反复应用 `x ← x*multiplier+addend` 的递推；这里不从输入数组逐项取数，而是给每个线程生成初值，再传入共同的乘数、加数和总递推次数。kernel 输出长度为 N 的 out 数组，每个有效线程计算一个 out[i]，不同线程不读写彼此的中间结果。若一个线程有多条链，它保存多个独立的 x，分别递推，最后把它们相加成为该线程的一个输出。

例如只看线程 0、总共做两次 FMA：单链从 `0.50390625` 开始，先计算 `fmaf(0.50390625,1.0001f,0.0001f)`，结果约为 `0.50405664`，再拿这个新结果做第二次，约为 `0.50420704`。这两个近似值用于说明递推，不是预设的打印输出。若分成两链，则从 `0.50390625` 和 `0.5078125` 各做一次 FMA 后求和。总 FMA 仍是两次，但不再必须等第一条链的结果才能开始第二条；最后的和也自然不等于单链结果。

Host（主机，即执行准备工作的 CPU 端）的 `run_fma` 分配 N 个 GPU 输出位置，按向上取整的块数启动 `fma_chain<Chains>`（尖括号中是编译期模板参数，编译器据 Chains 生成相应版本，不是运行时命令行选项），把 N、总次数、乘数和加数作为参数传入。没有输入数组上传，因为初值由线程和链编号计算。这里“上传/下载”分别指 CPU→GPU / GPU→CPU 的数据拷贝。重复启动后，host 下载全部输出，用 CPU 按同样的初值、逐链递推和最终求和检查每项。选择这个 kernel，是为了让大量重复运算主要围绕线程自己的中间结果展开，使“前一结果何时可用”成为明确的问题，而不是把一次数组读取的等待也混入算法设计；最终是否保留了预期的指令依赖，仍要静态检查。

**目的与代码：** 先读 [main.cu](kernels/main.cu)，默认 `use_reduction=false` 选择 FMA；再读 [fma.cuh](../kernels/fma.cuh) 的 `fma_initial`、`fma_chain`。本步只做纸上推导。

把每线程总 FMA 暂想成 8，线程 i=0：

Chains 是一个线程保留的独立累加链数，rounds 是每条链的递推次数。`f` 后缀表示单精度常量；`j=0..3` 表示四个链编号。箭头图只画数据先后关系，不表示真实时间轴。

| Chains | 初值 | 每链 rounds | 总 FMA | 最终输出 |
|---:|---|---:|---:|---|
| 1 | `0.5+1/256=0.50390625` | 8 | 8 | 第 0 链最终值 |
| 4 | `0.5+(j+1)/256`，j=0..3 | 2 | 8 | 四条链最终值之和 |

线程 i=1 的每个初值再加 `1/65536`。源码中的 `0x1p-16f` 与 `0x1p-8f` 是十六进制浮点写法，分别表示这两个二进制分数。乘数为 `1.0001f`、加数为 `0.0001f`，是传给 kernel 的参数。

```text
一条链：A0 → A1 → A2 → ... → A7
四条链：A0 → A1
        B0 → B1
        C0 → C1
        D0 → D1
最后再把 A1、B1、C1、D1 相加
```

**解释：** 箭头表示前一结果是后一输入。这是 **RAW（Read After Write，先写后读）依赖**：后一步必须读到前一步新写出的值。不同链递推阶段互不依赖，但最终求和有依赖；不能说“增加链数消除全部依赖”。四链输出不应与单链输出相等，正确性应各自对照 CPU 参考。

**理解检查：** 同样 8 次 FMA，8 链时每链几轮？1 轮；但多了 7 次最终加法。若把每链轮数也固定为 8，总工作就不是同一份了。

## 2. 编译最小例，认识全量核对和计时范围

**目的：** 先核对逻辑，再放大工作量。仅在未来实操时把入口改为 `n=16, block=256, chains=1, total_fmas_per_thread=8`，保持 `use_reduction=false`。本课没有命令行参数，不能用 `--chains` 改配置。

以下是未来实操用的 Bash 命令，在工作区根目录下的 `ncu/04_warp_scheduling/` 执行，不是在 PowerShell 中原样执行。先分别确认 GPU/驱动、编译器、分析器版本，再编译、运行；实际验证范围见文末：

```bash
nvidia-smi
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
nvcc --version
ncu --version
mkdir -p build log
nvcc -std=c++17 -O3 -lineinfo -arch=sm_89 kernels/main.cu -o build/lesson04
./build/lesson04
```

`nvidia-smi` 查看 GPU 与驱动；`--query-gpu=... --format=csv` 选择设备名、compute capability（计算能力，即 GPU 架构版本）、驱动并按逗号分列。`nvcc` 是 CUDA 编译器，NCU（Nsight Compute）是单 kernel 性能分析器，`--version` 显示工具版本。`nvidia-smi` 显示的 CUDA 版本不是本机 Toolkit 版本。反斜杠 `\` 在后续 Bash 示例中表示续行，`./` 表示当前目录。

`sm_89` 只是示例，须先按实际计算能力及 Toolkit 支持替换；查询字段不支持时按设备型号核对。`-std=c++17` 选语言并支持编译期分支，`-O3` 优化，`-lineinfo` 保留源码行映射而不是调试编译，`-arch` 选架构，`-o` 指产物；`mkdir -p` 创建可已存在的目录。

**代码与观察：** `run_fma` 分配 GPU 输出 → 共享 [time_kernel](../kernels/common.cuh) 启动 → 下载 → CPU 重算并核对全部 N 项 → 释放。应看到配置 `N=16 Chains=1 total/thread=8 rounds=8`，以及 `Check: PASS, checked=16, mismatches=0`。程序不逐项打印 FMA 输出，不要期待看到步骤 1 的整张数值表。

CPU 使用同样的 `fmaf` 次序；`abs` 是绝对值，`1e-5` 表示 `10⁻⁵`；只有两侧都是有限数，且 `abs(gpu-reference) <= 1e-5 + 1e-4*abs(reference)` 才通过。绝对容差照顾接近零的数，相对容差随参考值大小缩放。然后只改 chains=4，重编运行；核对也应针对四链参考值，不用单链输出作答案。

**计时解释：** 公共函数先执行 10 次预热并同步，再用 CUDA Event（GPU 时间标记）包围 20 次启动，返回平均毫秒（`ms=10⁻³ s`）。预热是先执行若干次、让首次执行的特殊影响不直接进入后续平均；同步是等此前 GPU 工作完成再继续。FMA 打印字段是 `average_kernel_ms`，不是乘法入口的 `event_average_ms`。时间不含分配、下载和 CPU 核对，可能含提交间隙；分析器下的该时间不是独立基线。

**异常检查：** 检查 total>0 且能被 chains 整除、chains 在 1..64、是否重编；编译失败不运行旧产物，核对失败先查第一处错误，不继续性能比较。

## 3. 固定工作量，只改变独立链数

小例确认了每种链数都能算对，接着才把每线程递推次数增大，让这种依赖在执行中重复出现。保持每线程 FMA 总数不变，是为了避免把“多做了几倍乘加”误读成并行性效果；同时固定 N 和 block，避免增加其他 warp 来替多链隐藏等待。若多链缩短时间，一个合理假设是同一线程获得了更多可穿插的计算，但这还只是候选解释，因为循环次数、末尾求和和寄存器需求也随之改变。

**目的与配置：** 最小例通过后，恢复主实验 `n=1<<14`（16384）、`block=256`、`total_fmas_per_thread=1024`，先 chains=1。`1<<14` 是整数左移，即 2 的 14 次方。之后固定 N、block、每线程总 FMA、设备和编译参数，只改 chains=1/4/8。

| chains | 每链轮数 total/chains | 每次启动总 FMA | grid |
|---:|---:|---:|---:|
| 1 | 1024 | 16777216 | 64 |
| 4 | 256 | 16777216 | 64 |
| 8 | 128 | 16777216 | 64 |

每种配置都重新执行步骤 2 的 nvcc 命令，再普通运行三次：

```bash
./build/lesson04
./build/lesson04
./build/lesson04
```

**预期与原理：** 先核对全部 16384 项，再记录时间范围。**ILP（指令级并行）**是在同一 warp 的指令序列中安排互不依赖的工作，例如 A 链等待结果时先算 B 链；它不要求同一时刻真的发射两条。增加链数可能提供 ILP，也可能增加寄存器压力（线程同时要保存的中间值更多，而 SM 的寄存器总量有限）、减少驻留 warp；64 个块是否足够铺满设备须看实际 SM 数，不能预设所有 GPU 都充足。

**FLOP** 是一次浮点运算，FLOPs 表示运算次数；`GFLOP/s` 表示每秒 `10⁹` 次浮点运算。逻辑 FMA FLOPs 为 `2*N*total=33554432`（一次 FMA 按乘、加两次浮点运算计），不含每线程末尾 `Chains-1` 次加法、索引和初始化。各版本写 N 个 float，即逻辑输出 65536B，不代表实测 DRAM 字节数。外层循环轮数不同，不能说机器指令工作完全相同。

**对照边界：** 若改成 `total=1024*chains`，每链固定 1024 轮，总工作随链数增加。这是另一组 fixed-rounds（固定轮数）实验，不能把耗时比称为本组 fixed-work（固定工作）加速。

**异常检查：** 多链不快不是失败，先保留原始数值；若平均时间为零先检查规模与分辨率，不计算无穷 GFLOP/s。改变 N 也会改变总工作，不与改 chains 混在一组归因。

## 4. 用同一组报告区分驻留、就绪和发射

普通计时只能说变化有没有发生，不能说明等待在哪里。单链下一轮要读上一轮的结果，所以一个 warp 即使已经驻留，也可能暂时不能推进。调度器用 Scoreboard 等机制跟踪某些未完成操作及依赖，像给寄存器标上“新值尚不可用”；固定延迟计算还可能使用不同的等待机制。多链则可能在同一条指令流中提供另一个已备齐输入的运算。下一条指令输入、资源条件都满足时叫 **Eligible（就绪）**，实际被选中发射叫 **Issued**。Active 的 warp 即使在等也仍计入驻留；Eligible 只是其中已准备好的候选。由此推导的观察顺序是：先确认驻留工作够不够，再看可就绪工作是否增加，最后看是否实际发射并缩短时间。只看占用率会遗漏依赖，只看某种等待比例又可能遗漏其他 warp 正在填补空档。

**目的：** 在普通运行正确后分析“为什么”。**Section** 是相关指标组，**metric** 是具体计数或派生指标。LaunchStats 描述启动配置，SpeedOfLight 是相对参考峰值的吞吐概览（不是时间份额），Occupancy 描述驻留比例，SchedulerStats 描述就绪与发射，WarpStateStats 描述等待状态。先确认本机支持以下指标组，再对当前 chains=1 采集：

```bash
ncu --list-sections
ncu --section LaunchStats --section SpeedOfLight --section Occupancy \
    --section SchedulerStats --section WarpStateStats \
    --kernel-name-base function -k 'regex:fma_chain' -s 10 -c 1 \
    --replay-mode kernel --cache-control all --clock-control base \
    -o log/d_chains1.ncu-rep ./build/lesson04
ncu --import log/d_chains1.ncu-rep --page details
ncu --import log/d_chains1.ncu-rep --page raw --csv
```

目录仍为本课目录。Section 是相关指标组；函数基础名过滤匹配模板 `fma_chain`，`regex:` 表示正则匹配。`-s 10 -c 1` 选择第 11 次匹配启动，应用仍提交 30 次。`--replay-mode kernel` 指为采不同指标重复执行选中的 kernel；`--cache-control all` 请求重放采集前清缓存；`--clock-control base` 请求基础时钟控制，支持与权限以工具输出为准。`-o` 导出；`--import` 读报告，details 看分组、raw/csv 核对具体指标单位。

4/8 链分别重编、普通运行核对后，用相同采集命令，报告名改为 `d_chains4.ncu-rep` / `d_chains8.ncu-rep`。已有同名文件先明确另存，不强制覆盖。

**阅读顺序：**

1. LaunchStats 确认 grid=64、block=256、目标模板与寄存器；先排除错误配置。
2. Occupancy 看理论/实际驻留与资源限制，不把占用率当发射效率。
3. SchedulerStats 看 Active、Eligible、Issue Activity（发射活动）；抄指标完整名、单位和周期分母，注意 SM 与 SMSP（SM 子分区）聚合范围不同。
4. WarpStateStats 看等待类型，先问是否真的缺少可发射工作，再研究原因。
5. 对照同配置 Duration 与普通运行分布，两种计时分别记录。

**周期**是硬件时钟的一拍，不是固定长度的秒数。Latency 是操作从指定起点到结果可用的延迟；Throughput 是单位时间能处理多少工作。等待一个结果时换另一个 warp 的独立工作来填空档，叫 TLP（线程级并行）；同一 warp 内换独立链则是 ILP，都不把原操作的延迟变短。

**纸上调度模型，不是硬件实测：** 假设一条 FMA 的相关结果 4 周期后可用、每周期只能发一条。单链在周期 0 发 A0，最早周期 4 发 A1；独立四链可能在 0/1/2/3 发 A0/B0/C0/D0，然后周期 4 发 A1。也可以用四个不同 warp 填空档。这说明 TLP 和 ILP 都可能隐藏延迟，但不说明真实 GPU 固定有 4 周期延迟或四链必然最优。

理解等待名之前先分路径：**L1TEX** 处理 Global/Local 等访问，**MIO** 处理 shared（同块线程共用的片上存储）等部分操作，也涉及特殊指令。Long/Short Scoreboard 分别关联这些路径的依赖等待，不按统一延迟阈值命名。**Not Selected** 是已经就绪但本周期调度器选了别的 warp，不是输入缺失。

**异常与反例：** Eligible/Issue 未上升，可能其他 warp 已足够隐藏延迟，或寄存器/管线/工作规模限制了收益。时间缩短但 Issue 不变，循环控制指令减少也是候选解释。不要预言纯 FMA 一定出现 Long/Short Scoreboard 中某一项：等待分类随指令与架构而变；Long Scoreboard 高不证明 DRAM 饱和，Not Selected 高可能只是已就绪 warp 富余。Stall 的归一化周期或样本比例不能相加为 kernel 总时间。

## 5. 用静态指令检查源码意图，而不是凭循环写法下结论

报告中的调度变化需要与真正执行的指令对上。C++ 中四个独立变量表达了四条链的意图，却不能保证编译后每轮的指令排列、循环控制开销或资源占用恰好符合想象。所以这里从动态读数退回二进制：确认递推仍存在、每条链读写哪些寄存器，再决定“独立工作更多”是否足以解释现象。静态检查能验证结构，不能替代运行时间和硬件等待的测量。

**目的与代码：** `fma_chain` 内层 `#pragma unroll` 请求展开独立链，外层 `#pragma unroll 1` 避免完全展开递推循环；全部链参与输出，初值依赖线程/链号。编译器仍可能重新安排代码，所以源码不能证明最终指令结构。

在本课目录，对当前已明确重编的同一产物做只读检查：

```bash
cuobjdump --dump-resource-usage build/lesson04
cuobjdump --dump-sass build/lesson04
```

`cuobjdump` 是 CUDA 二进制检查工具；前一选项显示资源信息，后一选项反汇编 SASS（GPU 机器指令）。这不是 GPU 运行，也不是计时。每种 chains 的结果要对应当时的配置，不用旧二进制解释新源码。

**看什么与如何理解：** 找到目标 `fma_chain`，检查 FMA 指令及循环回跳是否存在、不同累加器是否形成独立寄存器依赖、末尾是否汇总全部链；同时看寄存器资源与可能的 local load/store。Local Memory 是每线程私有地址空间，不保证在片上；load/store 是读取/写入，spill 是寄存器装不下时将中间值暂存到 Local 再读回，可能增加访存。不能仅看到某条访存就宣布 spill，需要结合其地址用途与编译资源。工具缺失就标“未检查”，不宣称源码结构已验证；多链寄存器增加/占用下降也都必须有实际证据。

## 6. 可选补例：用归约认识 Barrier，不和 FMA 混表

**Barrier（会合屏障）**要求规定的参与线程到齐后再继续；它不仅是耗时项，也是建立跨线程数据交接顺序的正确性手段。FMA 的依赖局限在一个线程自己的数值里，不需要同块线程互相等待。为了理解 Barrier（屏障同步）为什么会成为等待原因，现在换成多个线程合作求和。Reduction（归约）把多个输入合成一个结果；本例接收长度为 N 的浮点数组 input，但 GPU 输出的不是唯一总和，而是每块一个部分和组成的数组 partial_sums。最终总和由 CPU 把这些部分和再相加得到，因此不能把 GPU 阶段时间当成完整求和流程时间。

先想 N=8 的输入 `[-3,-2,-1,0,1,2,3,-3]/8`，答案为 `-3/8=-0.375`。GPU 仍启动一个 256 线程块：前八个线程各读一个数，余下线程把自己的值设为 0，但都参加必要的同步。shared 版先把 256 个值放入同块可访问的共享数组；第一轮让线程 0 加位置 128 的值，线程 1 加位置 129 的值，依此类推。下一轮把前 128 项压成前 64 项，直到位置 0 汇总整块。这个小输入的高位置都是 0，所以前几轮不改答案；到 offset=4 时前四项变为 `[-2,0,2,-3]/8`，offset=2 后前两项为 `[0,-3]/8`，最后得到 `-3/8`。

某轮写出的部分和正是下一轮要读的数据，且写入者和读取者可能不是同一个线程。块同步确保上一阶段的写入对下一阶段读取可见且参与线程已到齐，不能照搬 FMA 的“各算各的”思路把它删掉。shuffle（寄存器间的跨 Lane 取值指令，Lane 是 warp 中的线程位置）版则先在每个 32 线程 warp 内交换值求和，只把八个 warp 的结果放进共享内存，块同步后由 warp 0 汇总。两种实现算相同的块部分和，但交换方式和同步结构不同，适合观察协作等待，不是 FMA 多链实验的继续计时。

Host 的 `run_reduction` 生成输入并上传，按 `ceil(N/256)`（向上取整，例如 257 项需 2 块） 分配部分和数组并启动所选 kernel，下载后先逐块与 CPU 对应区间的和比较，再检查全部部分和相加后的总和。先查每块而不只查最终答案，可以避免不同块的错误恰好抵消。完成上述推导后再读 [reduction.cuh](../kernels/reduction.cuh)。设置 `use_reduction=true` 后，main 的 block/chains/total **不再控制该算法**，共享实现固定 block=256。

先只设 n=8、shuffle=false。输入为 `(-3,-2,-1,0,1,2,3,-3)/8`，手算和为 `-0.375`。其余线程以 0 参与，不提前退出：

- shared 版把值放入共享数组，按 offset=128、64、…、1 两两相加，每阶段用 `__syncthreads()` 等待同块线程到齐。
- shuffle 版用 `__shfl_down_sync` 从同一 warp 中较大编号的 lane 取得值来求和；调用的参与掩码说明哪些线程必须参与，不能让掩码内线程随意跳过。它在一个 warp 的 lane 间传值求和，再把 8 个 warp 的和写到共享内存，块同步后由 warp 0 完成第二级归约；最后一级中多余 lane 仍以 0 参与。

仍按步骤 2 重编并运行，应看到 `check=PASS block_errors=0`、partials=1、CPU/reference 总和 -0.375。这里是预测，程序会核对全部块部分和及最终总和。

小输入只是检查零填充、同步参与和答案。放大后，shared 版的逐级共享内存交换与多次块同步仍然重复发生，shuffle 版则把大部分交换限制在 warp 内；由此可以提出“等待与发射活动是否随协作结构改变”的假设。但少几次块同步不保证更快，还要看额外指令、数据交换和实际到达情况，不能预先把所有时间差归给 Barrier。

最小核对后固定 n=16384，分别只改 shuffle=false/true，每次重编、普通运行，再分别采集：

```bash
ncu --section LaunchStats --section SpeedOfLight --section SchedulerStats \
    --section WarpStateStats --kernel-name-base function -k reduction_shared \
    -s 10 -c 1 --replay-mode kernel --cache-control all --clock-control base \
    -o log/reduction_shared.ncu-rep ./build/lesson04
ncu --import log/reduction_shared.ncu-rep --page details
```

true 时过滤器和报告名都改为 `reduction_shuffle`。不能沿用 `fma_chain`，否则匹配不到目标。`partial_gpu_ms` 只测 GPU 部分和阶段，排除 CPU 最终求和，不能当全流程归约耗时。

**观察与边界：** 比较 Barrier 等待、发射活动和时间，但两版同时改变同步次数、数据交换与指令结构，不是纯粹只删一条 barrier 的对照。Barrier 高还可能来自线程到达不齐，不能简单删除同步。核对失败先看每块错误、边界参与与是否重编，不解读性能。

## 7. 验收：用证据排除替代解释

FMA 与归约分表，记录源码配置、普通时间范围、报告 Duration、Registers、Occupancy、Active/Eligible/Issue、等待指标的完整名称/单位，以及工具警告与退出码。未测量写明原因，不填零。

自测：

1. 8 链、total=1024 为什么是每链 128 轮？因为总次数守恒；不是每链 1024。
2. 多链时间缩短能否立即证明依赖等待减少？不能，还要排除循环控制、资源与工作分布变化，并检查指令证据。
3. Active=8、平均 Eligible=1 能否推出每周期都发射？不能，平均值可来自部分周期有多个、部分周期为零。
4. 为什么不能把所有 Stall 百分比当作“浪费总时间”？warp 状态、采样、调度器周期与 kernel 墙钟时间的统计对象和分母不同，其他 warp 还可能同时工作。

## 当前验证状态、入口与记录边界

**当前验证：** 已执行 `CUDA_ARCH=sm_89 bash commands/compile.sh`（nvcc 13.4、C++17、`-O3 -lineinfo -arch=sm_89`），见[编译日志](log/compile.log)，exit 0；随后执行 `bash commands/run.sh`，见[运行日志](log/run.log)，PASS/exit 0。只验证本轮入口配置，未测试全部参数；profile 仅通过 `bash -n`，未执行 NCU、NSYS、workers 或 SASS 检查，无性能结论。[旧 compile-only 日志](log/legacy_commands/compile/source_check.log)仅作历史证据。

理解底层命令后，在本课目录分别执行：

```bash
CUDA_ARCH=sm_89 bash commands/compile.sh
bash commands/run.sh
bash commands/profile.sh
```

脚本按自身位置定位，均无位置参数；仅 compile 必填 `CUDA_ARCH`（按设备替换）。compile 固定 C++17、`-O3 -lineinfo`，生成 `build/lesson04`，编译前移除旧目标，失败即停止；run/profile 缺产物报错，不隐式编译，改常量/共享头后须显式重编。profile 采 LaunchStats、SpeedOfLight、Occupancy、SchedulerStats、WarpStateStats，过滤 fma_chain/reduction_shared/reduction_shuffle 的函数基名，适用于已编译选中的分支，不自动遍历配置；skip 10/count 1，固定 kernel replay/cache all/clock base。

新记录分别写 `log/compile.log`、`log/run.log`、`log/profile.log`，含展开命令、输出与 `exit_code`；同动作重跑覆盖新日志，对照需先另存。报告 `log/profile.ncu-rep` 已存在则拒绝覆盖，NCU 失败即停止，报告内容由实验者检查；静态检查仍使用本文独立命令，必要时单独保存到本课 `log/`，不新增 commands 子目录。旧证据原样迁至 `log/legacy_commands/<原相对路径>`，命令不改；旧 compile-only **不代表改后源码或新脚本**。