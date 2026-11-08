# 实验 06：用共享内存中转，为什么还要多留一列？

## 本课要解决的问题

转置像把一排排座位改为一列列编号。输入按行连续存放，若相邻线程直接把这一行写到输出的一列，写地址就相隔整行。第五课只用向量乘法理解访存指标；本课从转置开始，将地址推导扩展到写端，并比较同一算法的不同实现。本课先用线程块共用的小方块中转数据，让输入和输出都能按行访问；再解释为什么小方块的物理布局仍会限制并行。

本课值得实际比较三个版本，是因为一次优化可能只是把代价从一个地方移到另一个地方：直接转置省去了中转，却让写地址分散；加入共享内存改善了全局写入，却增加中转、同步以及共享存储内部的争用；改变中转布局又可能消除争用。三个程序交付同一张输出矩阵，因而可以用同一份正确性标准约束优化，再分别寻找代价转移的证据。不需要先读第五课，本课会从一张小矩阵重新建立地址与线程的对应。

本课依次回答：转置下标怎样算？共享内存怎样交换线程分工？为什么 `tile[32][33]` 比 `[32][32]` 可能更好？理论配读：[lec06 第 2、6、8、9、12 节](lec06.md)。

前置知识：一维数组下标、二维矩阵行列。`float` 占 4B；行优先表示一行排完再排下一行，位置 `(row,col)` 对应 `row*N+col`。kernel 是 GPU 线程执行的函数，block 是线程块，warp 是 32 个线程的一组。global memory 是设备全局地址空间；shared memory 是同一 block 线程可共享的存储，不是不同 block 间的公共数组。bank 的分组机制将在步骤 4 随具体地址展开，不要求先背冲突口诀。B 是字节，下面所有矩阵下标从 0 开始。

下面的观察均为待验证预期，不是本次运行结果。

## 1. 先手算一个真正的小矩阵

转置的输入是一张按行存储的 N×N 浮点矩阵，输出是另一张同样大小的矩阵；它把行变成列，不修改数值。比如输入一行 `a b c d`，输出第一列就依次是 `a b c d`。这里输入输出不能共用一块存储：本例没有设计原地交换，某线程写出的值不能覆盖其他线程尚未读取的输入。

主机是负责准备工作的 CPU 端，设备是执行 kernel 的 GPU；上传/下载分别指 CPU→GPU / GPU→CPU 的拷贝。先看直接搬运的 naive（朴素）版本。主机把初始化好的输入上传到设备，另分配输出，用 `(32,8)` 个线程组成一个块，启动足够多的块覆盖整张矩阵；块数在两个方向都向上取整为 `ceil(N/32)`，ceil 表示向上取整，例如 N=65 时为 3。每块负责一个 32×32 区域，256 个线程各搬最多四个元素。例如 N=4 时，块内线程 `(tx=2,ty=1)` 在第一轮把输入 `(1,2)` 搬到输出 `(2,1)`，后面三轮已经越界，便不访问数组。主机随后拷回输出，逐个比较 `output[col*N+row]` 与 `input[row*N+col]`。所有版本都用这条标准，而不是互相比速度后假定结果正确。

这个版本适合做基线，因为数据始终由同一个线程读入并写出，容易检查。也正是这一分工产生了待验证的问题：同一 warp 相邻线程的 tx 依次增加，读的是相邻列，写的却是输出的不同行。读写都只有一个 float，并不意味着两端都能组成紧凑请求。这里的 **Request（请求）**是交给某个硬件单元的一项访存工作，不是一个线程或一个 float；**coalescing（合并访存）**是把同一 warp、一条指令的线程地址整理成对若干区段的访问。连续地址可能覆盖更少区段，分散地址则可能需要更多工作，但此时还不能据此直接推算显存流量。

**目的：** 在优化前先定义“正确”。读 [入口 main.cu](kernels/main.cu) 和 [matrix_transpose.cuh](../kernels/matrix_transpose.cuh) 的 `run_transpose`、`transpose_naive`。入口只有编译期常量，没有命令行参数；默认 n=65、variant=0。

本步先在纸上用 N=4。输入下标矩阵转置应为：

```text
输入下标                 输出应存放的输入下标
 0  1  2  3               0  4  8 12
 4  5  6  7               1  5  9 13
 8  9 10 11               2  6 10 14
12 13 14 15               3  7 11 15
```

实际值不是下标本身。runner 用 `(int(i%251)-125)*0.125f` 初始化，故输入 `(1,2)` 即下标 6 的值为 -14.875，必须到输出 `(2,1)` 即下标 9。核心语义是 `out[col*N+row]=in[row*N+col]`。

本步无需命令。若手算时输出仍与输入一样，先查是否把行列互换。理解检查：N=4 时输入下标 13 应去哪里？它是 `(3,1)`，输出 `(1,3)`，下标 7。后续可以只改入口 n=4、保留 variant=0，用步骤 3 验证；runner 只给全量核对结论，不逐项打印矩阵。

## 2. 从一个 tile 追踪数据由谁搬、由谁写

直接搬运的问题不在转置结果，而在“谁读谁就写”的分工。现在仍输入同一张矩阵、交付同一张转置矩阵，但让不同线程接力：一组线程先按输入行搬进共享小方块，全块等搬运完成，再由另一组线程按输出行取走。主机仍用相同的输入、块大小、网格和全量校验，只把启动的函数换成 `transpose_tiled<0>`，也就是 variant=1；尖括号里的 0 是编译期模板参数 Pad，表示每行不额外填充，`<1>` 则表示每行加一个 float 的空间，不是启动时多传一张矩阵；因此可以检查访问分工改变后的收益，而不改变任务。

沿用 N=4 的具体元素：值 -14.875 从输入 `(1,2)` 由线程 `(2,1)` 放进 `tile[1][2]`，同步后由线程 `(1,2)` 取出并写到输出 `(2,1)`。这里两个线程的坐标已经交换，但元素只搬一次进 shared、一次出 shared，不是每个线程独自完成一张小矩阵。接力的意义是让写出阶段相邻 tx 对应相邻输出列；预期全局写入更集中，代价则是共享读写和一次全块会合。是否更快必须连同这些新增工作一起测。

第三版 padded（variant=2）启动 `transpose_tiled<1>`。输入、输出、接力线程和主机校验都不变，仍把上述元素放进逻辑位置 `tile[1][2]`；只是每行实际预留 33 个 float，而不是 32 个。第 33 列不存放矩阵数据，不会多出输出元素。先知道它没有改变计算任务，再在步骤 4 推导为何改变物理行距能影响并行读取，而不是看到多一列就直接宣称提速。

**Barrier（会合屏障）**是让规定参与线程到齐再继续的同步点；`__syncthreads()` 用于同块会合，并使相关前序读写对后序访问满足所需可见性。若写入者还没准备好、另一个线程就读取，会读到不保证正确的数据。等待会合不等于线程已经结束。

**目的：** 明白 shared 不是自动加速缓存，而是代码显式安排的“换手区”。读共享头中的 `transpose_tiled<Pad>`。tile 是一次处理的小方块，本例大小 32×32；block 为 `(32,8)`，256 个线程，每线程循环处理 j=0/8/16/24 四行。

设块位置为 `(bx,by)`，线程块内位置为 `(tx,ty)`；代码中的 `blockIdx.x/y` 是块坐标，`threadIdx.x/y` 是块内线程坐标。二维线程块按 x 方向优先编号，所以 `(32,8)` 中每组 `tx=0..31`、固定 ty 的线程构成一个 warp：

| 阶段 | 具体下标 | 相邻 tx 的关系 |
|---|---|---|
| global 输入 | `in[(32*by+ty+j)*N + 32*bx+tx]` | 相邻 float |
| 写 shared | `tile[ty+j][tx]` | 同一行 |
| 全块同步 | `__syncthreads()` | 等其他线程写完并保证块内数据可见 |
| 读 shared | `tile[tx][ty+j]` | 同一列，换线程取走数据 |
| global 输出 | `out[(32*bx+ty+j)*N + 32*by+tx]` | 相邻 float |

不仅 tile 内行列互换，**tile 在整张矩阵的位置也互换**。两者缺一都会写错。

本步手算默认 N=65：grid 是 3×3，最右、最下是不完整 tile。输入 `(row=0,col=64)` 由输入 tile `(bx=2,by=0)` 搬入，写到输出 `(64,0)`，下标 4160。再自己追踪输入 `(64,1)` 到输出下标 129。

**为什么边界没有读到未初始化 shared？** 输出线程读取 `tile[tx][ty+j]` 对应输入 `(32*by+tx,32*bx+ty+j)`。输出的两个边界条件恰好也保证这个输入合法。越界线程可以不做 global 访问，却仍须到达全块 barrier（会合点）；不能为省事提前 return 绕过它。

若推导不通，先将“矩阵行列”“块坐标”“线程坐标”分开写。本步不需执行命令，也不要通过删同步试错。理解检查：为什么只转 tile 内下标，在 N=65 时会把跨 tile 的元素放错位置？

## 3. 分别编译和运行：先查正确性，不比小输入时间

**目的：** 让纸面映射经过全元素核对。以下 Bash 命令工作目录为 `cuda/ncu/06_shared_memory_and_transpose/`，相对工作区根目录为 `ncu/06_shared_memory_and_transpose/`。

未来执行前分别查看设备/驱动、Toolkit 编译器与 NCU 版本：

```bash
nvidia-smi
nvcc --version
ncu --version
```

`nvidia-smi` 查看 GPU 与驱动，`nvcc` 是 CUDA 编译器，`ncu` 是 Nsight Compute 单 kernel 分析器，`--version` 显示工具版本。另确认设备 compute capability（支持的架构版本）与 Toolkit 兼容。`nvidia-smi` 显示的 CUDA 版本不等于本机 Toolkit 版本；以下 sm_89 只是示例，须按设备替换。

先编译：

```bash
mkdir -p build log
nvcc -std=c++17 -O3 -lineinfo -arch=sm_89 kernels/main.cu -o build/lesson06
```

`./` 表示当前目录。`mkdir -p` 创建构建目录；`-std=c++17` 选择语言版本，`-O3` 优化，`-lineinfo` 保存源码行信息，`-arch` 选择 GPU 架构，`-o` 指定输出。编译失败先读首个错误，不继续使用旧二进制。

然后单独普通运行：

```bash
./build/lesson06
```

若按步骤 1 设置 N=4，先核对它；之后只改 n 回 65、重编运行。再固定 n=65，逐次改 variant=0/1/2，每次重编并运行。三个值分别为 naive、tiled（Pad=0）、padded（Pad=1）。预期每版 `check=PASS errors=0`；若失败，先查实际配置、重编状态和首个错误行列，不讨论性能。

runner 先将输出填成 NaN（非数值）哨兵，漏写位置不会等于有限参考值；之后精确比较所有元素，累计错误并报告首个位置。转置只搬数据，无浮点算术，因此无需误差容限。再固定某一版，单独改变 n=1/31/32/33/65 检查边界，预期全部通过；这些小尺寸的时间不用于带宽结论。

读 [common.cuh](../kernels/common.cuh)：`CUDA_CHECK` 报告 CUDA 错误；`time_kernel` 用 CUDA Event（GPU 时间标记）测量 10 次预热后 20 次启动的平均毫秒（`ms=10⁻³ s`）。预热是先执行若干次，不把首次执行的特殊影响直接计入后续平均；Event 是 GPU 工作序列里的时间标记，不是 CPU 秒表。输出 `kernel_ms` 排除分配、初始化、拷贝和 CPU 核对，但可能含提交间隙；小输入不代表设备已充分忙碌。

## 4. 只多一列，为什么 bank 分布变了？

前面已经说明 shared 能让两侧 global 地址连续，但还没证明中转本身高效。写入时线程沿 tile 的行排列，取走时却沿列排列：这正是第二个待检验的变化。共享内存不是单一服务窗口。**Bank** 是可并行服务的存储分组，常见模型有 32 个；**Word（字）**在这里固定指 4B 数据，不是任意类型的一个元素。连续字依次映射 bank 0、1、…、31，再回到 0。例如 float 元素 0 与 32 属于同一个 bank，却是两个不同位置。如果同一 warp 一次访问中多个不同字落到同一 bank，便需分轮服务，这叫 **bank conflict（bank 冲突）**；不同 bank 则可并行服务。完全同一字的读取可以广播同一份数据，不按不同字的冲突处理。padding 不减少元素搬运次数，而是尝试把这组列地址重新分散。

**目的：** 从下标推导 padding（填充空间）的作用，而不是记忆“共享内存一律加一列”。仍读 `transpose_tiled<Pad>`，不必执行命令。

对常见 32-bank、每 bank word 为 4B 的映射，假设 tile 基址映射到 bank 0，float 的相对 bank 编号如下；`%` 是除法余数，`r*(32+Pad)+c` 是行优先的元素/Word 偏移，乘 4 才是字节偏移：

```text
bank(tile[r][c]) = (r*(32+Pad)+c) % 32
```

固定一个 warp，ty 相同，tx=0..31。读取 `tile[tx][ty+j]` 时设固定列为 c：

| tx | Pad=0：word 偏移 / bank | Pad=1：word 偏移 / bank |
|---:|---|---|
| 0 | c / c | c / c |
| 1 | 32+c / c | 33+c / (1+c)%32 |
| 2 | 64+c / c | 66+c / (2+c)%32 |
| 31 | 992+c / c | 1023+c / (31+c)%32 |

Pad=0 时是 32 个**不同地址**落入同一 bank，预测经典 32 路冲突，不是“同地址广播”。Pad=1 时 bank 遍历一圈，预测消除这一列访问冲突；写 shared 的同行访问也仍分散到各 bank。

空间代价由 `32*32*4=4096B` 变成 `32*33*4=4224B`，多 128B。occupancy（驻留 warp 数占硬件上限的比例）还受寄存器、分配粒度等限制。驻留/Active 表示执行状态和资源已分配、尚未完成，等待数据或会合时通常仍驻留；寄存器保存线程中间值，分配粒度表示硬件按整组而非任意小份资源划拨。因此，不能从增加这 128B 就推断 occupancy 必然下降。

理解检查：lane 是 warp 内的线程位置，double 是占 8B 的双精度浮点数。若所有 lane 都读取 `tile[0][0]`，为什么不是 32 路不同地址冲突？若数据类型换成 double，为什么不能照搬上述每 lane 4B 的推导？先检查指令宽度和 bank word，再谈指标。

## 5. 建立大输入对照，再用 NCU 分两层观察

**目的：** 分清“global 访问改善”和“shared 冲突改善”两种机制。

现在把两条地址推导转成两个不同的观察假设：naive→tiled 应先检查 global 写请求是否更集中，tiled→padded 应先检查 shared 列读取的额外服务工作是否减少。前者即使成立，也可能被同步和冲突抵消；后者即使成立，也可能不是总时间的主要部分。因此不能只排三个耗时的名次，再把最快版本的所有机制都称为已验证。

在入口先单独改 n=2048，建立新的性能组，之后固定规模、输入、设备、编译和计时条件，按步骤 3 执行：

1. variant=0 → 1：加入 shared 与同步、改变线程写出分工。预期 global 写端更连续，但这不是只隔离 bank 的对照。
2. variant=1 → 2：同一模板只改 Pad，检验 shared 布局假设；也要检查资源分配的连带影响。

每版先全量 PASS，再多次普通运行比较中位数（排序后的中间值）与波动。N=65 有边界 warp，且行长未对齐 32 个 float，它适合核对，不适合作为理想合并访问样本。N=2048 也可能在重复访问中命中 L2，不能未经测量称为显存带宽基准。

先为后面的表建立服务单位。**Sector** 是按 32B 边界划分的数据区段，例如 32 个线程连续各读 4B、基址对齐 32B，就覆盖 4 个 Sector；若每人地址差 32B，就覆盖 32 个。L1TEX 是 SM（执行线程块的硬件单元）附近处理这些访问的路径，表里的 global requests/sectors 分别计其工作项和区段访问次数。重复访问可重复累计，不是整个 kernel 唯一地址的数量。

Shared 的 **Wavefront** 是内部一轮能处理的工作包。简化的 32-bank、每线程读一个不同 float 模型中，无冲突一轮，32 个不同字争同 bank 则需 32 轮；但每线程读更多字节时，无冲突也可能天然需要多轮。因此要比较 **Ideal（按指令宽度和参与线程必需的轮次）**、Actual（实际轮次）与 Excessive（额外轮次），而不是把 `Wavefronts/Request>1` 全判成冲突。Wavefront 不是 grid 的 Wave 容量估计，也不是 AMD 的线程执行组，更不等于整条指令的延迟周期数；transaction（事务）若未注明层次，也不能直接当成这些量中的一个。

本次启动全部 block 构成 grid。再回到同一目录，先查 NCU 支持的 Section（指标组）与 metric（具体指标）：

```bash
ncu --list-sections
ncu --query-metrics
```

LaunchStats 描述启动配置，MemoryWorkloadAnalysis 分层描述访存，Occupancy 看驻留，SpeedOfLight 是相对于硬件参考峰值的吞吐概览。吞吐是单位时间完成多少工作；峰值百分比不是“占 kernel 时间多少”。

随后针对 n=2048、variant=0 采一次：

```bash
ncu --section LaunchStats --section SpeedOfLight \
    --section MemoryWorkloadAnalysis --section Occupancy \
    --kernel-name-base demangled -k 'regex:transpose_(naive|tiled)' \
    -s 10 -c 1 -o log/transpose_n2048_v0 ./build/lesson06
ncu --import log/transpose_n2048_v0.ncu-rep --page details
```

反斜杠 `\` 是 Bash 续行符。`--section` 选择指标组；`demangled` 使用可读 C++ 名，`-k` 后的 `regex:` 指正则过滤，括号内 `naive|tiled` 表示两个名字择一，单引号防止 shell 先解释它；匹配包含这些函数名的模板实例。`-s 10 -c 1` 跳过匹配的 10 次预热，只采随后一次，改预热也要改 skip。`-o` 指报告路径，`--import` 读取已有报告，`--page details` 显示详情。variant=1/2 重编、普通核对后分别用 v1/v2 报告名，保留基线。

| 先看什么 | 本课要验证什么 | 异常时如何检查 |
|---|---|---|
| LaunchStats | 函数、grid/block 是否对应配置 | 过滤器、旧二进制、N 是否一致 |
| MemoryWorkloadAnalysis 的 global requests/sectors | 0→1 的写端请求是否更高效 | 读写别混算；sector 减少不等于 DRAM 字节同比减少 |
| shared requests、理想/额外 wavefront、bank conflicts | 1→2 的列读取是否减少额外服务工作 | wavefront 是服务请求的处理批次；超过 1 不自动等于冲突，核对指令宽度与活跃 lane |
| Occupancy 与 shared/寄存器资源 | padding 是否产生驻留代价 | 资源分配有粒度，高 occupancy 也不保证快 |
| SpeedOfLight、普通运行时间 | 局部改善是否变成整体收益 | 消除 32 路冲突不意味着全 kernel 快 32 倍 |

指标名称以本机版本为准，缺失写“未采集”，不填零。NCU 可能 replay（重复执行来采不同指标），其下打印的 Event 时间不作普通基线。若冲突下降而时间不变，结论应是“布局改善已观察，但尚无整体加速证据”；继续检查 global 路径、同步和其他限制，而不是否认数据。

## 6. 总结、验收与记录边界

请独立画出一个跨 tile 元素的完整路径，并解释：① 为什么输入输出两侧都能连续？② 为什么必须同步？③ 为什么多一列不改变转置语义却改变 bank？④ 为什么局部 32 路冲突不能直接换算成整体加速比？完成性能实验时只在同 N 与同条件下计算 `baseline_ms/candidate_ms`，保留“预测—普通时间—分层证据—反例检查”的链条。

**验证边界：** 本次验证仅限静态文本检查与脚本 `bash -n` 检查，未编译、运行或执行实验动作。以下为改前版本的历史记录，不是本次验证：已执行 `CUDA_ARCH=sm_89 bash commands/compile.sh`（nvcc 13.4、C++17、`-O3 -lineinfo -arch=sm_89`），见[编译日志](log/compile.log)，exit 0；随后执行 `bash commands/run.sh`，见[运行日志](log/run.log)，PASS/exit 0。当时只验证所选入口配置，未测试全部参数；profile 当时仅通过 `bash -n`，未执行 NCU、NSYS、workers 或 SASS 检查，无性能结论。[旧 compile-only 日志](log/legacy_commands/compile/source_check.log)仅作历史证据。

理解底层命令后，在本课目录分别执行：

```bash
CUDA_ARCH=sm_89 bash commands/compile.sh
bash commands/run.sh
bash commands/profile.sh
```

三个脚本均无位置参数、按自身位置定位；只有 compile 必填 `CUDA_ARCH`（按设备替换），固定 C++17、`-O3 -lineinfo`，生成 `build/lesson06`，编译前移除旧目标，失败即停止。run/profile 缺产物报错，不隐式编译，改常量/共享头后须显式重编。profile 采 LaunchStats、SpeedOfLight、MemoryWorkloadAnalysis、Occupancy，按函数基名过滤 transpose_naive/transpose_tiled（包括 Pad 模板实例），只执行所选变体，不扫描矩阵；skip 10/count 1，固定 kernel replay/cache all/clock base。

日志为 `log/compile.log`、`log/run.log`、`log/profile.log`，含展开命令、输出与 `exit_code`；同动作重跑覆盖新日志，对照需先另存。报告 `log/profile.ncu-rep` 已存在则拒绝覆盖，NCU 失败即停止，报告内容由实验者检查。旧证据原样迁至 `log/legacy_commands/<原相对路径>`，命令不改；旧 compile-only **不代表改后源码或新脚本**。