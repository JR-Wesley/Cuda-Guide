# 实验 12｜把“看起来更快”写成可以被推翻的性能解释

## 背景：kernel 加速为什么不等于程序加速

**kernel** 是 GPU 上执行的函数，一次 **launch** 是 CPU 提交一次函数调用；CUDA 是编写和运行这些程序的平台。**端到端**指从约定起点到最终结果可用的整个流程，**归因**是用可复查的事实解释为什么快或慢，而非只报一个百分比。下面先用串行小例子说明为什么两者不能混淆。

假设一个流程共 10 ms，其中转置占 2 ms，其余串行步骤占 8 ms。即使转置缩短为 1 ms，流程也只是 9 ms：kernel 加速 2 倍，流程加速却只有 `10/9≈1.11` 倍。这是纸上假设，不是本课测量；**异步**是提交后 CPU 可先继续，不等 GPU 完成；**重叠**是两项工作的实际执行区间有交集，例如 8 ms 的计算和 3 ms 的传输同时开始，都完成才返回，理想总时间是 8 ms 而非 11 ms。因此存在重叠时，连这种直接相加也不一定成立。

本课用转置 B 练完整归因，再用规约 C 与乘加 D 迁移方法。目标不是比较三种算法谁快，而是每次写清“相同任务、单个变量、计时范围、预测机制、支持与反证”。配合 [lec12](lec12.md) 第 1–3 节建立实验契约，第 4–5 节学习从指标回到地址与机器指令。

本实验可独立读；相关线程合作和地址机制会在用到时解释。[lab06](../06_shared_memory_and_transpose/lab06.md) 可作为额外练习，不是理解下面步骤的替代说明。

本课的三个程序分别把注意力放在不同的数据流上：转置保留数值、只换位置；规约把许多输入合为较少的块内和；FMA 在每个线程内部反复更新自己的值。它们不是一条自动串联的应用流水线，而是三个独立的归因练习。选择它们，是因为可以分别从地址排列、线程间合作、线程内依赖提出具体假设，再检查假设是否真的解释时间变化。读者不必先记住前课实现，下面会在每个例子动手前重新建立输入到输出的全过程。

host 是 CPU 上准备输入、调用接口和核对结果的部分，runner 是组织这些步骤的函数，不是另一种 GPU 硬件。CLI 指运行时的命令行参数；这里不用它来选择输入。读 [main](kernels/main.cu) 顶部常量：`example=0/1/2` 分别选择 B/C/D，一次只运行一个例子，不接受 CLI。算法分别在 [matrix_transpose.cuh](../kernels/matrix_transpose.cuh)、[reduction.cuh](../kernels/reduction.cuh)、[fma.cuh](../kernels/fma.cuh)；分配、核对和释放也在这些 runner 中。共享 [common.cuh](../kernels/common.cuh) 提供错误检查与批量计时。以下命令均为未来操作，在 Linux/Bash 的 `cuda/ncu/12_end_to_end_performance_attribution/` 执行。

## 1. 主例 B：先用小矩阵预测结果和尾部

**行优先/行主序**指一行元素连续存储，下一行紧随其后；第 row 行、第 col 列的线性下标是 `row*N+col`，行列从 0 编号。float 是每元素 4 B（字节）的单精度浮点数。转置 kernel 的输入是一块按行连续存放的 N×N float 矩阵，输出是另一块同大小矩阵。它不计算乘积，也不改变元素值：输入第 row 行、第 col 列的数，要出现在输出第 col 行、第 row 列。例如输入 `[[1,2],[3,4]]`，输出就是 `[[1,3],[2,4]]`，线性数组从 `[1,2,3,4]` 变为 `[1,3,2,4]`。这组整数只用于说明位置，实际 runner 的初始化值另见下文。

GPU 线程按 **block/CTA（线程块）**分组，**grid** 是一次启动的全部块；同块可使用一份 **shared memory（共享内存）**暂存合作数据。**barrier（屏障）**`__syncthreads()` 让块内参与线程到齐，并提供相应的块内内存可见性，保证读取其他线程的写入有正确顺序；不是所有 GPU 都停下等待。tx/ty 是块内线程的横/纵坐标。

naive（直接实现）版让一个线程读到某个元素后直接写到转置位置。tiled 版则让一个线程块先把 32×32 区域装进块内 shared memory，等所有参与线程到齐，再重新分工写出；例如块内 `[0][1]` 的值由 `(tx=1,ty=0)` 装入，随后由 `(tx=0,ty=1)` 取走写出。padding 版保持这套分工，只让 shared 每行多一个不输出的存储槽。三版必须产生同一个矩阵，但访问地址排列和合作开销不同，所以很适合检验“少一些低效访存，是否足以抵偿新增合作成本”。

variant 是实现变体，NaN 是“非数”浮点值；用 NaN 作**哨兵**是先填一个正常输出不该具有的标记，以便发现漏写。host 生成输入、分配两块 GPU 存储、上传矩阵，并将输出预填为 NaN 哨兵；随后按所选 variant 启动对应 kernel，下载输出，按原输入的坐标逐项检查转置位置。这里 CPU 不做 GPU 输出之后的第二段矩阵运算，所以当前计时只覆盖搬动矩阵的 kernel，而不是一个完整应用。先搞清这个语义契约，才有资格把三个实现放进同一张性能表。

**目的与代码：** 读 `transpose_naive`，先纸算 N=3 的抽象位置：

```text
输入位置       输出位置
00 01 02       00 10 20
10 11 12  →    01 11 21
20 21 22       02 12 22
```

图中数字是坐标标签，不是实际初始化值。`%` 是取余，`0.125f` 的 f 表示 float 常量。真实输入为 `(linear_index%251-125)*0.125f`，输出必须满足 `out[col*N+row]=in[row*N+col]`。只做位置变换没有加法，因此逐元素精确比较。

默认保留 `transpose_n=65`：N=65 比 N=3 多覆盖 tile 边界。tile 是一次协作处理的 `32×32` 小块，block 固定 `(32,8)`，每线程用 j=0/8/16/24 处理四行，grid 是 `(3,3)`。例如输入 `(row=0,col=64)` 在线性索引 64，输出位置 `(64,0)` 在线性索引 4160；其值应为 `(64-125)*0.125=-7.625`。

**预期与解释：** 65 包含两个完整 tile 和一个尾部，既检查跨 tile 坐标交换，也检查不足 32 的访问。输出预填 NaN（非数）哨兵，漏写不会误判成正常值。所有 N² 项都检查，不只查这个例子。

**理解检查与异常排查：** 预测 `(64,64)` 的输出位置；若只最后一行/列出错，先检查 global 边界和 block 坐标交换，不先解释成性能问题。

## 2. 显式编译与普通运行三版正确性

**目的：** 任何性能结论先有同一语义的正确结果。未来执行前分别查询：

```bash
nvidia-smi
nvcc --version
ncu --version
ncu --list-sections
```

`nvidia-smi` 查看 GPU/驱动（驱动负责系统和设备交互），`nvcc --version` 查看 CUDA 编译器版本；Toolkit 是包含编译器、头文件等的开发工具集合。**NCU（Nsight Compute）**是 kernel 性能分析器，`ncu --version` 查版本，`--list-sections` 列 section（按问题组织的指标组）。确认设备/驱动、Toolkit 编译器、NCU 版本与指标组。compute capability 是设备指令能力编号，需按实际 GPU 确认；以下 `sm_89` 仅为示例，执行前替换为设备及 Toolkit 支持的架构。

```bash
mkdir -p build log
nvcc -std=c++17 -O3 -lineinfo -arch=sm_89 kernels/main.cu -o build/lesson12
```

`mkdir -p` 创建缺失目录；`-std=c++17` 选择语言标准，`-O3` 优化，`-lineinfo` 供源码关联，`-arch` 指目标 GPU，`-o` 指输出。编译后单独运行：

```bash
./build/lesson12
```

**看什么：** 默认 example=0、variant=0 应打印 `transpose variant=0(naive) N=65 block=(32,8)` 和 `check=PASS errors=0`，这是预期验收而非本次结果。只把 `transpose_variant` 改为 1，显式重编、运行；再只改为 2，重编、运行。三版必须核对同一输入的全部 4225 项。

**异常检查：** 先查首个 mismatch、打印配置和是否重编；编译错误不拿旧二进制继续跑。shared 实现不能让边界线程在 barrier 前提前返回：即使该线程没有合法 global 元素，也要参与块同步。

**预热**是先运行但不记入目标计时，以排除首次准备并形成可说明的缓存状态。**stream（流）**是 GPU 工作的有序提交队列；**CUDA Event** 是放在其中的完成点，计时事件还能记时间戳。例如 start→20 次目标工作→stop，等待 stop 后读间隔再除以 20，才是本批的每次平均。它不自动覆盖另一流或另一设备的所有工作。

共享 `time_kernel` 默认预热 10 次，Event 包住 20 次 kernel 启动，打印每次平均毫秒；分配、上传、下载、CPU 核对在计时外，批量区间可能有提交空隙。small 配置主要验正确性，不拿单个小 grid 的时间宣称带宽饱和。

## 3. 推导两次优化分别改变了什么

三版在 N=65 都正确，只排除了明显的搬错、漏写和尾部错误，并未说明为什么某版应该更快。**warp** 是通常由 32 个 GPU 线程组成的执行组，**lane** 是其中的位置编号 0–31；本例 x 方向有 32 个线程，所以同一 warp 的 tx 连续。推导要从同一个 warp 的地址开始：若相邻线程读写相邻元素，服务请求更容易合并；若相隔整行，即使逻辑字节数相同，也可能需要更多请求。tiled 引入 shared 是为了重排这种线程—元素对应关系，padding 则进一步重排 shared 地址。因此下面分开研究 0→1 与 1→2，不能把两步统称为一个“共享内存优化”。

**目的与代码：** 对照三个 kernel，先纸上预测，暂不采集。

naive 版中同 warp 的 tx 从 0 到 31，读取 `in[y*N+x]` 连续，写入 `out[x*N+y]` 却相隔 N 个 float。合并访存是把相邻线程的访问组合成较少请求；这里连续读不代表连续写。

tiled 版先把连续 global 读取写到 `tile[ty+j][tx]`，全块同步，再读 `tile[tx][ty+j]`，同时交换 block 的 x/y 坐标，使输出端相邻线程连续写。它不是免费变换：新增 shared 访问和 barrier。因此 0→1 的预测是 global 请求利用改善，但仍需指标与时间验证。

**bank** 是共享内存可以并行服务地址的分组，不同线程读同一 bank 的不同地址可能需分批服务；多个线程读同一地址则可能广播，不是这种冲突。这里 word 是 4 B 的字，字索引就是字节偏移除以 4。对 32-bit float，在常见 32-bank 模型下，共享地址的 bank 为字索引模 32。列访问 `tile[tx][c]`：

```text
Pad=0：bank=(tx*32+c)%32=c          不同地址落在同一 bank
Pad=1：bank=(tx*33+c)%32=(tx+c)%32  相邻线程落到不同 bank
```

前者是不同地址冲突，不是同地址广播。Pad 从 0 到 1 将共享数组从 4096 B 增至 4224 B，预期缓解列访问 bank conflict（同组请求需分批服务）。真实指令宽度、访问拆分和分配粒度仍须核对，不能把这个纸模型当测得 wavefront 数。

**Occupancy** 是实际驻留的 warp 数占硬件最大可容纳数的比例，驻留表示分配了执行资源，不代表每周期都能发射指令。shared 分配也有硬件粒度，增加 128 B 不必然改变可驻留块数。

**理解检查：** 为什么 tiled 必须同时交换 tile 内坐标与 block 坐标？为什么 padding 可能减少冲突却不改变 occupancy（驻留 warp 比例），甚至不明显加速？

## 4. 大输入单变量对照：先基线，后 NCU

**目的：** 正确性通过后，先单独把 N 改为 2048，variant 恢复 0，重做步骤 2，建立新性能配置。此后固定 N、block、输入、GPU、编译参数、预热/重复与采集设置，只比较 0→1，再比较 1→2。不要拿 N=65 和 N=2048 算优化 speedup。

这里 N 始终为矩阵边长（不同于讲义曾用 N 表示总元素数）；每项 float 4 B，一读一写为 8 B。每版逻辑字节数 `Q=2*N*N*4=8*N²`，N=2048 时为 32 MiB，表示读写需求而非 DRAM 实际流量。1 MiB=2²⁰ B，GB/s 按 10⁹ B/s；若用毫秒计时，先除以 1000 得秒再算 `Q/t`。**中位数**是排序后的中间值（偶数个常取中间两项平均），离散程度描述波动，例如最小/最大值或分位区间。普通运行多次，保存每次批量均值、样本数、中位数和离散程度；不只保留最好的一次。加速比是基线时间/优化时间，差异若和波动相当就不能下强结论。

确认支持所需 Section 后，从本课目录采集；下例用于 naive，切换变体并重编后把报告名分别改为 `b-tiled`、`b-padded`：

```bash
mkdir -p log
ncu --section LaunchStats --section SpeedOfLight \
  --section MemoryWorkloadAnalysis --section Occupancy \
  --kernel-name-base demangled -k 'regex:transpose_(naive|tiled)' \
  -s 10 -c 1 -o log/b-naive ./build/lesson12
```

`--section` 指指标组；`demangled` 表示按可读函数名匹配，`-k` 正则包含模板基名 `transpose_tiled`，其中 `(naive|tiled)` 表示匹配 naive 或 tiled，函数模板是由同一份代码生成不同参数版本的机制；`-s 10` 跳过匹配的预热，`-c 1` 仅采一个正式启动；`-o` 指报告前缀。已有同名报告就改名，不强制覆盖。若共享计时的 warmup 被改，skip 也需重新核对。

先认清表中量：**sector** 是相关内存路径的服务单位，常见计数为 32 B；Request 是该路径请求，`Sectors/Request` 是服务 sector 数除以请求数，不是除以线程数。**wavefront** 是服务共享请求的批次，额外批次可能来自冲突，也可能是宽访问正常拆分。LaunchStats 看启动形状与资源，SpeedOfLight 看相对峰值吞吐，MemoryWorkloadAnalysis 看内存活动，Occupancy 看驻留资源限制。

**按这个顺序看与解释：**

| 对照 | 先核对的事实 | 机制证据 | 能推翻简单解释的结果 |
| --- | --- | --- | --- |
| 所有版本 | 函数实例、N、grid/block、编译配置 | LaunchStats、SpeedOfLight 概览 | 采到错误规模/预热则不能比较 |
| 0→1 | 输出一致、逻辑字节相同 | global 请求的 sectors、读写效率，shared/barrier 代价 | sectors 改善但时间无收益，不能说 global 是唯一瓶颈 |
| 1→2 | 仅 Pad 变化 | shared 冲突/额外 wavefront 与资源占用 | 冲突下降但时间不变，可能已转移到别的限制 |

具体指标定义、单位和作用域按本机报告核对，不把所有路径的 sector/wavefront 当同一口径。缺指标先查 Section/设备支持，不能补零。NCU replay（为采指标重复执行）可能改变缓存与调度，其 Duration 与普通 Event 均值分开记录。**SASS** 是 GPU 最终机器指令的表示，一行源码可能变成多条 SASS；需要机器码证据时另做静态动作，不用源码推测替代 SASS。

## 5. 迁移 C：同步变少是否真让 partial 更快

转置的输出项数与输入相同，接下来的规约却要合并数值：例如输入 `[-0.375,-0.25,-0.125]`，三个数相加得到 `-0.75`。`ceil` 表示向上取整，例如 `ceil(257/256)=2`（此处按实数除法理解）。本例 kernel 接收一维 float 输入、长度 n 和 partial 输出数组，每块 256 线程各读一项，越界线程取零，合作求出这一块的和，最后只有线程 0 写一个 partial sum（局部和）。因此输出长度是 `ceil(N/256)`，不是 N，也不一定只有一个结果。

host 生成输入并上传，给 partial 分配存储并预填 NaN，再启动向上取整数量的线程块。GPU 完成后，host 下载每个 partial，用 CPU double 计算相同块范围的参考和逐块检查，再汇总并核对总和。这个流程意味着 GPU 的任务到“生成局部和”为止，完整求和还需要下载与 CPU 汇总。它适合检验线程间合作方式的代价：两版读取相同输入、生成相同数量的 partial，主要改变中间值如何交换及何时等待，而不是减少输入数据量。这里不再运行转置，不能拿 B 的时间或报告解释 C。

double 是这里 CPU 参考计算用的双精度浮点格式，通常每值 8 B，比 float 精度高，但也不是任意求和都精确。输入 `(i%7-3)*0.125f` 的每 7 项和为 0；N=65 有 9 个完整周期和剩余 `-0.375,-0.25`，所以预期一个 partial、总和 `-0.625`。N=257 时应有两个 partial，分别 `-0.75`、`0.125`，总和仍为 `-0.625`。这些是手算预测，程序会核对每块结果及总和，避免错误 partial 相互抵消。

**shuffle** 是让同一 warp 的线程用指令交换寄存器值；寄存器是线程保存临时值的快速存储。**mask（掩码）**以每一位表示某个 lane 是否参加，32 位全为 1 的 `0xffffffffu` 表示全部参加。lane 0 就是 warp 的第一个线程位置，不是整个 grid 的线程 0。

shared 版把 256 个值写入 shared，offset 从 128 减半到 1，每层加法后全块同步。shuffle 版先由每个 warp 用寄存器交换指令合成局部和，8 个 lane 0 写共享数组；全块同步后 warp 0 的 32 个 lane 再规约 8 个值和 24 个零。full mask `0xffffffffu` 指 warp 全部 32 个 lane 参加；越界线程不可提前退出，第二级也不能只让前 8 个 lane 执行 full-mask shuffle。

shared 树的每一层都要把中间值放在块内可见的位置并等待，shuffle 则先将合作限制在 warp 内，只在合并八个 warp 结果时跨 warp 交换。由此可以预测 shared 访问与全块 barrier 减少，但新增的寄存器交换也有成本；读输入、写 partial 和启动成本没有消失。只有这些合作开销确实限制当前配置时，减少它们才更可能降低时间，这正是后面需要检验而非预填的结论。

**独立步骤与命令：** 先把 main 的 example 改为 1，设 `reduction_n=65`、`use_shuffle=false`，按步骤 2 显式编译、运行建立 shared 正确性基线。小输入再只改 use_shuffle=true，重编运行，确认 `check=PASS block_errors=0`；再单独用 N=257 验边界。CPU 用 double 参考值，每块及总和容差是 `1e-6*(1+abs(reference))`，NaN 也失败。当前小的二进制小数理论上可精确求和，不把这个误差特性推广到任意浮点输入。

`1<<20` 是整数左移，等于 2²⁰=1,048,576。SchedulerStats 看调度器如何挑选就绪 warp，WarpStateStats 看 warp 等待状态，InstructionStats 看执行指令统计；**stall** 是暂时不能前进，其样本比例的分母是对应样本数，不能直接当作微秒占比。性能阶段先单独设 N=`1<<20`，再固定 N 只切 shared/shuffle。复用步骤 2 的编译和普通运行，之后用 C 专属采集命令：

```bash
ncu --section LaunchStats --section SpeedOfLight \
  --section SchedulerStats --section WarpStateStats --section InstructionStats \
  --kernel-name-base demangled -k 'regex:reduction_(shared|shuffle)' \
  -s 10 -c 1 -o log/c-shared ./build/lesson12
```

shuffle 版另名 `c-shuffle`。**看什么：** partial 数是否相同，shared/barrier 指令与等待是否变化，普通运行时间是否超过波动地降低。指令更少或 stall 比例更低不自动等于更快，比例可能只因分母改变。失败先查参与线程、mask、补零和是否重编。

`partial_gpu_ms` 只测生成 partial 的 GPU 阶段，不含 CPU 汇总/拷贝/分配。逻辑字节是 `4N+4*ceil(N/256)`，不能套转置的 `8N²`。理解检查：为什么只核对总和不够？为何“GPU partial 加速”不等于“完整 reduction 加速”？

## 6. 迁移 D：固定 FMA 工作量再谈并行链

规约依靠线程交换不同输入，FMA 例子则没有线程间交换。FMA（融合乘加）把 `a*m+b` 作为一次融合操作，只在最终结果处舍入；逻辑运算量计一次乘法加一次加法，即 2 FLOPs（浮点运算次数）；FLOP/s 才是每秒运算速率，不把次数与速率混为一谈。kernel 接收输出数组、n、每线程 FMA 总次数、乘数和加数，不接收输入数组。线程 i 自己生成各条链的初值 `0.5+i/65536+(j+1)/256`，反复执行 `fmaf(acc[j],1.0001f,0.0001f)`，最后按链编号顺序相加，写一个 out[i]。

例如线程 0 的单链从 0.50390625 开始，第一轮约为 0.50405664，第二轮必须使用第一轮的新值。四链时初值分别是 0.50390625、0.5078125、0.51171875、0.515625；每条链的下一轮仍等待自己的上一轮，却不需要等待其他链。纸例 total=8 时，一链做 8 轮，四链各做 2 轮再求和。这不是把四个线程合成一个，而是在同一线程里提供四组可独立更新的值。

host 只分配 GPU 输出，不上传数组；按向上取整的 grid 启动后下载 N 项，并在 CPU 以相同初值、每链轮数和最后求和顺序重算，检查有限性及误差。不同链数输出不必相等，因为它们的初值与末尾求和都不同；公平性来自固定逻辑 FMA 总量并分别核对，不是强行要求同一输出。这个程序末尾才写一次结果，因而适合把观察焦点从大量数据交换转到线程内依赖。

更多独立链给编译器提供穿插运算的机会，但也要同时保存更多中间值；循环轮数还会变少。所以本例的假设有两面：依赖等待可能减轻，资源与控制开销也可能变化。需要机器指令和资源证据来区分，不能看到加速就直接归因于并行链。详细依赖与 SASS 实践见 [lab09](../09_ptx_sass_and_source/lab09.md)。

现在再把 main 的 example 设为 2，默认 `fma_n=65`、`fma_block=256`、`fma_chains=1`、total/thread=256。读共享 `fma_chain` 和 `run_fma`，按步骤 2 编译运行，先通过全部 N 项 CPU 递推核对。

先单独建立 N=`1<<20` 的性能配置，再固定 N、block、total=256，只改 chains=1→4。每线程 FMA 数不变，逻辑 FLOPs=`2*N*256`，不计末尾 `Chains-1` 次加法；rounds、初始化和循环控制仍变化，不声称机器工作完全一致。总次数必须整除链数。

```bash
ncu --section LaunchStats --section ComputeWorkloadAnalysis \
  --section SchedulerStats --section Occupancy \
  --kernel-name-base function -k 'regex:fma_chain' \
  -s 10 -c 1 -o log/d-chains1 ./build/lesson12
```

本命令的 `--kernel-name-base function` 只按去掉参数和模板参数的函数基名匹配；不是把前面的 demangled 写法无条件用于所有场景。ComputeWorkloadAnalysis 关注计算指令/管线（执行某类指令的硬件通道）活动。调度器（scheduler）从已就绪的 warp 中挑选下一条指令发射（issue）；**可发射**表示依赖结果已准备好且满足其他执行条件，不是所有驻留 warp 都随时可发射。**spill（溢出）**指临时值未能全部保存在寄存器而转存线程私有的 local memory，后者通常由设备内存承载，不是 shared。

**看什么与反证：** 四链报告另名，核对指令独立性、寄存器与驻留资源、可发射 warp 和普通时间。更多链可能改善依赖等待，也可能增加资源压力；未确认实际 SASS，不能仅凭数组名断言寄存器数或 spill。异常先检查总工作是否真的固定、整除和重编，再检查编译器变换。不能让四链各做原本全部轮数后仍称“同量任务优化”。

## 7. 回到端到端：写报告，而不是把局部指标相加

**关键路径**是依赖关系中决定最终完成时刻的路径，例如开头 8 ms 和 3 ms 的并行例子，必须看哪条路径拖住最终结果。**Amdahl 定律**用于无重叠、任务分解稳定的情形：若原总时间中目标占 p，它加速 s 倍，整体倍数为 `1/((1-p)+p/s)`；p 为 0–1 时间占比、s 为正的无单位倍数。例如开头 p=0.2、s=2，得到 `1/0.9≈1.11`。不能把重叠 kernel 时长相加后当 p。

三组实验即使得到显著的 kernel 加速，能够直接支持的仍只是各自的计时窗口：转置的搬运、规约的 partial 生成、FMA 的输出生成。尤其规约还有 CPU 汇总，局部加速可能被窗口之外的成本淹没；反过来，如果应用存在重叠，缩短一个非关键阶段也可能完全不改变完成时刻。因此最后不是把不同报告的 Duration 相加，而是先问“使用者要等待哪个最终结果”，再判断当前证据覆盖了这条完成路径的哪一段。

**目的：** 把以上任一组对照写成可复查解释。无需新增命令，先明确现有计时能力：

```text
初始化 → 分配 → 上传 → [GPU 预热、批量目标启动] → 下载 → CPU 核对/汇总 → 释放
                         当前只提供其中目标批次的 Event 平均
```

CUDA **context（上下文）**是进程在设备上的 CUDA 状态与资源容器，首次建立也可能耗时。CPU 墙钟是现实中经过的起止时间；**NSYS（Nsight Systems）**用于记录 CPU/API/GPU 时间线，以检查等待和重叠；**NVTX** 可给某阶段起名字，但不是同步或计时替代物。

真正端到端测量需要另行规定：是否包含 CUDA context 初始化、是否重复分配、起止边界在哪、结束前如何等待 GPU。CPU 墙钟适合完整流程，NSYS 适合查看时间线与重叠，NCU 适合解释特定 kernel；当前 runner 不能由 kernel_ms 自动补出总耗时。本次不为此添加计时代码或脚本。

提交一张自己的证据表，未采字段写“未采集”，不用讲义假设数字填空：

| 项目 | 必须回答 |
| --- | --- |
| 配置与正确性 | 同一输入/结果粒度？每版如何核对？ |
| 模型 | 逻辑字节/FLOPs 怎么推导？哪些不是实际流量？ |
| 普通计时 | 时间范围、批量次数、样本数、中位数与波动？ |
| 因果证据 | 哪条指标/指令支持机制？单位和作用域是什么？ |
| 反证与复测 | 哪种观测会推翻假设？是否只改一个变量？ |
| 未测边界 | 是否测过端到端、真实 DRAM/链路流量、所有边界？ |

理解检查：padding 冲突下降但总时间不变，能宣布优化成功吗？规约 kernel 快一倍但 CPU 汇总占主导，下一步该看什么？能回答这些问题比生成一份 full 报告更重要。

## 8. 已有入口、compile-only 与旧记录边界

理解底层命令后，在本课目录分别执行：

```bash
CUDA_ARCH=sm_89 bash commands/compile.sh
bash commands/run.sh
bash commands/profile.sh
```

三个脚本均无位置参数、按自身位置定位；仅 compile 必填 `CUDA_ARCH`（按设备替换），固定 C++17、`-O3 -lineinfo`，生成 `build/lesson12`，编译前移除旧目标，失败即停止。run/profile 缺产物报错，不隐式构建，改常量/共享头后须重编。profile 默认采转置主实验所需 LaunchStats、SpeedOfLight、MemoryWorkloadAnalysis、Occupancy，函数基名过滤器也包含本课归约/FMA 候选，仅执行已编译分支，不遍历矩阵；skip 10/count 1，固定 kernel replay/cache all/clock base。C/D 专项调度或计算指标仍使用第 5/6 节明确命令另行采集，不把默认报告冒充全部专项证据，也不增加端到端计时功能。

新日志为 `log/compile.log`、`log/run.log`、`log/profile.log`，含展开命令、输出与 `exit_code`；重跑覆盖各自新日志，对照需先另存。报告 `log/profile.ncu-rep` 已存在则拒绝覆盖，NCU 失败即停止，报告内容由实验者检查；静态检查单独保存到本课 `log/`。旧证据原样迁至 `log/legacy_commands/<原相对路径>`，原始命令不改；旧 compile-only **不代表改后源码或新脚本**。

**当前验证：** 已执行 `CUDA_ARCH=sm_89 bash commands/compile.sh`（nvcc 13.4、C++17、`-O3 -lineinfo -arch=sm_89`），见[编译日志](log/compile.log)，exit 0；随后执行 `bash commands/run.sh`，见[运行日志](log/run.log)，PASS/exit 0。只验证本轮入口配置，未测试全部参数；profile 仅通过 `bash -n`，未执行 NCU、NSYS、workers 或 SASS 检查，无性能结论。[旧 compile-only 日志](log/legacy_commands/compile/source_check.log)仅作历史证据。