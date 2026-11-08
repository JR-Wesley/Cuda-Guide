# Stage 3：Memory Hierarchy

## 当前独立动作约定

以下命令从项目根目录执行。先单独构建要使用的目标，再选择运行、校验或分析；后续动作缺产物只报错，不自动编译。源码或编译配置变化后必须重新执行构建。

`CUDA_ARCH=sm_89 python3 commands/build/run.py transpose`
`CUDA_ARCH=sm_89 python3 commands/build/run.py cache_working_set`
`CUDA_ARCH=sm_89 python3 commands/build/run.py local_array`

日志固定为 commands/<action>/log/，重复同动作覆盖该目录；先阅读或手动保留需要对照的结果再重跑。历史 results/ 不写入。本页历史分析链接不代表新入口已通过 GPU 复验。


## Learning Objectives

主教材：[BPG 13.1 §10.2.2–10.2.4、§10.2.7](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-c-best-practices-guide/index.html)。本阶段包含三个独立程序：

| 实验 | 完成后应具备的能力 |
| --- | --- |
| transpose | 推导全局访问与 shared bank 映射；证明矩形边缘 tile 的同步与初始化正确性 |
| cache_working_set | 固定读取次数和输出规模，仅改变源工作集；区分请求率与 cache hit/DRAM 流量 |
| local_array | 从 PTX/SASS、stack/spill 统计区分寄存器、可寻址 local 数组与寄存器 spill |

编译器寄存器预算的补充依据：[Programming Guide — Maximum Number of Registers per Thread](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-c-programming-guide/index.html#maximum-number-of-registers-per-thread)。该编译器限定符不要求 SM90；项目通过编译探针检测可用性，不假定所有旧 Toolkit 都支持。

## 先看一个 37×65 转置

在根目录运行 `CUDA_ARCH=sm_89 python3 commands/verify/run.py transpose`，再用 `CUDA_ARCH=sm_89 python3 commands/run/run.py transpose --variant naive --rows 37 --cols 65 --warmup 1 --repeats 2 --iterations 3` 与仅将 `--variant naive` 改为 `--variant tiled` 的命令对照。先手算输入 `(r=1,c=2)` 在输出中的下标 `2×37+1`，再核对两份 `samples.csv` 都通过校验。**预期**两个版本产生相同矩阵，但 37×65 小输入不保证 tiled 更快；未看到预期时先查行列与尾部处理，以及本次 `commands/<action>/log/` 的状态和原始日志。之后固定线程布局只将 shared pitch 32 改成 33（`padded` 版本），再读 bank 模型；cache 与 local 是后续两个独立问题，勿一次修改三种因素。

## Theory / Model：Transpose

row-major 输入 R×C，`out[c*R+r]=in[r*C+c]`。naive 每线程处理一个线性输入 i，读取连续而转置写入跨步。tiled 以 32×8 threads 协作搬运 32×32 tile，每线程处理四行；先 coalesced global→shared，block barrier 后从转置方向读 shared，coalesced 写 global。padded 只把 shared pitch 从32改为33，其他布局相同。

完整 warp 的 32-bit word bank 模型为 `bank=word_index mod 32`。当 lane 变化的是 shared 行号时：

| pitch | lane 0/1/2/3 的 word index（固定列0） | bank | full warp 的 distinct banks |
| ---: | --- | --- | ---: |
| 32 | 0 / 32 / 64 / 96 | 0 / 0 / 0 / 0 | 1 |
| 33 | 0 / 33 / 66 / 99 | 0 / 1 / 2 / 3 | 32 |

第一行是不同地址落入同 bank，不是同址广播。完整 warp 的理想冲突度为 `gcd(pitch,32)`，partial warp 需按实际活跃线程重算。padding 预计减少 shared 转置读取的冲突，但真实 shared transaction/conflict 指标仍需 NCU。

每次 requested=`8RC` B，设备输入输出工作集同为 `8RC`；静态 shared 为 `32×32×4=4096 B` 或 `32×33×4=4224 B`。增加128 B可能影响资源分配，必须查询 register/shared 和 occupancy 上界；更小矩阵还可能由 launch/barrier/尾部成本主导。

同步证明：所有 tile 内线程无条件到达 `__syncthreads`。边界 load/store 各自保护；每个有效 store 读取的 `tile[tx][ty+j]` 对应原矩阵有效 `(blockY*32+tx,blockX*32+ty+j)`，因此必有一个线程在 barrier 前写过该元素。禁止在边缘 load 判断失败时提前 return。

## Theory / Model：Cache Working Set

固定 T=`--n` 个输出线程、S=`--steps` 次循环，读取 `input[(i+step*T)&(W-1)]` 并累加，最后写一个输出。W=`--working-set` 必须是2的幂且 W≤T×S，确保整个输入工作集被访问。W≥32 保持 warp 内基本连续的布局；尾部按实际 T 处理。

每次加载数 T×S，requested=`4T(S+1)` B，FP32 add=T×S，唯一输入/输出工作集=`4(W+T)` B。固定 T/S/block 扫 W，计算工作量、输出和循环体不变，工作集独立改变；与 S2 扫 stride 同时改变跨度/sector 利用率的实验互补。

输入是 [-8,8] 的整数值 float，S≤65536，所有部分和绝对值≤524288，因此可逐位验证。CPU reference 执行相同索引。该实验是**请求吞吐/执行时间与工作集的关系**，不是依赖式 pointer-chase latency 测量；也不从耗时推算 cache hit。重复小工作集可能复用缓存，大工作集也可能存在缓存命中或受计算/循环依赖限制。

SASS 检查确认 global load 仍在真实回跳循环内。仅观察到末尾填充用的自跳转不算通过；解析失败要调查。

## Theory / Model：Local Array 与 Spill

每线程有32个独立 float 值，初值 `input[i]+j/8`，每轮执行同样的 `fma(x,1.0001,0.01)`，最后按固定 j 顺序求和。

| 版本 | 唯一处理因素 | 预期证据 |
| --- | --- | --- |
| static_index | 编译期可见索引，unroll 32个位置 | 值可能标量化到寄存器，PTX/SASS 无对应 local 数组访问 |
| dynamic_index | 每轮旋转索引 `(j+r+shift)&31` | 地址可寻址，可能出现 PTX local 数组与栈；并不自动意味着 register spill |
| register_limited | 与 static_index 完全相同计算，但 `__maxnreg__(32)` | PTX register budget、ptxas spill、SASS local spill 访问；需实际确认 |

旋转是0..31的双射，每个值每轮恰好更新一次；独立值的更新顺序不影响其结果，最终求和顺序保持一致。CPU 用 float `std::fma` 验证，`atol=rtol=1e-5`，拒绝非有限值。

模型：核心 update FLOP=`2×N×32×rounds`，不含初始化及最终求和；全局输入输出逻辑字节=`8N`。源级数组访问量为 `N×32×4×(2×rounds+2)`，可能由寄存器消除或由后端 spill 改变，**不是实际 local/DRAM 流量**。CSV 的 `working_set_bytes` 在此仅指 global I/O；private backing 的物理总量未测量，另有每线程资源字段和 `working_set_scope`。

local 是线程私有地址空间，不能由名字理解为片上快速内存。显式 addressable local 与寄存器预算不足的 spill 是不同原因。动态索引还增加地址计算，因此总耗时差不能全部归因为 local load latency。register_limited 若在某目标未产生 spill，要诚实记录；本机实测依据见分析。

## Questions

1. tiled transpose 如何同时让读写两端合并？为什么 naive 不能仅靠连续输入解决？
2. padding 改变了哪一项 bank 地址公式？哪些情况不适用完整 warp 的冲突度预测？
3. 为什么边缘线程仍必须参加 barrier？哪些 shared 元素无需初始化？
4. Cache 实验固定了哪些量？为什么不能只看曲线判断 L1/L2 容量？
5. private array 出现在栈上是否必然因为寄存器不够？怎样比较 PTX 与 ptxas/SASS？
6. 限制寄存器能否保证更快？真实流量、occupancy 上界与 achieved 值各需要什么证据？

## Benchmark Implementation / Parameters

从项目根目录运行，所有实验使用通用 warmup/repeats/iterations、CSV 和证据记录；默认5/21/100。

```bash
CUDA_ARCH=sm_89 python3 commands/run/run.py transpose --rows 1024 --cols 1024
CUDA_ARCH=sm_89 python3 commands/run/run.py transpose --rows 2048 --cols 2048
CUDA_ARCH=sm_89 python3 commands/run/run.py transpose --rows 37 --cols 65

# 固定 T=65536、S=256，仅改变 W；四种输入分别为64KiB/1MiB/16MiB/64MiB。
CUDA_ARCH=sm_89 python3 commands/run/run.py cache_working_set --n 65536 --steps 256 --working-set 16384
CUDA_ARCH=sm_89 python3 commands/run/run.py cache_working_set --n 65536 --steps 256 --working-set 262144
CUDA_ARCH=sm_89 python3 commands/run/run.py cache_working_set --n 65536 --steps 256 --working-set 4194304
CUDA_ARCH=sm_89 python3 commands/run/run.py cache_working_set --n 65536 --steps 256 --working-set 16777216

CUDA_ARCH=sm_89 python3 commands/run/run.py local_array --n 4096 --rounds 32 --shift 7
```

| 程序 | 专用参数与边界 | variant |
| --- | --- | --- |
| transpose | rows/cols 默认1024，范围1..65536；固定 block=256，tile threads=32×8；拒绝 --n，用 rows×cols 派生 n | all / naive / tiled / padded |
| cache_working_set | n默认65536，steps默认256（1..65536），working-set默认1048576元素；每op最多2^28读取以限制 CPU 验证耗时 | scalar / all |
| local_array | n默认4096且≤65536，rounds默认32（1..256），shift默认7（0..31） | all / static_index / dynamic_index / register_limited |

所有显式 GPU 分配默认≤256 MiB，最多占当时 free memory 一半。local 私有存储不属于 cudaMalloc 预算，额外限制 N/rounds；默认分配很小，不以无限增大输入作为测量方法。`__maxnreg__` 由 CMake 编译探针检测；失败时明确输出该 variant 的 Not Tested，其他路径仍可构建，不能把跳过当通过。

## Correctness Verification

```bash
CUDA_ARCH=sm_89 python3 commands/verify/run.py transpose
CUDA_ARCH=sm_89 python3 commands/verify/run.py cache_working_set
CUDA_ARCH=sm_89 python3 commands/verify/run.py local_array
CUDA_ARCH=sm_89 python3 commands/sanitizer/run.py transpose
CUDA_ARCH=sm_89 python3 commands/sanitizer/run.py cache_working_set
CUDA_ARCH=sm_89 python3 commands/sanitizer/run.py local_array
```

Transpose 9种形状覆盖单行/列、31/32/33邻域、37×65、1023×1031 等边缘；Cache 6组 T/W/S 验证全覆盖、partial warp 与一步循环；Local 15组 N/rounds 验证尾部和非整轮数。每个程序还检验非法参数和 CSV 模型。所有采样前后校验，输出先填 NaN；失败不得发布有效 CSV。

Transpose 的 racecheck/synccheck 检查实际共享协作；局部数组无跨线程共享，相关无错误报告不推广到其他同步模式。普通计时与 sanitizer 插桩分开。

## Profiling

```bash
CUDA_ARCH=sm_89 python3 commands/static/run.py transpose
CUDA_ARCH=sm_89 python3 commands/static/run.py cache_working_set
CUDA_ARCH=sm_89 python3 commands/static/run.py local_array
CUDA_ARCH=sm_89 python3 commands/nsys/run.py transpose --variant padded --rows 1024 --cols 1024
CUDA_ARCH=sm_89 python3 commands/ncu/run.py transpose --variant tiled --rows 1024 --cols 1024
```

NCU 先查询版本支持的 sections/metrics：transpose 对应 global sectors、shared transactions/bank conflicts、resources/occupancy；working set 对应 L1/L2 hit 和 DRAM bytes/throughput；local 对应 local load/store、register、stall。不可用则 Not Tested。PTX 的 local、ptxas 的 stack/spill、SASS 的 LDS/STS/LDL/STL 与最终资源分别保存，不混成同一层证据。

Event 包含 kernel 批量平均；host E2E 包含整份输入上传和输出回读，排除分配、初始化、校验。Cache E2E 传输量随 W 改变，不能把其变化当作等量设备工作差异。

## Analysis / Exercises / 验收

实测与适用边界见 [analysis.md](analysis.md)，练习与答案分别在 [exercises.md](exercises.md) 和 [answers.md](answers.md)。验收须有：正确性与边界、同步证据、global/shared/local 地址模型、真实指令/资源、重复计时、模型差异解释、未测项及反例。下一阶段以本阶段 register/spill 观察为起点，系统扫描执行配置。
