# Stage 7：传统CUDA Core FP32 GEMM

## 当前独立动作约定

以下命令从项目根目录执行。先单独构建要使用的目标，再选择运行、校验或分析；后续动作缺产物只报错，不自动编译。源码或编译配置变化后必须重新执行构建。

`CUDA_ARCH=sm_89 python3 commands/build/run.py gemm`

日志固定为 commands/<action>/log/，重复同动作覆盖该目录；先阅读或手动保留需要对照的结果再重跑。历史 results/ 不写入。本页历史分析链接不代表新入口已通过 GPU 复验。


## 学习目标与教材

本阶段把地址映射、复用、资源、指令与测量串起来，完成Naive→Coalesced→Shared Tiled→Register Tiled→Unrolled的完整实验链。五版已在原生SM89路径完成正确性、静态、Sanitizer和普通计时验收；[分析记录](analysis.md)保留小矩阵反例、A/B/B/A漂移、NSYS补测与尚未测量的计数器，不预设优化必然加速。

主教材：[BPG 13.1矩阵乘共享内存章节](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-c-best-practices-guide/index.html#shared-memory-in-matrix-multiplication-c-ab)、[数值精度章节](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-c-best-practices-guide/index.html#numerical-accuracy-and-precision)，并回顾Ch.11资源与Ch.12–13指令/控制流。本项目的Coalesced指线程映射对照，命名不等同于教材示例中使用shared的coalescedMultiply；所有公式以这里的源码为准。

统一问题为row-major FP32矩阵`A[M,K] * B[K,N] = C[M,N]`，alpha=1、beta=0，无C输入、不转置。输入、输出、CPU参考和校验规则在五版之间固定。传统SIMT线程执行FP32 FMA，不使用Tensor Core、cuBLAS、CUTLASS或SM90+特性。

## 先核对一个小矩形输入

在项目根目录执行 `CUDA_ARCH=sm_89 python3 commands/verify/run.py gemm`，然后运行 `CUDA_ARCH=sm_89 python3 commands/run/run.py gemm --rows 37 --cols 65 --depth 33 --variant naive --warmup 1 --repeats 2 --iterations 2`；仅将 `--variant naive` 改成 `--variant coalesced` 再运行。先画 `A[37,33]`、`B[33,65]` 到 `C[37,65]` 的下标，预测输出大小，并核对两次 CSV 的 `validation` 和误差列。**预期**都通过容差，但不能承诺小矩阵速度排序；失败先核对维度、显存约束及本次 `commands/<action>/log/` 的状态与命令。先理解线程到输出位置的变化，再读后面的共享内存/寄存器/展开版本，每次只增加一层机制。

## 从lane地址到tile复用

### Naive与Coalesced

线程块T×T。Naive：`row=block_x*T+tx, col=block_y*T+ty`；Coalesced：`row=block_y*T+ty, col=block_x*T+tx`。每线程顺序遍历k、累加一个C元素，两版总输出、kernel数和线程块数量相同，二维grid轴交换。区别是warp内哪个输出坐标随lane递增。

某次k的字节地址为`A:4*(row*K+k)`、`B:4*(k*N+col)`、`C:4*(row*N+col)`。T32完整首warp、M/N/K足够且行对齐时，Naive A/C跨32行，B同址广播；Coalesced A同址广播，B/C连续32 float。T16一个warp含两个16线程行，必须按实际(tx,ty)展开，不能套T32模型。

以M=N=K=256、首tile、k=0、32 B sector、每lane标量4 B为例：

| T | 映射 | A / B / C首warp覆盖sector数 |
| --- | --- | --- |
| 16 | Naive | 16 / 1 / 16 |
| 16 | Coalesced | 2 / 2 / 4 |
| 32 | Naive | 32 / 1 / 32 |
| 32 | Coalesced | 1 / 4 / 4 |

重复地址只算集合中的一个sector。表是单次标量请求的地址模型，不是动态cache miss、DRAM transaction或整个kernel流量；后续k可复用cache line，尾部mask和非整对齐行会改变集合。程序按实际尺寸计算首warp模型。

### Shared Tiled

每块负责BM=BN=T输出，沿K分为BK=T段，协作加载A[T,T]与B[T,T]到shared，再计算T轮。每段开始对越界输入写0；写完shared后全块barrier，计算后再barrier防止下一段覆盖其他线程仍在读的tile。越界输出线程仍必须参与加载、置0与同步，最后只屏蔽store。

使用row-major shared布局。T32时计算阶段As[ty,k]对warp同址广播，Bs[k,tx]连续32bank；T16时warp的两个半warp读A的两个不同bank地址，B的16个地址在两半warp重复广播。该布局无需照搬transpose的padding；shared地址模型与真实指令宽度仍需反汇编核对。

### Register Tiled与Unrolled

保持T×T线程、BN=BK=T，每线程负责R个竖直输出；BM=R*T，`row_r=block_y*BM+ty+r*T`、`col=block_x*T+tx`。每个k先读一个B值，再更新R个acc。合作加载A[RT,T]、B[T,T]，R个A行和R个acc用静态索引，R循环展开，避免把acc变成动态寻址local数组。

相对Shared Tiled，寄存器版扩大输出tile、A共享tile与每线程工作；减少M方向grid数和B重复加载，暴露独立累加链，也可能增加寄存器压力、减少驻留量和小矩阵并行度。这些是同一策略的派生变化，应一起报告。Unrolled只改变同一register tile内的BK循环展开，R和布局保持不变；检查SASS后再判断是否真的减少loop控制和增加代码体积。

## 可计算性能模型

有用FLOP=`2MNK`，包括每个fmaf的乘法和加法，首项加到0也计入。唯一逻辑数据量`B_min=4(MK+KN+MN)`；不读C。它不是保证实现的实际流量下界计数器值。

| 实现 | 源级global load元素数 | 逻辑global requested B |
| --- | --- | --- |
| Naive/Coalesced | 2MNK | 4(2MNK+MN) |
| 分块BM×BN、BK=T | MK×ceil(N/BN)+KN×ceil(M/BM) | 4×(load元素数+MN) |

尾部global加载只计有效元素，shared的越界置0没有global load。shared计算仍包含padding：源级FMA模型`ceil(M/BM)*ceil(N/BN)*BM*BN*ceil(K/T)*T`，大于或等于有用MNK。这个模型不等同于最终动态指令次数，编译器、predicate及warp活跃lane还会影响它。

`AI_requested=2MNK/B_requested`；`AI_min=2MNK/B_min`是理想复用参照。以完整tile大矩阵、忽略C store为近似：direct AI≈1/4 FLOP/B；Shared Tiled≈T/4；Register Tiled≈R*T/[2(R+1)]。公式来自每个k-tile做2R*T³ FLOP，搬(R+1)T²个float。T16/R4时约6.4 FLOP/B，并非因为R4就比R1的4 FLOP/B提高4倍。

条件化吞吐下界可写为`T≥max(2MNK/P_FP32, B_DRAM/BW_DRAM)`；实际DRAM字节、持续FP32吞吐和持续带宽需要测量/明确的硬件上界，不能把源级requested直接代入当成实测DRAM，也不能用未知峰值填数。

实际性能主比较同一M/N/K的耗时与有用GFLOP/s=`2MNK/(time_us*1000)`。CSV的requested GB/s随实现的逻辑请求模型变化，不能把它跨版直接当成DRAM吞吐或单独作为优化排名。缓存可使请求多的版本仍有很少DRAM流量，NCU缺失时不反演其数值。

shared资源：Shared为2T²×4 B；Register/Unrolled为(R+1)T²×4 B。实际register/local和理论blocks/SM由Runtime API记录，ptxas/SASS复核；粗上界受threads、register allocation、shared和架构block数共同限制。T32有1024线程，不必然优于T16的256线程；API理论occupancy不等于achieved occupancy。

## 正确性与数值误差

CPU对所有MN输出计算FP64参考，同时累加绝对积和`S_ij=Σ|A_ik*B_kj|`，不抽样。输入先提升double再相乘，两个FP32正常有限数的乘积可由FP64准确表示，参考求和仍有FP64舍入。输入范围有界，无NaN/Inf/非正规极小值；GPU显式fmaf、默认精度、没有fast math。

对K次FP32 FMA递推，标准逐步舍入模型给出`|error|≤gamma_K*S_ij`，其中`u=2^-24, gamma_K=Ku/(1−Ku)`。本实验使用`2*gamma_K*S_ij+1e-7`作保守阈值，余量覆盖FP64参考误差和很小数值的绝对误差；K≤4096确保Ku<1。每版报告最大abs error、误差/阈值，以及以`max(|reference|,1e-12)`为分母的最大相对误差。后者只是诊断，不能在接近0的取消求和中单独决定PASS。

random输入来自固定seed的有符号float；identity让B的矩形主对角为1，其余0；cancellation使相邻k的精确积相消，奇数K另有尾项。取消模式能显示FMA/舍入误差，但单独不能代替随机输入的索引验证。每轮先将C填NaN，检查每个输出有限且误差达标，初验、每次batch后和独立E2E后均检查；任何失败不发布有效CSV。

## 运行、计时和边界

参数用`--rows M --cols N --depth K`，通用`--n`被禁用避免歧义；CSV的n是输出元素MN，matrix_m/n/k保留矩阵维度。默认256³、tile16/block256、reg-rows4、warmup3/repeats7/iterations5。reg-rows仅允许2/4，控制Register Tiled与Unrolled，其他版本每线程仍为一个输出。tile仅16/32，block必须等于tile²；tile32需同时指定block1024。

```bash
CUDA_ARCH=sm_89 python3 commands/run/run.py gemm --rows 256 --cols 256 --depth 256
CUDA_ARCH=sm_89 python3 commands/run/run.py gemm --rows 37 --cols 65 --depth 33 --pattern random
CUDA_ARCH=sm_89 python3 commands/run/run.py gemm --rows 37 --cols 65 --depth 32 --pattern cancellation
CUDA_ARCH=sm_89 python3 commands/run/run.py gemm --rows 512 --cols 512 --depth 512 --tile 32 --block 1024
CUDA_ARCH=sm_89 python3 commands/verify/run.py gemm
CUDA_ARCH=sm_89 python3 commands/sanitizer/run.py gemm
CUDA_ARCH=sm_89 python3 commands/static/run.py gemm
CUDA_ARCH=sm_89 python3 commands/nsys/run.py gemm --variant coalesced --rows 512 --cols 512 --depth 512
CUDA_ARCH=sm_89 python3 commands/ncu/run.py gemm --variant coalesced
```

M/N/K各1..4096，MNK≤2^30保证全量CPU参考有界；host数组≤256 MiB；device默认≤256 MiB且≤free显存一半。每版C=A*B重新计算，不以重复计时累计C改变数值问题。Event只包围输入驻留的kernel batch，仍可能包含launch空隙；host E2E另测pageable A/B上传+同kernel+C回读+同步。分配、CPU参考、重置、校验均不计时，不能把两个独立样本相减为精确PCIe开销。

## 观察什么、不能推断什么

先看源码模型，再用PTX/SASS确认LDG/STG、LDS/STS/BAR、FFMA及回跳；记录寄存器、stack/local、shared与展开代码体积。不同variant按symbol分别分析，不能把多kernel二进制中的总指令数当某版复杂度。未使用MMA/Tensor指令也应静态核对。

NSYS核对一个op确为一个kernel、grid/block正确，比较kernel活跃与Event batch包围范围，排除短工作量提交间隙误读。NCU需先查询当前sections/metrics，再看global sectors/bytes、shared bank、SM吞吐、occupancy与stall；不硬编码跨版本指标名。工具缺失写Not Tested，普通计时不使用重度profile替代。

理解问题：为什么Coalesced仍有每输出2K个源级load？为什么shared可能更慢？R增大是否必然提高AI或吞吐？padding让哪些有用/执行工作量不相等？为什么相同FMA次数不能单独预测速度？近0结果的相对误差为什么可很大？每个回答都应连接源码、模型和当前证据。

[exercises.md](exercises.md)和[answers.md](answers.md)分别提供综合练习与参考答案。验收需五版全部正确、完整性能/资源/误差证据和逐步优化报告，不以代码存在或编译成功代替完成。
