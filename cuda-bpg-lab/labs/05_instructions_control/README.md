# Stage 5：Instruction & Control Flow

## 当前独立动作约定

以下命令从项目根目录执行。先单独构建要使用的目标，再选择运行、校验或分析；后续动作缺产物只报错，不自动编译。源码或编译配置变化后必须重新执行构建。

`CUDA_ARCH=sm_89 python3 commands/build/run.py integer_division`
`CUDA_ARCH=sm_89 python3 commands/build/run.py loop_unroll`
`CUDA_ARCH=sm_89 python3 commands/build/run.py branch_flow`

日志固定为 commands/<action>/log/，重复同动作覆盖该目录；先阅读或手动保留需要对照的结果再重跑。历史 results/ 不写入。本页历史分析链接不代表新入口已通过 GPU 复验。


## Learning Objectives / Theory

主教材：[BPG 13.1 Ch.12–13](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-c-best-practices-guide/index.html#instruction-optimization)。本阶段从CUDA C++语义出发，对照PTX与SASS，再解释整体执行时间。指南讨论整数除法、循环展开、warp分支与predication；实际序列由本次编译器和目标决定，不硬编码“每个除法固定多少周期”。

| 实验 | 完成后应能解释 |
| --- | --- |
| integer_division | runtime/constant/shift的合法等价条件；signed负数向0截断；PTX操作如何变成SASS序列 |
| loop_unroll | unroll因子、尾部、独立chain与依赖链；循环控制减少和代码/寄存器增长的权衡 |
| branch_flow | 同工作量的任务重排；真实分支与predication；源码if不能直接证明warp divergence |

## 先验证等价，再比较指令

从根目录执行 `CUDA_ARCH=sm_89 python3 commands/verify/run.py integer_division`。先手算 unsigned `7/8=0`、`8/8=1`，以及 signed `−7/8=0`（向零截断）；再运行 `CUDA_ARCH=sm_89 python3 commands/run/run.py integer_division --n 33 --divisor 8 --signed 0 --variant runtime --warmup 1 --repeats 2 --iterations 3`，仅把 `--variant runtime` 改为 `--variant constant` 再测。**预期**输出字相同；是否变快需看数据，不能由源码行数判断。若报不支持 variant/除数，先核对参数合法范围和本次记录；正确性通过后用 `CUDA_ARCH=sm_89 python3 commands/static/run.py integer_division` 看 PTX/SASS，再解释为何编译期常量可能改变序列。循环展开与分支是后续独立对照。

## Integer Division：语义先于优化

每线程读一个uint32 state，每轮计算商q，再执行`state=mix(state)^q`，其中`mix(x)=(x*1664525+1013904223)^(x>>13)`，所有混合均为uint32模2^32运算。q参与下一轮输入，最终state写回；CPU逐轮计算，精确比较所有输出字。使用依赖递推避免`x/=d`很快变0，也避免编译器删除不可观察结果。

| variant | 除数与实现 | 同工作量比较范围 |
| --- | --- | --- |
| runtime | 正除数作为kernel参数，1..INT_MAX | 与同d常量版本比较 |
| constant | 模板常量7或8 | 仅支持d7/8，不偷偷改变用户输入 |
| manual_shift | unsigned/8直接右移；signed先取unsigned幅值、右移，再恢复符号 | 仅支持d8 |

signed模式把32位模式解读为[-2^31,2^31−1]。host reference用int64减2^32解码，device用无溢出的表达式转换；除数始终正，排除除零和INT_MIN/−1。signed商必须向0截断：−7/8=0，而常见算术右移−7>>3为−1，所以直接移位不是合法替代。manual_shift在unsigned域计算幅值，包括INT_MIN的幅值2^31，再将商恢复为模2^32的有符号位模式，不执行有符号取负溢出。混合允许无符号回绕，但并未依赖有符号溢出优化。

输入显式包含0、1、7、8、INT_MAX、INT_MIN、UINT_MAX及−7/−8位模式，后续为确定性混合序列。d1和INT_MAX边界使用runtime；constant在不支持d下直接拒绝。

## Loop Unrolling：固定递推，改变展开因子

每线程有1或4个独立uint32 chain，初值`input+c*0x9e3779b9`。每轮对各chain执行：

```cpp
x = x * 1664525u + 1013904223u;
x ^= x >> 13;
```

variants u1/u4/u8仅改变`#pragma unroll`参数；rounds是运行时值，非整4/8倍数由编译器生成尾部。每个chain单独输出到`out[c*N+i]`，不能只校验相互抵消的checksum。固定c时warp输出连续。所有整数操作精确、无浮点容差。

一条chain有跨轮依赖，多个chain暴露独立工作，但也增加寄存器需求。比较unroll必须固定N/chains/rounds；若要比较1/4 chains，可令N相反缩放保持总update数，但grid和每线程依赖结构随之改变，不能把差异仅称为ILP或单指令延迟。

## Branch Layout：只重排同一批逻辑任务

逻辑task的flag=`(task/32)&1`，低位存flag，其余31位为初始state。路径A执行乘加后右移xor，路径B执行另一组常数乘加后左移xor。两路每轮各有一次源级更新；机器指令数/吞吐是否相同须查SASS。

uniform按task顺序摆放；interleaved在每64个task中按`0,32,1,33,…31,63`排列。尾部只保留task<N的项，因此同一N下工作集合和两路task数量完全相同。CPU先生成逻辑reference，再按相同排列比对；global访问仍连续，性能运行时没有额外重排kernel。

完整64任务组中，uniform有两个各自同路的warp，interleaved有两个各16/16的混合warp。CSV的`model_mixed_warps`按实际输入（含尾部）计数，属于flag模型，不是硬件branch效率指标。block要求32的倍数，使线性输入的每32项对应warp；不把warp一致推广为整个block一致。

short body只更新1轮，loop body在各分支内执行rounds轮，是两个不同工作量场景，不能直接用其总时间比作为分支开销。每个场景只比较uniform/interleaved；源码if可能变成predicate/select或合并，反汇编确认之后才能命名实际控制流。

## Performance Model / Questions

| 实验 | 逻辑核心工作/op | global requested / 唯一工作集 |
| --- | --- | --- |
| integer_division | N×rounds次商计算，另有固定mix | 8N B |
| loop_unroll | N×chains×rounds次update | 4N(1+chains) B |
| branch_flow | N×rounds次所选路径update | 8N B |

以上都不是FP32 FLOPs，CSV的flops/GFLOPs为0表示源级核心工作没有计入FP32 FLOP，整数工作量另有logical_divisions/updates；编译器实现整数除法时仍可能使用浮点辅助指令，这个0不是实测硬件浮点指令数，更不是把未测指标填零。整数有效Gop/s=`logical_work/(event_us*1000)`，只用于同语义工作比较；不叫SASS instruction throughput。实际动态指令还包含索引、尾部、loop/branch、runtime除法辅助序列等。

理想时间需同时满足吞吐约束`T≥总实际工作/可持续吞吐`与依赖约束`T≥最长依赖路径延迟`；本实验未校准每条指令的周期或SM持续吞吐，不代入猜测值。批量Event还含提交间隙，足够多线程的kernel测得的是整体执行，不是单warp dependency latency。若要测指令周期需另建时钟/依赖链微基准。

1. 为什么q必须影响下一轮而不能只是反复除同一个常数输入？
2. 为什么d7与d8不能直接当成同数学运算的优化前后？
3. 负数/8的等价shift怎样保留向0截断，INT_MIN如何避免取负溢出？
4. u8比u1的源码短循环更少，是否保证动态指令更少或更快？
5. 为什么每个chain单独输出？跨chain独立性能被哪些资源代价抵消？
6. 同task集合重排如何控制branch比例？N=33与N=64的mixed warp模型分别是什么？
7. PTX的bra和SASS的predicate是否一一对应？如何检验短body真的没有数据分支？
8. 哪些结论需要NCU，哪些仅凭静态与计时即可支持？

## 运行与参数

从项目根运行，默认warmup5/repeats21/iterations100。每个程序可独立构建与运行。

```bash
CUDA_ARCH=sm_89 python3 commands/run/run.py integer_division --divisor 7 --signed 0
CUDA_ARCH=sm_89 python3 commands/run/run.py integer_division --divisor 8 --signed 1
CUDA_ARCH=sm_89 python3 commands/run/run.py loop_unroll --n 65536 --chains 4 --rounds 128
CUDA_ARCH=sm_89 python3 commands/run/run.py loop_unroll --n 65536 --chains 4 --rounds 131
CUDA_ARCH=sm_89 python3 commands/run/run.py branch_flow --n 65536 --body loop --rounds 128
CUDA_ARCH=sm_89 python3 commands/run/run.py branch_flow --n 65536 --body short --rounds 1
```

| 程序 | 参数/默认值 | variant |
| --- | --- | --- |
| integer_division | n65536，rounds128（1..1024），divisor7（1..INT_MAX），signed0（0/1） | all/runtime/constant/manual_shift，受d限制 |
| loop_unroll | n65536，rounds128（1..1024），chains4（仅1/4） | all/u1/u4/u8 |
| branch_flow | n65536，rounds128（1..1024），body loop；short要求rounds1 | all/uniform/interleaved |

共同N≤2^20、每op逻辑工作≤2^28、默认device分配≤256 MiB且≤free一半；block默认256，除branch外允许1..1024并检查设备/kernel限制。短body默认参数不会默默丢弃rounds，错误组合直接拒绝。CSV拒绝覆盖，任何CUDA/校验失败不发布有效性能数据。

## 验证、Profiling与证据用途

```bash
CUDA_ARCH=sm_89 python3 commands/verify/run.py integer_division
CUDA_ARCH=sm_89 python3 commands/verify/run.py loop_unroll
CUDA_ARCH=sm_89 python3 commands/verify/run.py branch_flow
CUDA_ARCH=sm_89 python3 commands/sanitizer/run.py integer_division
CUDA_ARCH=sm_89 python3 commands/sanitizer/run.py loop_unroll
CUDA_ARCH=sm_89 python3 commands/sanitizer/run.py branch_flow
CUDA_ARCH=sm_89 python3 commands/static/run.py integer_division
CUDA_ARCH=sm_89 python3 commands/static/run.py loop_unroll
CUDA_ARCH=sm_89 python3 commands/static/run.py branch_flow
CUDA_ARCH=sm_89 python3 commands/nsys/run.py loop_unroll --variant u1
CUDA_ARCH=sm_89 python3 commands/ncu/run.py branch_flow --variant interleaved --body loop --rounds 128
```

12组除法CTest覆盖正/负边界及d1/7/8/INT_MAX；18组unroll覆盖chains1/4、rounds1/3/4/7/8/9/31/128及N1/4099；24组branch覆盖N1/31/32/33/63/64/65/4099和短/长body。每个正式计时样本后逐字验证；Sanitizer各工具另跑，输出不作为基线。

静态层按kernel分开检查：除法的div.u32/div.s32、shift与最终辅助序列；unroll的非填充回跳、每体update条目、代码字节及register/stack；branch的输入flag条件、BSSY/BSYNC/BRA与predicate/select。循环尾部的自跳转填充不算被测循环，计数SASS文本条目不等于动态执行次数。NCU先查询当前sections/metrics，再找instruction/branch/active-lane/stall/occupancy相关指标；不可用项写Not Tested。NSYS用来检查时间范围与间隙，不能替代分支效率计数器。

## Analysis / Exercises / 验收

[analysis.md](analysis.md)保留采集前预测、工具发现、普通计时、反推边界与后续实验；练习与答案见[exercises.md](exercises.md)、[answers.md](answers.md)。完成标准：整数语义和边界、等量工作对照、最终代码形式、独立计时、资源与代码代价、负面结果和未测项完整。只编译成功不构成阶段验收。
