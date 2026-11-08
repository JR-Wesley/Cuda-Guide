# Stage 4：Execution Configuration

## 当前独立动作约定

以下命令从项目根目录执行。先单独构建要使用的目标，再选择运行、校验或分析；后续动作缺产物只报错，不自动编译。源码或编译配置变化后必须重新执行构建。

`CUDA_ARCH=sm_89 python3 commands/build/run.py reduction`
`CUDA_ARCH=sm_89 python3 commands/build/run.py register_pressure`

日志固定为 commands/<action>/log/，重复同动作覆盖该目录；先阅读或手动保留需要对照的结果再重跑。历史 results/ 不写入。本页历史分析链接不代表新入口已通过 GPU 复验。


## Learning Objectives / Theory

主教材：[BPG 13.1 Ch.11](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-c-best-practices-guide/index.html#execution-configuration-optimizations)。本阶段将“更多线程可能隐藏延迟”变成可计算、可检验的资源与工作分配问题；不能仅凭occupancy高低选赢家。

| 实验 | 完成后应能解释 |
| --- | --- |
| reduction | 多pass归约、尾部、树形与barrier/shuffle、block和每线程工作量的相互影响 |
| register_pressure | register allocation、spill与理论驻留台阶；同一计算在不同预算下的时间权衡 |

BPG提出occupancy、寄存器分配和延迟隐藏之间的联系，但更高occupancy不保证更快。这里通过自有实现和本机资源报告检验，不套用其他架构的峰值或粒度。

## 先用一个不能整除的输入

从根目录执行 `CUDA_ARCH=sm_89 python3 commands/verify/run.py reduction`，再运行 `CUDA_ARCH=sm_89 python3 commands/run/run.py reduction --n 4099 --block 128 --items 3 --variant interleaved --warmup 1 --repeats 2 --iterations 3`。先预测一轮中每个 block 最多处理 `128×3=384` 个数，最后一个 block 必有尾部；核对本次 CSV 的 `validation`、`passes` 和时间范围。**预期**校验通过，但不预测这个短输入的速度。第二次仅改 `--variant warp`，比较通信方式与生成的资源；错误先从 `commands/verify/log/` 的 CTest、状态和日志查起，不能凭更高理论 occupancy 断言一定更快。寄存器压力是另一个单变量实验，放在归约正确性之后。

## Reduction：代码模型与正确性

每个block负责至多`B*I`个元素，B=`--block`，I=`--items`。线程t依次读取`blockIdx*B*I+t+j*B`，j从0到I−1；同一轮j中warp访问连续。越界读取贡献0，所有线程继续参加block同步。每个block输出一个部分和，下一pass继续同样过程，直到一个输出。

| variant | block内通信 | 共享字节/block |
| --- | --- | ---: |
| interleaved | stride=1,2,4…，`t%(2*stride)==0`的线程相加，每层barrier | 4B |
| interleaved_mask | 与interleaved相同树，仅将余数条件改为等价位掩码 | 4B |
| sequential | stride=B/2…1，`t<stride`的连续线程相加，每层barrier | 4B |
| warp | 完整warp内shuffle；每warp写一个shared结果；一次block barrier；warp0再归约 | 4B/32 |

interleaved_mask的独立因素是将`t%(2*stride)`改为`t & (2*stride−1)`；stride始终为2次幂，因此活跃线程和读写地址不变。初轮静态检查发现原版取模生成倒数/商余修正序列，新增对照用于分开检查算术成本与活跃线程分布，不把全部差异归因于divergence。

B要求32..1024的2次幂，保证物理warp完整。即使输入尾部不足32元素，所有lane仍执行shuffle，越界lane的值是0；full mask指参加shuffle的线程，不是有效输入数。warp0第二次归约同样有32个参与者，不足的warp和填0。shared读取发生在block barrier后，不依赖隐含warp同步；不用volatile代替通信保证。

令m0=N，`m(k+1)=ceil(mk/(B*I))`，直到输出1。**N=1也执行一个pass**，没有跳过操作导致零时间。逻辑请求字节=`4*Σ(mk+m(k+1))`；数学有效add=N−1；实际动态add还包括初始零累加、padding和未被lane0消费的shuffle结果，需单独看SASS/计数器。两个交替partial缓冲区容量均为m1，分配=`4*(N+2*m1)`。每次op含全部pass，在同一stream顺序执行，无跨block全局同步假设。

例如N=4099、B=128、I=3：4099→11→1，两个pass，requested=`4*(4099+11+11+1)=16488 B`。第一pass只有11 blocks，小于本机24 SM；即使API理论occupancy为100%，这个有限grid仍不能铺满设备。

CPU以double累加实际float输入。mixed为正数`1+((17*i)%257)/257`；cancellation将奇数项取负；ones全1。N≤2^24时ones所有树节点都是可精确表示的非负整数，要求误差为0。其他输入使用`|actual-reference|≤max(1e-6, gamma_h*sum|input|)`，u=FP32 epsilon/2，`gamma_h=h*u/(1-h*u)`；h保守取`passes*(I+log2(B)+5)`，覆盖每线程串行sum及两级shuffle深度。它是无溢出、正常有限数条件下的舍入上界，不是对所有输入的固定相对容差；强抵消时可能很宽，因此CSV同时记录reference、sumabs、实际误差和容差，另外保留严格ones与边界检查，不能靠宽容差掩盖漏算。

## Register Pressure：资源模型

64个独立值每轮执行相同FMA，最后固定次序求和；unlimited与r32/r48/r64只改变`__maxnreg__`预算。所有版本使用同一内联helper，CPU逐值用float `std::fma`，atol=rtol=1e-5，拒绝NaN/Inf。预算是上限，实际register数以编译/runtime为准；编译器不一定恰好用满。

对本实验的零shared内核，先计算粗略上界：

```text
resident_blocks <= min(max_blocks_SM,
                       floor(max_threads_SM / B),
                       floor(registers_SM / (R * B)))
warps_per_block = ceil(B / 32)
occupancy_theory = resident_blocks_API * warps_per_block / max_warps_SM
```

简单寄存器乘法忽略warp/block分配粒度；非32倍数block也浪费部分warp资源，因此上式只是宽松上界。API给定真实kernel与block后得到的驻留量应不超过它；achieved occupancy、eligible warps和stall仍需NCU。增加寄存器预算可能减少spill并暴露指令级并行，也可能减少驻留warp，预测允许非单调。

核心update FLOP=`2*N*64*rounds`，不含初始化及最后求和；global requested=`8N`。spill流量不是global requested的组成，实际local/DRAM流量未测时不能用8N计算物理带宽。private backing额外限N≤65536、rounds≤256，避免无界增长。

## Questions

1. 为何最后一个不完整block仍必须全部参加barrier和shuffle？
2. 为什么多pass的requested略大于4N？为何数学add与动态指令数不同？
3. block增大减少partial数量，是否一定减少pass数？用4099与2^24分别计算。
4. 相同API occupancy却有不同时间，可能排除哪种过强解释？还不能判断什么？
5. 寄存器预算、实际register数、stack和spill分别来自哪一层证据？
6. 理论100% occupancy但只有11 blocks，为什么可能无法填满24 SM？
7. 高occupancy更慢时，如何区分spill、指令级并行、clock漂移及launch开销？

## 实现与运行

从项目根运行；默认warmup5、repeats21、iterations100，CSV保存每个原始样本与Min/Median/P95。默认device分配上限256 MiB且≤free一半。

```bash
CUDA_ARCH=sm_89 python3 commands/build/run.py reduction
CUDA_ARCH=sm_89 python3 commands/run/run.py reduction --n 16777216 --block 256 --items 4
CUDA_ARCH=sm_89 python3 commands/run/run.py reduction --n 4099 --pattern cancellation
# 固定其他参数，逐项扫描；不要把一次同时改变B/I当单变量实验。
for block in 64 128 256 512 1024; do
  CUDA_ARCH=sm_89 python3 commands/run/run.py reduction --n 16777216 --block "$block" --items 4 --variant warp
done
for items in 1 4 16; do
  CUDA_ARCH=sm_89 python3 commands/run/run.py reduction --n 16777216 --block 256 --items "$items" --variant warp
done
CUDA_ARCH=sm_89 python3 commands/run/run.py register_pressure --n 65536 --block 256 --rounds 32
```

| 程序 | 参数 | variants |
| --- | --- | --- |
| reduction | n默认2^20，≤2^24；block默认256且为32..1024的2次幂；items默认4、1..32；pattern mixed/ones/cancellation | all/interleaved/interleaved_mask/sequential/warp |
| register_pressure | n默认65536且≤65536；rounds默认32、1..256；block1..1024但必须满足实际kernel资源限制 | all/unlimited/r32/r48/r64 |

无`__maxnreg__`编译能力时预算版本明确Not Tested，unlimited仍可运行；CMake编译探针不会把Toolkit默认当保证。某些大block+高register组合可能无法驻留，必须报CUDA错误，不把它作为零时间样本。

## 验证与工具如何帮助判断

```bash
CUDA_ARCH=sm_89 python3 commands/verify/run.py reduction
CUDA_ARCH=sm_89 python3 commands/verify/run.py register_pressure
CUDA_ARCH=sm_89 python3 commands/sanitizer/run.py reduction
CUDA_ARCH=sm_89 python3 commands/sanitizer/run.py register_pressure
CUDA_ARCH=sm_89 python3 commands/static/run.py reduction
CUDA_ARCH=sm_89 python3 commands/static/run.py register_pressure
CUDA_ARCH=sm_89 python3 commands/nsys/run.py reduction --variant warp --n 4099
CUDA_ARCH=sm_89 python3 commands/ncu/run.py register_pressure --variant r32 --n 65536
```

CTest覆盖N=1、31/32/33、非整block与多pass，三类归约输入，覆盖四个variant；寄存器实验覆盖不同N/block/rounds。verify还独立复算CSV模型/统计、预算与驻留上界，检查非法参数与覆盖保护。Sanitizer用非整block输入验证全部variant，四工具分别运行；racecheck/synccheck特别检查归约shared通信。

PTX/SASS检查要确认SHFL、BAR、LDS/STS是否存在；资源报告与ptxas说明register/stack/spill。NSYS用于数清各pass和判断launch间隙，不能用总span反推单pass活跃时长。NCU先查询sections/metrics，关注资源限制、achieved occupancy、eligible warps、local traffic、memory/compute throughput、stall；不可用即Not Tested。两类profiler耗时均不替换独立基线。

## 验收与后续

见[analysis.md](analysis.md)的采集前假设、逐轮实测与归因边界；[exercises.md](exercises.md)和[answers.md](answers.md)分离。验收要求四种归约正确、B/I/register sweep可复现、数学与资源模型可复算、工具证据与预期对照、负面结果和未测指标保留。未完成测量/报告时不能仅凭编译通过标记阶段完成。
