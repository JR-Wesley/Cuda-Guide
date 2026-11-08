# Stage 4：实验设计与分析记录

## 第0轮：采集前假设（2026-10-23）

本节在Stage 4代码与测量之前记录。后续数据追加在本文件，不将事后解释改写为本轮预测。S3已观察到32-register限制导致spill，却未增加理论驻留量，因此本阶段需要一个真正可能跨资源台阶的对照。

### Reduction / Block Size

实现多pass归约，每个block先由B个线程各读取最多I个元素，访问为`block*(B*I)+thread+j*B`，保持warp合并；每个block写一个部分和，重复至1个输出。基线interleaved使用跨步shared树；sequential改变树的活跃线程与索引；warp版本先warp shuffle，再经shared传递各warp结果。三者输入、输出和逻辑归约目标相同，树形和同步代价不同。

令m0=N、m(k+1)=ceil(mk/(B*I))，直到输出1；requested=`4*Σ(mk+m(k+1))`，有效数学add=N−1。实际代码还会执行零填充、局部初值累加和shuffle中的冗余运算，所以N−1不是动态指令计数。Event一次op包含全部pass，不能将它标成一个kernel的执行时间。

预测：interleaved的活跃线程模式和shared访问较差；warp减少block barrier与shared通信，但小N可能由多pass启动/间隙主导。block增大可能减少pass/partial数量，也可能降低resident blocks或加长树，不预设单调加速。固定N/I/variant扫B，再固定N/B/variant扫I；不能一次改变二者后把收益归因于单个变量。

可推翻条件：若最终SASS没有预期的barrier/shuffle，不按源级名字解释；如果两版本理论occupancy相同而性能不同，“差异全部来自occupancy”即不充分。若小N的变化与大N不同，先检查grid是否填满SM与计时范围。

### Register Pressure

使用64个独立累加值，每轮全部执行同样FMA，最终按固定顺序求和。比较unlimited与32/48/64寄存器预算，其他代码、输入、轮数和block固定。相比S3的32个值，增加活跃值用于形成更高的实际寄存器需求；这不是跨Stage性能比较，四个新版本才构成同工作量对照。

粗略资源上界：`blocks≤min(maxBlocksSM, floor(maxThreadsSM/B), floor(regsSM/(R*B)), floor(sharedSM/S))`。分配粒度会收紧上界；用Runtime occupancy API检查，不能将除法结果冒充实际资源分配。理论warp占用=`residentBlocks*ceil(B/32)/maxWarpsSM`，不等于achieved occupancy，也不代表所有SM在有限grid中都有这么多工作。

预测：压低寄存器可能提高理论驻留，但也可能spill或改变调度；性能可能非单调。若所有预算都被其他资源限制在相同驻留量，就不能声称成功测出了occupancy收益。静态spill增加但总时间下降时，也不能归纳spill总有害；需要同时看驻留变化、计算工作和时间分布。首先收集ptxas/SASS与资源，再决定是否值得做更细预算扫描。

## 第0轮记录时的状态

当时尚未采集Stage 4数据；以上为采集前模型与待验证假设，后续实测追加如下。

## 第1轮观察后提出的新对照：modulo → mask（实施前记录）

初轮三版本通过验证，静态输出中interleaved的shared树条件包含`MUFU.RCP`及商余修正指令，sequential没有该序列（[SASS](../../results/20261023T155712Z_static_whcax81h/reduction.sass.txt)）。源码的`stride`虽按2次幂递增，后端仍生成通用余数路径。因此“interleaved与sequential只差活跃线程分布”不成立；不能把全部时间差都叫warp divergence。

新增interleaved_mask，保留相同shared地址、树、barrier与活跃线程，只将`t%(2*stride)==0`替换为`(t & (2*stride-1))==0`。等价依据：正整数d=2*stride是2次幂，无溢出（B≤1024），无符号余数`t%d=t&(d−1)`。预期mask版本去掉该倒数/商余修正序列；不要求总体耗时同比下降，因为输入加载与多pass固定成本仍存在。

判定条件：先比较最终SASS，若相关序列未消失则调查优化是否表达成功；用同N/B/I的四版本新运行比较，小数据或巨大漂移不支持稳定收益。若1MiB级输入受益而64MiB级输入不受益，可作为执行指令与访存相对权重变化的线索，但无NCU不能唯一定位瓶颈。此记录在新增variant和其采样之前写入，随后追加实际结果。

## 第2轮：验证、测量与复核（2026-10-23至24，UTC+8）

当前环境重新记录为RTX 4060 Laptop GPU、CC8.9、24 SM、driver596.21、nvcc13.4.92、Runtime13040。资源API报告65536 registers/SM、1536 threads/SM、最多24 blocks/SM、L2=32 MiB；数值仅属于本机。C++17、Release、`-O3 -lineinfo --ptxas-options=-v`，原生SM89。未锁频或持续采集clock；没有将S2的漂移来源判定为某个特定因素。

普通基线均warmup5/repeats21/iterations100、pageable host、device缓冲重复使用、无profiling插桩。CSV保留input/reference/error/resource/编译元数据，独立目录保存命令、退出码、源码和二进制哈希。Event一个op覆盖全部归约pass；register实验一个op为单个update kernel，但仍是batch span/op。Host E2E包含上传、运算和结果回读，不含分配/重置/参考计算；原始CSV有其Min/Median/P95。

| 验收 | 证据 |
| --- | --- |
| 最终四版本归约：24个CTest，mixed/ones/cancellation、尾部和多pass；CSV/非法参数/覆盖保护 | [PASS](../../results/20261023T160015Z_verify_0ucg_lfi/verification.txt) |
| Register：18个N/B/rounds配置、四预算路径及CSV/边界检查 | [PASS](../../results/20261023T155703Z_verify_1sdav08i/verification.txt) |
| 归约四工具sanitizer，N4099/B128/I3、四variant | [PASS](../../results/20261023T160037Z_sanitizer_sjbi7u5z/status.txt) |
| Register四工具sanitizer，N33/B32/rounds3、四variant | [PASS](../../results/20261023T155718Z_sanitizer_fewllc_u/status.txt) |
| 最终归约PTX/SASS、mask倒数序列消失 | [PASS](../../results/20261023T160035Z_static_2fxorbeo/status.txt) |
| Register PTX/SASS/resources | [PASS](../../results/20261023T155714Z_static_bpdjhgwg/status.txt) |
| 14份基线CSV、35组variant的统计/单位和source/binary哈希复核 | [PASS](../../results/20261023T160231Z_stage4-audit_x7tv_7pe/status.txt) |

归约mixed的N=2^24中，实际绝对误差约0.817或1.183，参考结果约2.51×10^7；保守gamma界随B/I为约57–87。实际误差明显小于界，但此界不能排除任意漏算错误，因此另有要求零误差的ones、非整尾部、负数抵消及内存/同步检查。每次正式采样后全量输入对应的最终和都检查；失败时不会发布有效性能CSV。Sanitizer日志均无错误/hazard，插桩时间不纳入下表。

## 第3轮：归约规模和资源对照

### 初轮三个版本，固定B256/I4

| N | B / I | variant | Median / Min / P95 (us/op) | 理论occupancy | requested GB/s | 证据 |
| ---: | --- | --- | --- | ---: | ---: | --- |
| 4099 | 256 / 4 | interleaved | 13.578 / 10.608 / 22.845 | 100.0% | 1.21 | [CSV](../../results/20261023T155816Z_run_n4pqwrd8/samples.csv) |
| 4099 | 256 / 4 | sequential | 21.123 / 13.261 / 26.194 | 100.0% | 0.78 | [CSV](../../results/20261023T155816Z_run_n4pqwrd8/samples.csv) |
| 4099 | 256 / 4 | warp | 13.444 / 11.018 / 21.083 | 100.0% | 1.22 | [CSV](../../results/20261023T155816Z_run_n4pqwrd8/samples.csv) |
| 1048576 | 256 / 4 | interleaved | 22.764 / 22.610 / 26.634 | 100.0% | 184.62 | [CSV](../../results/20261023T155817Z_run_9ucy1dki/samples.csv) |
| 1048576 | 256 / 4 | sequential | 17.869 / 15.821 / 26.459 | 100.0% | 235.19 | [CSV](../../results/20261023T155817Z_run_9ucy1dki/samples.csv) |
| 1048576 | 256 / 4 | warp | 17.367 / 15.183 / 26.243 | 100.0% | 241.98 | [CSV](../../results/20261023T155817Z_run_9ucy1dki/samples.csv) |
| 16777216 | 256 / 4 | interleaved | 281.364 / 278.414 / 284.396 | 100.0% | 238.98 | [CSV](../../results/20261023T155818Z_run_zgbfe3cf/samples.csv) |
| 16777216 | 256 / 4 | sequential | 280.003 / 277.002 / 280.617 | 100.0% | 240.14 | [CSV](../../results/20261023T155818Z_run_zgbfe3cf/samples.csv) |
| 16777216 | 256 / 4 | warp | 279.276 / 276.367 / 279.784 | 100.0% | 240.77 | [CSV](../../results/20261023T155818Z_run_zgbfe3cf/samples.csv) |

**模型与偏差。** N4099第一pass仅5 blocks，第二pass1 block；理论occupancy100%只是资源容许上界，无法使这个小grid覆盖24 SM。sequential在这次小N中反而更慢，且Min/P95范围宽，不据此否定其地址/活跃线程模型。N=2^20时interleaved较慢；到2^24时三版本仅约279–281 us，差异远小于1Mi元素时的比例。这里的2^20元素输入实际为4 MiB；第1轮“MiB级”只指规模类别。

大N requested=67240068 B，时间约279 us，对应约241 GB/s。用以前大Copy约226 GB/s作为经验参照会得到约298 us，同量级支持调查访存吞吐，但这是跨实验经验比值，不是硬件峰值或严格时间下界。归约输出远少于Copy且访问/工作集不同，不能声称它达到或超过了某个DRAM上限。

四个最终归约kernel均40 registers、0 stack/spill；shared是动态分配，cuobjdump的静态SHARED=0不表示运行时未用shared。B256时interleaved/sequential/mask用1024 B，warp仅32 B；API驻留均6 blocks/SM，因此性能差异不能全归因于occupancy差异。

### 固定warp的B/I扫描

| N | B / I | variant | Median / Min / P95 (us/op) | 理论occupancy | requested GB/s | 证据 |
| ---: | --- | --- | --- | ---: | ---: | --- |
| 16777216 | 64 / 4 | warp | 280.658 / 277.614 / 282.255 | 100.0% | 240.99 | [CSV](../../results/20261023T155822Z_run_2a9qhu2d/samples.csv) |
| 16777216 | 128 / 4 | warp | 279.839 / 276.818 / 280.156 | 100.0% | 240.75 | [CSV](../../results/20261023T155824Z_run_zcurq0jb/samples.csv) |
| 16777216 | 512 / 4 | warp | 279.470 / 276.356 / 280.781 | 100.0% | 240.36 | [CSV](../../results/20261023T155825Z_run_wzpk_7z5/samples.csv) |
| 16777216 | 1024 / 4 | warp | 277.074 / 274.371 / 279.603 | 66.7% | 242.32 | [CSV](../../results/20261023T155827Z_run_mjkz3hk1/samples.csv) |
| 16777216 | 256 / 1 | warp | 311.050 / 308.378 / 313.344 | 100.0% | 217.44 | [CSV](../../results/20261023T155829Z_run_v_o1w21e/samples.csv) |
| 16777216 | 256 / 16 | warp | 276.992 / 274.022 / 279.112 | 100.0% | 242.40 | [CSV](../../results/20261023T155831Z_run_ymucw8d4/samples.csv) |

B256/I4基线见上一表，279.276 us。B64/128/256/512驻留分别24/12/6/3 blocks，即理论100%；B1024只有1 block，理论66.7%，却没有变慢。B1024还把pass数从3减为2，因此这不是“只改变occupancy”的实验，不能反推出更低occupancy本身更好。能否认的是“occupancy降至2/3，时间一定增加至1.5倍”这种单因素预测。约1%的差值不作稳定排序。

I1→4→16在B256下约311→279→277 us，API occupancy不变。I1拥有更多first-pass blocks（65536 vs16384 vs4096）、更多partial流量；I16进一步少一个pass。有效请求字节只从67635204降到67141636，约0.7%，而I1到I16时间差约11%，说明只按global逻辑字节变化解释也不充分。调度、每线程工作摊销与launch数是合理候选；真实stall及memory transactions未测，不能给各自分摊比例。

## 第4轮：静态发现驱动的mask优化，结果如何修正解释

| N | B / I | variant | Median / Min / P95 (us/op) | 理论occupancy | requested GB/s | 证据 |
| ---: | --- | --- | --- | ---: | ---: | --- |
| 1048576 | 256 / 4 | interleaved | 29.512 / 22.589 / 32.850 | 100.0% | 142.40 | [CSV](../../results/20261023T160117Z_run_6kellpys/samples.csv) |
| 1048576 | 256 / 4 | interleaved_mask | 16.657 / 16.445 / 24.863 | 100.0% | 252.29 | [CSV](../../results/20261023T160117Z_run_6kellpys/samples.csv) |
| 1048576 | 256 / 4 | sequential | 16.323 / 15.780 / 26.275 | 100.0% | 257.47 | [CSV](../../results/20261023T160117Z_run_6kellpys/samples.csv) |
| 1048576 | 256 / 4 | warp | 16.169 / 14.193 / 25.242 | 100.0% | 259.91 | [CSV](../../results/20261023T160117Z_run_6kellpys/samples.csv) |
| 16777216 | 256 / 4 | interleaved | 281.354 / 278.344 / 281.590 | 100.0% | 238.99 | [CSV](../../results/20261023T160119Z_run_f4aduazc/samples.csv) |
| 16777216 | 256 / 4 | interleaved_mask | 280.176 / 277.195 / 280.778 | 100.0% | 239.99 | [CSV](../../results/20261023T160119Z_run_f4aduazc/samples.csv) |
| 16777216 | 256 / 4 | sequential | 280.102 / 277.135 / 282.418 | 100.0% | 240.06 | [CSV](../../results/20261023T160119Z_run_f4aduazc/samples.csv) |
| 16777216 | 256 / 4 | warp | 279.368 / 276.286 / 292.137 | 100.0% | 240.69 | [CSV](../../results/20261023T160119Z_run_f4aduazc/samples.csv) |

最终SASS中原interleaved仍有`MUFU.RCP`及相关商余序列，mask版本没有；两个kernel均40 registers、同shared容量与理论驻留量。因此干预成功改变了条件计算，没有借换树或提高occupancy制造收益。N=2^20时29.512→16.657 us，和预期方向一致；原版Min22.589/P9532.850，mask Min16.445/P9524.863，波动仍不小。初轮原版Median22.764也低于新一轮29.512，故不能宣称固定1.77倍收益。

N=2^24时281.354→280.176 us，仅约0.42%；即使指令被明确删除，大问题上的总吞吐也几乎不变。结论限定为“余数序列确实移除，较小工作集本次有明显观测差异，大规模收益很小”，不把SASS优化自动等价为稳定端到端优化。

还需修正第0轮的笼统共享访问预期：这里interleaved的活跃lane各访问不同bank，不能把它直接当作S3的同bank转置冲突。它的特点是活跃线程分散、更多warp在较深层仍有少数活跃lane，以及通用余数路径；mask对照只隔离了后者。下一轮若要分开计算与通信成本，应固定合并输入和树工作量，查询动态指令/eligible warp/stall；这部分待NCU环境，不从总时间唯一反演。

## 第5轮：Register Pressure的反例

编译后的实际资源和静态指令：

| 预算 | registers/thread | stack B/thread | ptxas spill stores / loads B | 静态LDL / STL条目 | update FFMA条目 |
| --- | ---: | ---: | --- | --- | ---: |
| unlimited | 71 | 0 | 0 / 0 | 0 / 0 | 64 |
| r64 | 64 | 48 | 80 / 80 | 20 / 20 | 64 |
| r48 | 48 | 168 | 320 / 320 | 80 / 80 | 64 |
| r32 | 32 | 280 | 492 / 492 | 123 / 123 | 64 |

依据：[最初ptxas编译日志](../../results/20261023T155500Z_build_vsfmi8m6/012_cmake.log)、[SASS](../../results/20261023T155714Z_static_bpdjhgwg/register_pressure.sass.txt)及运行CSV。条目数是反汇编文本中的静态出现次数，循环会重复执行，不能把它称作动态instructions/thread；spill B也不是实测每op流量。每版本64个FFMA条目支持核心update没有被整个消除，不代表整个kernel只有64条指令。

| N | B / I | variant | Median / Min / P95 (us/op) | 理论occupancy | requested GB/s | 证据 |
| ---: | --- | --- | --- | ---: | ---: | --- |
| 65536 | 128 / — | unlimited | 22.548 / 22.426 / 25.242 | 58.3% | 23.25 | [CSV](../../results/20261023T155833Z_run_2clvtnt0/samples.csv) |
| 65536 | 128 / — | r32 | 519.975 / 517.622 / 523.179 | 100.0% | 1.01 | [CSV](../../results/20261023T155833Z_run_2clvtnt0/samples.csv) |
| 65536 | 128 / — | r48 | 357.315 / 356.362 / 359.076 | 83.3% | 1.47 | [CSV](../../results/20261023T155833Z_run_2clvtnt0/samples.csv) |
| 65536 | 128 / — | r64 | 100.594 / 100.301 / 102.881 | 66.7% | 5.21 | [CSV](../../results/20261023T155833Z_run_2clvtnt0/samples.csv) |
| 65536 | 256 / — | unlimited | 25.373 / 22.580 / 26.409 | 50.0% | 20.66 | [CSV](../../results/20261023T155837Z_run_mqzuquv7/samples.csv) |
| 65536 | 256 / — | r32 | 519.158 / 516.424 / 521.144 | 100.0% | 1.01 | [CSV](../../results/20261023T155837Z_run_mqzuquv7/samples.csv) |
| 65536 | 256 / — | r48 | 358.676 / 357.960 / 361.871 | 83.3% | 1.46 | [CSV](../../results/20261023T155837Z_run_mqzuquv7/samples.csv) |
| 65536 | 256 / — | r64 | 101.701 / 101.519 / 103.984 | 66.7% | 5.16 | [CSV](../../results/20261023T155837Z_run_mqzuquv7/samples.csv) |
| 65536 | 512 / — | unlimited | 25.672 / 25.610 / 27.812 | 33.3% | 20.42 | [CSV](../../results/20261023T155840Z_run_8s0d1twk/samples.csv) |
| 65536 | 512 / — | r32 | 536.522 / 533.381 / 539.645 | 100.0% | 0.98 | [CSV](../../results/20261023T155840Z_run_8s0d1twk/samples.csv) |
| 65536 | 512 / — | r48 | 382.717 / 380.413 / 387.881 | 66.7% | 1.37 | [CSV](../../results/20261023T155840Z_run_8s0d1twk/samples.csv) |
| 65536 | 512 / — | r64 | 105.093 / 104.999 / 109.179 | 66.7% | 4.99 | [CSV](../../results/20261023T155840Z_run_8s0d1twk/samples.csv) |

**代入资源模型。** B256时，用65536/(R*256)得到raw register约束3/4/5/8 blocks，取线程约束6后为3/4/5/6，实际API一致；因此这次确实跨越了50%/66.7%/83.3%/100%台阶。B128 unlimited为7 blocks（58.3%），B512只有1 block（33.3%），说明block选择与寄存器预算不能独立看。

**观测与判定。** B256 unlimited约25.373 us，r32约519.158 us，时间增加约20.5倍，同时理论occupancy翻倍。这个量级远大于各自本次Min/P95跨度；足以形成“提高理论occupancy但性能下降”的实测反例。static spill与LDL/STL支持更严格预算引入额外工作这一解释，但没有实际local流量、cache和stall指标，不能把20倍全部分配给某一种延迟，也不能说achieved occupancy已达到100%。

B128 unlimited约22.548 us，update工作量268435456 FLOP，对应约11905 GFLOP/s；只计算64路FMA，不含初始化/final sum。这一独立累加值设计暴露了较多ILP，说明允许更多寄存器保留局部状态可能比追求满驻留更合适；ILP程度与运行stall仍需工具量化。CSV的global requested GB/s不能衡量spill代价；8N并不包含真实local流量。

**优化决定。** 当前保留unlimited作为此特定计算的候选，保留全部预算版本作为反例。B128/256/512间较小差异尚未做充分顺序交换，不指定通用最佳block。若后续要寻找中间预算的收益，应先确认spill消失或驻留台阶的位置再细扫；不能先按“少寄存器”选参数再只报告最好点。

## 第6轮：用NSYS核对多pass计时对象

对N4099/B256/I4/warp单独采集[warmup5/repeats3/K100时间线](../../results/20261023T155916Z_nsys_64075cs_/timeline.nsys-rep)。当前collect流程先验证1次，随后每轮5次warmup、100次batch、1次E2E，共`1+3*(5+100+1)=319`个逻辑op，每op两pass，共638个kernel；[NSYS统计](../../results/20261023T160052Z_stage4-nsys-analysis_ketwkgfq/summary_cuda_gpu_kern_sum.csv)正好638次。

[只读时间戳分析](../../results/20261023T160231Z_stage4-audit_x7tv_7pe/002_python3.log)进一步检查单stream、grid5/1交替和每对非重叠；两个pass活跃Median约1.280/1.248 us，同一op的活跃时间和Median2.528 us；从第一pass start到第二pass end的span Median4.384 us。按源码采样顺序定位三个正式batch，kernel包围区间/100分别12.765、13.884、15.704 us，明显大于两pass活跃和，表明op之间也有间隙。

同次[Event CSV](../../results/20261023T155916Z_nsys_64075cs_/instrumented_samples.csv)Median14.380 us，其边界是Event而非首末kernel时间戳；不要求这两种范围完全相等，也不将不同集合的Median直接相减作严格分摊。未插桩小N基线仍是13.444 us，不能用profile的14.380替换。这个结果支持小N时先调查提交/间隙而非仅按shared或ALU优化解释全部时间。

## 阶段结论、未测项与下一阶段

Stage 4基础验收完成：四个归约对照、block/items扫描、四寄存器预算及block扫描、CPU/边界/同步检查、静态资源与指令、独立计时/CSV审计、NSYS计时核对、原理与练习/答案齐备。实际完成了一轮“从SASS发现额外算术→提出等价改动→静态验证→计时→收窄收益结论”的循环，原始基线和预测均保留。

NCU当前仍不可用，achieved occupancy、eligible warp、stall、实际DRAM/local bytes及cache hit均Not Tested；未执行跨进程排序稳定性确认、其他SM7x/8x或精确Toolkit13.1复现。原有PTX JIT兼容性失败仍未消除，本次只使用原生SM89。

下一阶段S5从这里的运行时取模发现出发，系统比较无符号整数除法/常量/2次幂、unroll与分支布局；每次都从明确语义和最终SASS检查开始，不把源码操作数当作硬件指令数。
