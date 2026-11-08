# Stage 6：传输、流水线和并发实验记录

## 第0轮：实现前模型与判定条件（2026-10-24，UTC+8）

S1/S4/S5已经区分kernel活跃时间、Event批量span和host端到端。S6继续沿用这些区分，同时增加copy与kernel的时间区间交集，只有实际交集才是重叠证据。以下在本阶段实现与采样之前记录。

### Host Transfer

在相同N、方向、stream和调用方式下比较pageable与cudaMallocHost pinned；两个版本都调用cudaMemcpyAsync并最终同步。H2D/D2H分别运行，requested=4N，不将双向字节与单向时间混算。Event测copy所在stream的批量span，host batch及单次E2E都包含提交到同步完成；pin/free、初始化和验证在计时外，另明确适用边界。

预测pinned可能提高传输吞吐且使API提交更适合流水线，但Async名字不保证所有参数组合对host非阻塞；pageable可能经staging并阻塞。NSYS的API区间与device memcpy区间分别检查，不只根据host E2E反推staging成本。小copy可能受固定成本影响，不预设pinned所有规模都更快。

### Stream Pipeline

固定总N、chunk、FMA轮数和buffer数，比较pageable_serial、pinned_serial、pinned_pipeline、pageable_pipeline。serial有效worker=1，pipeline使用指定streams；同一case仍保持相同chunk划分和设备buffer ring，逐项改变host memory type或stream数。每个chunk依次H2D→kernel→D2H，输入不可变，host输出区间互不重叠。

device ring有独立buffer数量；chunk c用slot=c%buffers和worker=c%streams。复用slot前该worker等待slot_done事件；事件在上次D2H后记录，因此buffers与streams不必相等也不产生覆盖。coordinator记录gate，workers等待gate，所有worker完成事件回到coordinator汇合；下一op的gate在该汇合之后。所有stream使用nonblocking创建，不依赖legacy default stream隐式顺序。

模型：每op H2D=4N、D2H=4N、kernel逻辑读写=8N、FMA FLOP=2*N*rounds；host-transfer requested字段用8N，kernel字节另列，不相加冒充单个内存层的带宽。分配为2*slot_elements*buffers*4，尾chunk受实际元素数保护。K个完整chunk的简单串行时间≈K*(H+C+D)，理想三阶段流水线≈H+C+D+(K−1)*max(H,C,D)，需足够buffer、独立引擎和无竞争；不能将模型的max值当实测。

普通Event每op覆盖完整pipeline及事件汇合，host单次E2E包含同一批传输、计算与同步，排除CPU参考和分配。每个op之间保留全局依赖，batch不偷偷变成跨op复用数据的不同工作。预期适当chunk/buffer/stream能重叠，但细chunk可能被API提交成本主导，pageable可能限制重叠；必须用NSYS验证。

### Concurrent Kernels

同一总N切为jobs个互不重叠区间，使用相同FMA递推和总轮数。serial在一个worker中顺序发出jobs个kernel；concurrent给每job一条worker stream。设备输入预先驻留，Event只测kernel组与汇合，host E2E另含上传和回读。小job可能不能填满SM，多job并发有机会填补空闲；大job可能已占满资源而没有收益。不能通过单个整grid基线与多个小kernel对照混淆launch数量，两个版本都发出相同jobs数。

用区间并发度和kernel-kernel交集证明并发；API concurrentKernels仅说明能力，不保证本次重叠。CPU reference逐元素验证，测试不整除jobs、tail、streams/buffers不匹配。没有copy/compute或kernel/kernel交集时，报告没有观察到并发及可能条件，不因工具或预期而捏造收益。

### 本轮范围

先完成以上必做实验；SM80+ cp.async为可选扩展，本轮不实现，不借此替代pinned/streams/pipeline验收。所有结果仅单GPU，最高SM89，不引入SM90特性。

## 第1轮后的干预设计：提交顺序（采集下一轮前记录）

首轮普通N=2^20、chunk=2^16、rounds128、buffers3/streams3，pinned_serial/pinned_pipeline Median分别1873.306/1886.310 us，未观察到加速。NSYS相关ID分析显示四种variant的copy-compute交集全为0；pinned_pipeline一个正式op内kernel活跃和约120 us、copy活跃和约845 us，而活动包围span约1776 us。也就是说，现有数据不支持“已经成功重叠但加速较小”，它直接显示本次没有交集。设备报告asyncEngineCount=1。

从源码看，host按H0→K0→D0→H1→K1→D1提交。单copy引擎上，若待执行D0排在H1之前，而D0依赖K0，可能让后续H1等候，形成队列头部阻塞；另一个竞争解释是host/driver提交间隙已把短kernel机会耗尽。时间线有大量空隙，不能仅凭引擎数量认定唯一原因。

下一轮增加`--order phased`对照：每组最多buffers个chunk先提交全部H2D，再全部kernel，再全部D2H；保留`interleaved`原版，同样N/chunk/rounds/buffer/stream及kernel数，不改计算。每组仍用slot_done保护跨组复用，不增加host全局同步。预测phased可能让后续H2D与前面的kernel交叠；若交集仍0，则不能宣称已解除瓶颈，需要改变计算长度或检查提交间隙。phased也会改变serial的阶段顺序，因此必须分别记录order，不把它混入只改变stream数量的比较。

验证先覆盖buffers≠streams、单buffer、尾部和重复op，再采普通计时和NSYS。对phased固定buffers3扫描streams1/2/3/4；固定streams3扫描buffers1/2/3；增加chunk同时报告launch数与buffer容量的派生变化。若只改变提交顺序就有正交集，支持顺序影响重叠；时间差仍需A/B/B/A复测，不能由理论模型承诺固定加速。

## 第1轮证据复盘：先辨明计时与实际重叠

以下为采样后的分析，与上面的采集前预测区分。设备为RTX4060 Laptop，CC8.9、24SM、8 GiB、Runtime报告asyncEngineCount=1、concurrentKernels=1；driver596.21、Toolkit13.4.92、原生sm89。编译使用项目Release参数，未锁频或修改驱动/计数器权限。环境和命令随每份证据保存。

### Copy：host内存类型改变了什么

warmup3、repeats11、batch K10；N=4096对应单向16 KiB，N=4194304对应16 MiB。表中Event为Min/Median/P95中的Median，完整分布在CSV；host E2E为另一次单copy+同步的Median。

| 方向 / 大小 | pageable Event us / host E2E us | pinned Event us / host E2E us | 原始CSV |
| --- | --- | --- | --- |
| H2D / 16 KiB | 10.528 / 22.051 | 6.195 / 21.530 | [copy small H](../../results/20261023T164322Z_run_y3om24sj/samples.csv) |
| D2H / 16 KiB | 41.571 / 38.960 | 6.013 / 34.195 | [copy small D](../../results/20261023T164328Z_run_5yzd38sl/samples.csv) |
| H2D / 16 MiB | 2268.752 / 2487.361 | 1586.368 / 1614.169 | [copy large H](../../results/20261023T164330Z_run_2tmfnnd0/samples.csv) |
| D2H / 16 MiB | 2305.686 / 2601.289 | 1750.006 / 1961.822 | [copy large D](../../results/20261023T164333Z_run_aqb5e691/samples.csv) |

大copy的requested有效带宽：pageable H/D约7.395/7.276 GB/s，pinned约10.576/9.587 GB/s。公式是16,777,216/(time_us×1000)，使用十进制GB/s；这些是本次完整stream span对应吞吐，非PCIe理论峰值。小copy有明显固定开销和波动，例如small H pinned Event Min/Median/P95约3.571/6.195/17.498 us；不从一个微秒级中位数推广稳定排序。

为什么pinned更快还不能直接归因于“DMA本身更快”？后面的同chunk NSYS给出一个区分证据：首轮pipeline的262144 B copy，pageable_serial H/D的device duration Median约26.016/25.760 us，而host API约28.411/127.783 us；pinned_serial device约26.672/25.552 us，API约5.613/9.356 us。device copy量级相近，host调用差异很大，支持pageable引入提交/同步等额外成本。这里没有测staging内部CPU memcpy，不能把两个API时间之差全部命名为staging成本；也不能把pipeline的chunk profile直接当作16 MiB独立copy的组成分解。

### Pipeline：负结果推动提交顺序对照

[首轮普通CSV](../../results/20261023T164336Z_run_j_5la2wv/samples.csv)固定N=2^20、chunk=2^16、rounds128、B256、buffers3、requested streams3，3/11/10。16个kernel/op，host总传输8 MiB、device ring1.5 MiB、算术268,435,456 FLOP/op。

| variant | Event Median us | host E2E Median us |
| --- | ---: | ---: |
| pageable_serial | 3149.296 | 3074.666 |
| pinned_serial | 1873.306 | 1851.410 |
| pinned_pipeline | 1886.310 | 1800.507 |
| pageable_pipeline | 2818.662 | 2695.344 |

pinned内存类型的差异明显大于此轮pinned serial/pipeline的差异。两种计时列来自不同样本，排序不完全一致不构成矛盾，也不能选择其中有利的一列当全部结论。

[NSYS原始记录](../../results/20261023T164343Z_nsys_coevxn3a/timeline.nsys-rep)使用1/1/2，每variant有`1+1*(1+2+1)=5`个op，合计320 kernels、640 memcpy；每variant H/D各80次、各20 MiB。通过runtime correlation恢复host提交顺序，再检查每chunk H→K→D、stream映射、grid、copy大小、slot复用和op间join。device start排序可能与host提交不同，因此不直接按device时间切chunk。

[独立解析结果](../../results/20261023T164612Z_stage6-timeline-analysis_2ztr3c18/pipeline.json)中两个正式batch op的Median如下，单位us；其余初验/warmup/E2E逐op保留，未混入这张正式batch表。

| variant | kernel active和 | copy active和 | copy∩compute | 活动包围span |
| --- | ---: | ---: | ---: | ---: |
| pageable_serial | 120.320 | 824.666 | 0 | 2709.850 |
| pinned_serial | 120.255 | 826.071 | 0 | 1849.937 |
| pinned_pipeline | 119.999 | 845.112 | 0 | 1775.697 |
| pageable_pipeline | 120.031 | 847.592 | 0 | 2835.992 |

这里0来自实测区间交集，不是把未测值写0。各kernel约7.5 us、copy约26 us，但整个op远长于活动之和；相邻活动间存在显著空隙。无重叠条件下，pinned_pipeline包围span−活动并集约810.586 us，属于时间线空隙合计，不等于单一API开销。coordinator gate/join的边界还使Event span约1794.224 us大于活动包围span。以上数字只用于解释插桩运行，不能减去普通1873/1886 us做差值归因。

### Concurrent Kernels：保持launch数与总工作量

两个版本都jobs4、rounds4096、B256，3/11/10。N4096每job1024元素，仅4 blocks，单job不足以覆盖24SM；N65536每job16384元素、64 blocks，grid更大，但也不能仅由64>24推定每个SM饱和。

| N | serial Event / E2E Median us | concurrent Event / E2E Median us | 证据 |
| --- | --- | --- | --- |
| 4096 | 210.419 / 286.531 | 83.763 / 123.066 | [小grid](../../results/20261023T164340Z_run_rcsmtqm1/samples.csv) |
| 65536 | 302.490 / 587.802 | 208.781 / 421.765 | [较大grid](../../results/20261023T164341Z_run_k688kbk0/samples.csv) |

本次小grid Event约2.512倍、较大grid约1.449倍，但样本未覆盖所有规模与时钟状态。较大grid仍有收益，否定了“只要单job覆盖所有SM就必然没有并发收益”的过强推论。资源还有驻留block/warp、长依赖链与发射等维度；没有NCU，不能唯一归因为某种stall。

[并发NSYS](../../results/20261023T164357Z_nsys_w75rzwpi/timeline.nsys-rep)与[解析](../../results/20261023T164612Z_stage6-timeline-analysis_2ztr3c18/concurrent.json)核对40 kernels=2variants×5ops×4jobs。正式batch每op中serial kernel active和约204.079 us、包围208.478 us、并发≥2长度0；concurrent active和约204.286 us、并集87.919 us、并发≥2长度56.816 us，观察到峰值4个kernel。工作活跃和相近、并集缩短，这是本次真实kernel重叠的证据；它比单纯时间下降或capability字段更有区分力。Event另含gate/join及间隙，不能将87.919直接等同普通83.763。

### 静态代码把模型约束到实际指令

[transform SASS](../../results/20261023T164122Z_static_8ciademt/stream_pipeline.sass.txt)在0x100有LDG，0x150–0x180为含FFMA、计数/比较/回跳的单轮循环，0x190有STG；没有把递推折叠成一次表达式。ptxas/resource显示8 registers、0 stack/local/shared；B256的Runtime理论驻留6 blocks/SM（48 warps，线程上限），不是实测achieved occupancy。Nsight活动记录中的registersPerThread为16而程序Runtime属性为8，本文保留这个报告差异，不拿插桩字段替换普通二进制资源，也未证明差异原因。

[host_transfer静态状态](../../results/20261023T164110Z_static_pdkf3vr6/host_transfer.static.txt)明确Not Applicable：Runtime copy不是本程序自写kernel。源码包含的未调用transform不能当作被测传输实现。NCU未安装，cache/DRAM流量、active warp、stall、pin/unpin成本均未测；时间线活动区间与源码推导不替代这些指标。

## 第2轮后的补测设计：延长计算窗口（补测前记录）

phased首轮仍没有观察到copy∩compute交集；正式op中却出现约17.968 us的kernel-kernel交集。这说明不同worker的kernel可以并发，但仅改变提交顺序尚不足以实现本次copy-compute目标。此次普通和NSYS绝对时间比第1轮明显变长，不能跨轮直接认定phased更差或更好，A/B/B/A与单变量扫描正在采集。

下一项区分实验固定N=65536、chunk4096、16chunks、buffers3/streams3、rounds4096，对比interleaved/phased。总FMA工作2^28次，仍在CPU参考上限内；每kernel计算更长、每copy16 KiB，预计提供更大的计算窗口。这个测试同时改变N/chunk/rounds，只能用来检测“该工作点是否有重叠”，不能当作单因素加速对照旧N；同一新工作点的两种order仍只改提交顺序。若交集仍0，应如实报告当前环境/提交方式下未观察到copy-compute，并保留可能的host/driver排队、引擎调度限制，不强行继续调整直到得到有利数据。

解析脚本的phased支持首次运行失败：循环内stage变量覆盖了op的phase标签，导致字符串筛选报AttributeError。原始[错误日志](../../results/20261023T165105Z_stage6-phased-analysis_xy9zpum7/002_python3.log)保留；修正变量名后对原始两份trace及新phased trace全部重算，[重检通过](../../results/20261023T165124Z_stage6-timeline-recheck_l6wqi0he/status.txt)。没有忽略解析错误引用未生成结果，也没有改写原始trace。

## 第2轮实测：提交顺序有收益，仍不能宣称copy-compute重叠

新增phased后，14个pipeline CTest（两种order各7组）通过，四类Sanitizer分别检查两种order；仍逐输出验证。原interleaved版本保留，final binary的两种路径共用相同stage函数，确保A/B/B/A对照是同一实现中的提交次序差异。

[phased全variant首轮](../../results/20261023T164849Z_run_1o6hyflr/samples.csv)的Event Median分别为pageable_serial8854.118、pinned_serial3855.565、pinned_pipeline3424.752、pageable_pipeline8282.419 us（完整精度以CSV为准）。相比第一轮有明显跨进程漂移，因此不能直接拿3424与旧1886比较order优劣。

接着固定N2^20、chunk2^16、rounds128、streams3、buffers3、pinned_pipeline、3/11/10，仅改变order，按A/B/B/A运行：

| 顺序 | order | Event Min / Median / P95 us | host E2E Median us | CSV |
| --- | --- | --- | ---: | --- |
| A1 | interleaved | 5991.526 / 8118.391 / 9802.342 | 9967.028 | [A1](../../results/20261023T165104Z_run_l1ph_0te/samples.csv) |
| B1 | phased | 3121.152 / 3990.118 / 5961.831 | 4553.035 | [B1](../../results/20261023T165109Z_run_6xgorfk9/samples.csv) |
| B2 | phased | 3208.192 / 4255.027 / 5530.931 | 4201.618 | [B2](../../results/20261023T165112Z_run_j1tsp1oa/samples.csv) |
| A2 | interleaved | 5592.781 / 8580.013 / 10837.709 | 9318.240 | [A2](../../results/20261023T165115Z_run_aenjjs7s/samples.csv) |

这组数据支持在该运行区间phased更快，两个B中位数都小于两个A；波动仍大，不报普适固定2倍。其[NSYS](../../results/20261023T164856Z_nsys__q616dzj/timeline.nsys-rep)与[重检解析](../../results/20261023T165124Z_stage6-timeline-recheck_l6wqi0he/phased.json)只有80 kernels、160 copies=5ops×16chunks，正式batch的copy∩compute仍0、kernel并发≥2约17.968 us；活动包围约5783.166 us、Event约6422.896 us。速度变化因此不能命名为copy-compute overlap收益；它与分组提交、排队/间隙以及少量kernel并发同时相关，内部driver排队未直接测量。

环境补充：当前Linux内核为`6.18.40.1-microsoft-standard-WSL2`。它限定本报告的平台范围；没有原生Linux对照，也没有证据证明“WSL必然禁止重叠”。不能用平台标签替代本次测量。实际引擎映射、系统调度和驱动提交策略尚未独立归因。

### 参数扫描：分开stream、buffer、chunk

以下固定phased、pinned_pipeline、N2^20、rounds128、B256、3/11/10。除表中变量，其他采用streams3、buffers3、chunk65536。参考点采用上表B1/B2区间3990–4255 us，不挑选一次最小值。表中Min/Median/P95来自各自完整11个raw样本。

| 改变量 | Event Min / Median / P95 us | host E2E Median us | CSV |
| --- | --- | ---: | --- |
| streams1 | 3476.889 / 4267.213 / 5274.522 | 5161.812 | [S1](../../results/20261023T165118Z_run_jizcwwy1/samples.csv) |
| streams2 | 2365.235 / 3689.779 / 5802.086 | 4030.822 | [S2](../../results/20261023T165121Z_run_ge1iezts/samples.csv) |
| streams4 | 3176.141 / 3667.027 / 7338.291 | 4612.923 | [S4](../../results/20261023T165124Z_run_1c8dm18o/samples.csv) |
| buffers1 | 8507.597 / 11205.530 / 15336.983 | 12592.625 | [B1](../../results/20261023T165128Z_run_tdqse2x3/samples.csv) |
| buffers2 | 3716.505 / 4205.671 / 6608.598 | 4210.642 | [B2](../../results/20261023T165132Z_run_9m1hhhq4/samples.csv) |
| chunk262144 | 2126.742 / 2474.189 / 2924.339 | 2791.780 | [大chunk](../../results/20261023T165136Z_run_sver5s29/samples.csv) |
| chunk16384 | 9394.893 / 11599.360 / 13265.511 | 10708.561 | [小chunk](../../results/20261023T165139Z_run_bn5h4rlx/samples.csv) |

stream1也能执行phased，只是在一条stream串行执行；多stream本次中位数略低但分布重叠且有跨进程漂移，不宣称最优stream数是4。buffers1使每次复用等待前一D2H且每组只有一个chunk，phased退化为逐chunk提交；streams再多也不能并行使用同一slot。buffers2/3则允许不同slot工作，其分配分别1/1.5 MiB；对phased而言buffer数还决定分组宽度，不能把差异只归因于容量。

chunk16384/65536/262144对应64/16/4 kernels和双倍copy次数，device ring分别0.375/1.5/6 MiB；单kernel grid分别64/256/1024 blocks。总host字节和FMA总量保持相同，但launch/事件次数、每个kernel大小、buffer分配与组数同时变化，这是chunk参数本身的派生影响。较粗chunk显著减少本次整流水耗时，支持细分提交成本重要；尚无计数器证明唯一瓶颈，也未做全范围调参寻找全局最优。继续扩大到单chunk会失去跨chunk重叠机会，不能从三点拟合“越大越好”。

## 第3轮：延长计算后出现少量copy-compute交集

新工作点N65536、chunk4096、rounds4096，streams3、buffers3、pinned_pipeline、3/11/10。两种order仍各16个kernel/op、H/D各256 KiB、总536,870,912 FLOP；每kernel16 blocks，比第1轮256 blocks更小，且每线程依赖递推更长。不同工作点之间不做单因素速度比较。

| order | Event Min / Median / P95 us | host E2E Median us | 普通CSV |
| --- | --- | ---: | --- |
| interleaved | 7705.293 / 9062.893 / 11226.112 | 10928.158 | [I](../../results/20261023T165218Z_run_mu4pjhaj/samples.csv) |
| phased | 1659.392 / 2517.709 / 5175.296 | 2699.983 | [P](../../results/20261023T165303Z_run_br8dbr0h/samples.csv) |

phased本次普通时间较低，分布仍宽。两个独立NSYS运行各包含80 kernels/160 copies，对应5个完整op；[interleaved trace](../../results/20261023T165232Z_nsys_fk1p8zvx/timeline.nsys-rep)、[phased trace](../../results/20261023T165307Z_nsys_gt7nwqge/timeline.nsys-rep)以及[解析记录](../../results/20261023T165348Z_stage6-long-compute-analysis_2pqd3h9w/status.txt)均保留。下表为两个正式batch op的Median，us：

| order | kernel活跃和 | kernel并集 | kernel并发≥2 | copy并集 | copy∩compute | 活动包围span |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| interleaved | 3551.539 | 3551.539 | 0 | 139.678 | 0 | 16666.368 |
| phased | 3105.456 | 1323.186 | 984.854 | 143.501 | 5.952 | 6398.846 |

[逐op JSON](../../results/20261023T165348Z_stage6-long-compute-analysis_2pqd3h9w/phased.json)的全5个op copy-compute交集总计39.680 us，正式batch每op中位数5.952 us；这是从device区间计算得到的正交集，说明该工作点确实发生过copy-compute重叠。其正式交集/copy并集约4.15%，且远小于约984.854 us的kernel-kernel并发时间。不能把9063→2518 us的大幅变化全解释为copy被隐藏；并发计算、提交间隙、跨进程/插桩状态差异都有贡献，实际kernel活跃和也不完全相同。

因此本轮修正后的结论是：多stream与pinned只提供条件；提交顺序和工作点影响实际重叠。当前短计算配置没有copy-compute交集；长计算phased有少量交集及大量kernel并发。这个结论由可核对的trace支撑，比“phased一定让传输与计算充分重叠”更窄，也更符合观测。计时收益需按工作点单独评估。

后续可在受控系统负载和原生Linux/另一copy引擎配置复现相同二进制语义，或采集更长的重复trace、分析driver排队，以区分平台/调度因素；这些尚未执行。当前不修改系统配置，也不继续筛选数据来追求理想流水线图。

## 验证、边界与阶段验收

| 检查 | 本阶段结果与证据 |
| --- | --- |
| 双向copy 10个GPU CTest及CLI/CSV | [PASS](../../results/20261023T164104Z_verify_t0bnaww3/verification.txt) |
| pipeline两种order共14个GPU CTest及CLI/CSV | [PASS](../../results/20261023T164747Z_verify_31jh6b5l/verification.txt) |
| concurrent 10个GPU CTest及CLI/CSV | [PASS](../../results/20261023T164128Z_verify__jsv9z__/verification.txt) |
| copy四工具×双向，共8次Sanitizer | [PASS](../../results/20261023T164111Z_sanitizer_kdtqh966/status.txt) |
| pipeline四工具×两种order，共8次Sanitizer | [PASS](../../results/20261023T164835Z_sanitizer_rs3gp3kc/status.txt) |
| concurrent四工具，共4次Sanitizer | [PASS](../../results/20261023T164134Z_sanitizer_4_hup0no/status.txt) |
| 计算kernel PTX/SASS，最终pipeline仍有循环FMA | [pipeline静态记录](../../results/20261023T164834Z_static_ttt1y9zn/status.txt)，[concurrent静态记录](../../results/20261023T164133Z_static_u1r791ye/status.txt) |
| 14个已实现程序统一构建、15个host/help检查 | [PASS](../../results/20261023T165321Z_build_s84u8nfs/status.txt) |
| Nsight Compute | [Not Tested：ncu不可用](../../results/20261023T165431Z_ncu_x4z_0y4l/status.txt) |

GPU测试覆盖NaN重置后全量输出、双向精确copy、非整chunk/job、小N、单buffer、多stream/buffer不等，以及phased跨组复用。Sanitizer无错误只支持本次检查范围；stream依赖和host生命周期由指南的协议证明约束，不以一次无错代替。工具插桩结果均不用于普通性能表。

其他SM7x/8x、Toolkit13.1构建、不同driver/原生Linux平台尚未复现；当前仅确认SM89原生cubin路径。没有测试pin/unpin成本、CPU工作与GPU的应用级流水、NCU动态计数器或可选cp.async；这些不假报通过。本阶段必做的传输、stream pipeline、独立kernel并发及分析已经完成，实际观测中的失败条件和后续问题一并保留。

数据收尾：[独立审计](../../results/20261023T165703Z_stage6-data-audit_4uvtp9on/status.txt)对21份普通CSV、33组variant逐一复算raw数量、Min/Median/P95、GB/s/GFLOP/s、pipeline字节/分配/worker模型，并校对采集时源码和二进制哈希；[清单](../../results/20261023T165703Z_stage6-data-audit_4uvtp9on/samples_manifest.json)保存每份CSV哈希及完整中位数元数据。时间线并集算法另用可手算重叠区间例验证。长计算phased两个正式op的copy交集分别3.552/8.352 us，5.952是二者中位数，不表示固定每次重叠量。
