# Stage 7：FP32 GEMM逐步优化记录

## 第0轮：实现前模型与判定条件（2026-10-24，UTC+8）

本阶段接续已验收S1–S6。矩阵统一row-major、FP32输入/输出，`C=A*B`，alpha=1、beta=0，无C输入、不转置；使用传统CUDA Core的fmaf，不使用Tensor Core/cuBLAS/CUTLASS。保留Naive→Coalesced→Shared Tiled→Register Tiled→Unrolled五版。先实现和验证前两版，再增加分块版本，避免把未验证的大段代码当交付。

### 模型与控制变量

A[M,K]、B[K,N]、C[M,N]；有用算术按2MNK FLOP，包含每项FMA的加法，即首项加到0也计入。唯一逻辑工作集4(MK+KN+MN) B；这只是一次读取输入/写输出的数学数据量，不代表GPU实际DRAM字节。

Naive/Coalesced每线程一个输出，按k递增累加；源级global requested=`4(2MNK+MN)`。Naive让threadIdx.x沿输出row变化，Coalesced让x沿column变化；相同T×T线程块、相同输出覆盖和launch数（二维grid轴交换），只改变线程映射。T32的完整warp在某个k上，Naive A跨行stride K、B广播，C跨行stride N；Coalesced A广播、B/C连续。T16的warp跨两行/列，不能直接套用32列一warp；用地址集合另算首warp sector。

Shared Tiled的输出tile BM=BN=T、BK=T；每块每k-tile加载A/B各T²个值，内循环从shared读。考虑尾部有效元素，实际源级global load计数为`MK*ceil(N/BN)+KN*ceil(M/BM)`，再加MN个store。零填充仅保护边界，所有线程参加两次barrier，不能在barrier前按row/col提前return。

Register Tiled保持T×T线程、BN=BK=T，每线程计算R个竖直输出，BM=R*T；thread(ty,tx)负责`row=block_y*BM+ty+r*T, col=block_x*T+tx`。每轮共享一个B值更新R个累加器，减少重复B读取并增加独立累加链；A tile扩大R倍、输出tile扩大R倍、grid block数下降，是该策略必须记录的派生变化。Unrolled在相同register tile上展开BK循环，其余布局/输入/算术顺序保持相同。R循环在两版中都展开以避免动态可寻址acc数组。

源级global字节与AI使用上述公式，各variant不同，不直接拿其GB/s做跨版本性能排序；主指标为相同2MNK对应的GFLOP/s和耗时。shared版本会执行padding后的FMA：源级模型`ceil(M/BM)*ceil(N/BN)*BM*BN*ceil(K/T)*T`，与有用MNK分开报告；它不是动态硬件指令计数。

### 数值与验证计划

CPU对全部输出用FP64乘加参考，并计算每项绝对积之和S_ij。输入float先提升double再相乘；FP32两个有限乘数的精确乘积可在FP64中表示，参考求和仍有FP64舍入。GPU显式fmaf按k递增，shared尾部加0；不启用fast math。以u=2^-24、gamma_K=K*u/(1-K*u)构造前向误差界，使用`2*gamma_K*S_ij+1e-7`作本实验保守容差，并报告最大绝对误差和相对容差的比例；接近0的取消求和不能只用相对C误差判错。

覆盖random、identity、cancellation输入；小N、矩形、非T/R倍数、K尾部、T16/32和R2/4。输出先填NaN，逐元素全量验证；M/N/K及CPU参考总工作和host/device容量都有上限。CPU参考、初始化、分配不计时。Event输入驻留；host E2E另测A/B上传+同一GEMM+C回读+sync。每轮计时后验证，通过后才发布CSV。

### 可否定的预测

1. Naive→Coalesced可能降低warp地址覆盖的sector数并提高大矩阵性能；若实际SASS已改变访存形式或小矩阵提交成本主导，不预设速度比等于sector比。NCU缺失时只报告地址模型与有限计时支持。
2. Coalesced→Shared减少global源级重复请求，但增加shared/barrier和边界开销；小矩阵可能更慢。静态必须看见LDS/STS/BAR及FMA，源级请求减少不直接证明DRAM流量下降。
3. Shared→Register提高复用和每线程独立工作，也增加register/shared并减少grid；若理论驻留减少或尾部浪费增大，可能回退。资源API、ptxas及SASS共同检查，不把occupancy高低作为唯一性能指标。
4. Register→Unrolled可能减少循环控制并改变调度，但代码体积/寄存器可能增长；只在相同tile/R下比较，做顺序交换复测，不预设所有尺寸都更快。

先在小边界建立正确性，再采方形256/512/1024、矩形及非整tile工作点，选代表规模查SASS/NSYS；参数扫描明确派生工作量与资源。实际采样数量根据已有证据选择，所有反例、失败与未测指标保留，后续文字不冒充本段采样前预测。

## 第1轮：Naive与Coalesced基线已验证，之后再增加分块

[22个GPU CTest与CLI/CSV检查](../../results/20261023T170348Z_verify_dzc_mgsd/verification.txt)通过，覆盖T16/32、小/矩形/尾部、K4096，以及identity/cancellation。这里只包含前两版，不将其当作后续共享内存同步已通过。实际[PTX/SASS与资源](../../results/20261023T170433Z_static_3w1k2__x/status.txt)已采集。

T16、3/7/5、同random输入：256³的Naive/Coalesced Event Median约691.405/241.050 us（[CSV](../../results/20261023T170435Z_run_vfh76uhm/samples.csv)）；512³约4952.269/1282.458 us（[CSV](../../results/20261023T170437Z_run_vjshm17s/samples.csv)）。当前线程映射改变后时间下降，方向符合地址模型；但Naive的B同址、Coalesced的A同址，加上cache和每次k访问变化，使sector数比不能直接变成时间比。尚无NCU动态sector/cache指标，不能声称测得了指定DRAM流量下降。

下一步按第0轮设计加入shared、register和unrolled。以512³/T16为例，direct源级global请求约1.075 GB；Shared的有效global加载为`512²*32+512²*32`个float，含输出约68.157 MB；R4将M方向tile数从32降至8，模型含输出约42.992 MB。这是源级请求复用模型，DRAM可能已被cache过滤。若shared时间没有按请求比例下降，应检查shared/barrier、依赖链、资源和计时范围，而不是修改模型去拟合时间。

## 第2轮：五版正确性与编译结果

新增三版之后，[44个GPU CTest及CLI/CSV](../../results/20261023T170753Z_verify_3gj2g1ou/verification.txt)全部通过；两种tile、两种R以及random/identity/cancellation、K尾部均覆盖。[8次Sanitizer](../../results/20261023T170841Z_sanitizer_b6j9yqvv/status.txt)为四工具×T16/R4 random与T32/R2 cancellation两配置，每次都跑五variant；没有共享内存hazard或同步告警。结论限定为本次覆盖，算法的完整依赖见指南。

[最终SASS/资源](../../results/20261023T170839Z_static_5seajc5p/gemm.sass.txt)和[按symbol解析](../../results/20261023T171053Z_stage7-sass-analysis_jokg32yc/sass_summary.json)提供14个实例的证据，均为FFMA，未发现MMA/Tensor Core指令，所有实例STACK/LOCAL为0。下表列出关键资源；代码字节为cuobjdump列出的PC范围，含padding，非执行指令数。

| 实例 | registers | shared B | 列出代码B | 真正的回跳 |
| --- | ---: | ---: | ---: | --- |
| direct T16/32，naive/coalesced | 16 | 0 | 768 | K循环1个 |
| shared T16 | 21 | 2048 | 1152 | BK内循环与K-tile外循环 |
| shared T32 | 21 | 8192 | 1152 | 同上 |
| register T16/R4 | 37 | 5120 | 2048 | 内/外2个 |
| unrolled T16/R4 | 40 | 5120 | 3328 | 只剩K-tile外循环 |
| register T32/R4 | 37 | 20480 | 2048 | 内/外2个 |
| unrolled T32/R4 | 40 | 20480 | 4864 | 只剩K-tile外循环 |
| register T16/32、R2 | 26 | 3072 / 12288 | 1408 | 内/外2个 |
| unrolled T16/R2、T32/R2 | 39 / 40 | 3072 / 12288 | 2304 / 3072 | 只剩外循环 |

这一步修正了对unroll作用的简单描述：T16/R4未展开的内BK循环有14个列出指令，其中4 FFMA、5个标量LDS，每k执行一次；展开版有64条FFMA、16个标量LDS及16个LDS.128，去掉内层回跳。LDS.128每次读4个float，所以列出load数减少不意味着少读了相同语义需要的数据；编译器还改变了shared读取宽度/调度。性能差异不能只归因于“少了几条BRA”。T32/R4展开同样使用32个标量LDS+32个LDS.128，128 FFMA。

每块仍有两条静态BAR.SYNC对应每个K-tile装载后/计算后屏障；静态2不意味着整个kernel只同步2次。global的LDG/STG和shared的LDS/STS要分别统计。source层的bank/复用模型需要按最终宽访存核对，NCU未测bank冲突或实际shared吞吐。

### 下一轮复测预先约定

初始五版扫描已观察到两类不确定性：同512³ direct配置跨运行明显漂移；矩形128×1024×256中unrolled的普通中位数85.811 us略高于register的81.510 us，不能由一次小差异断言稳定回退。将对512³和该矩形分别做register(A)/unrolled(B)/B/A，固定T16/R4、输入和编译，统一warmup5/repeats11/K20。若B在顺序交换后仍整体较低，只支持该工作点的收益方向；若分布/中位数交错，撤回固定排序。初始数据保留，不删除不合预期的矩形样本。

已经安排512³和32³的NSYS补测，核对每variant的5个kernel（初验1、warmup1、正式2、E2E1）及二维grid/block；短工作点不只看Event表来归因kernel本身。

## 第3轮：普通性能与参数扫描

环境仍为RTX4060 Laptop / SM89 / Toolkit13.4.92 / driver596.21 / WSL2，未锁频。以下3/7/5、random输入，单位us；这是独立运行的Event span/op Median，不是单kernel活跃时间。前六组T16/R4，最后三组只扫描T/R，完整Min/P95及所有raw保留于CSV。

| M×N×K，T/R | Naive | Coalesced | Shared | Register | Unrolled | CSV |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 32×32×32，16/4 | 80.077 | 72.499 | 26.214 | 19.251 | 18.842 | [数据](../../results/20261023T171042Z_run_bm58658x/samples.csv) |
| 256×256×256，16/4 | 833.741 | 270.541 | 244.941 | 158.925 | 106.086 | [数据](../../results/20261023T171044Z_run__41t0sz2/samples.csv) |
| 512×512×512，16/4 | 1969.971 | 593.510 | 417.382 | 279.962 | 115.302 | [数据](../../results/20261023T171046Z_run_t_m3lr8s/samples.csv) |
| 1024×1024×1024，16/4 | 9825.652 | 3494.912 | 3244.646 | 2134.016 | 1072.742 | [数据](../../results/20261023T171048Z_run_r4sh61o6/samples.csv) |
| 257×509×127，16/4 | 110.387 | 111.411 | 103.424 | 90.931 | 55.501 | [数据](../../results/20261023T171053Z_run_ncgd9nut/samples.csv) |
| 128×1024×256，16/4 | 408.986 | 149.299 | 114.278 | 81.510 | 85.811 | [数据](../../results/20261023T171055Z_run_ih97es1b/samples.csv) |
| 512×512×512，16/2 | 1445.069 | 467.354 | 384.000 | 278.733 | 195.584 | [数据](../../results/20261023T171057Z_run_mp0klm2o/samples.csv) |
| 512×512×512，32/2 | 3406.029 | 909.312 | 397.926 | 321.946 | 313.344 | [数据](../../results/20261023T171059Z_run_9vlgre7e/samples.csv) |
| 512×512×512，32/4 | 2952.397 | 684.442 | 484.557 | 245.146 | 188.416 | [数据](../../results/20261023T171102Z_run_si9u5x8y/samples.csv) |

512³/T16/R4同一次运行的详细统计：

| variant | Event Min / Median / P95 us | 有用GFLOP/s@Median | host E2E Median us |
| --- | --- | ---: | ---: |
| naive | 1680.384 / 1969.971 / 2052.096 | 136.264 | 3611.083 |
| coalesced | 537.395 / 593.510 / 652.288 | 452.284 | 2465.962 |
| shared_tiled | 371.686 / 417.382 / 517.325 | 643.140 | 2035.406 |
| register_tiled | 262.349 / 279.962 / 303.104 | 958.830 | 2069.290 |
| unrolled | 111.411 / 115.302 / 123.904 | 2328.100 | 1323.402 |

### 按优化步骤解释，保留边界

Naive→Coalesced：方形256/512/1024的数据支持映射改善方向，但257×509×127的中位数约110.387/111.411 us，排序没有改善。该形状改变行对齐、K长度、缓存复用与系统状态，不能简单把“非整tile”视为唯一原因。模型只承诺改变地址集合，不能保证所有工作点更快。不同运行中512³ Naive约4952→1970→1445 us，说明跨进程漂移明显；不要将这些数据串成逐次代码优化的加速曲线。

Coalesced→Shared：512³/T16源级global请求约减少15.77倍，但同一普通运行的Event仅593.510→417.382 us。这个差异推翻了“所有source请求都以相同带宽直达DRAM并决定总时间”的简单模型。cache已经复用、shared/barrier、依赖链和执行效率都是竞争解释；没有NCU不能定量分配各自比例。当前约3 MiB的唯一工作集小于设备32 MiB L2，只是容量条件，不是100% L2命中的证据。

Shared→Register：512³/T16/R4的global请求从68.157降到42.992 MB，输出tile由16×16增至64×16，grid块数1024→256，register21→37、shared2048→5120 B；Event417.382→279.962 us。复用和每线程独立累加链同时增加，不能把全部收益单独归为ILP。T16下Runtime理论驻留都是6 blocks/SM，资源增加尚未降低该API上界；它不证明运行时occupancy或stall相同。

Register→Unrolled：同一512³运行279.962→115.302 us，register37→40、代码2048→3328 B，理论驻留仍6。SASS支持内层控制消除和宽shared读变化，不能给每条指令分配未测周期。矩形初始数据的85.811>81.510 us则说明不能普遍套用该收益；后续ABBA用于进一步区分。

T/R扫描：T32每block1024线程、理论驻留1 block/SM；T16为256线程、理论6。更大的tile减少global源级请求但改变资源/grid，T32/R4在512³本次unrolled188.416 us高于T16/R4的115.302 us，不过跨进程漂移使此差异只能作为当前样本观察。R2→R4增加复用和寄存器，而T16 register时间278.733/279.962 us几乎相同；不以“R更大”自动宣称收益。最终挑选参数需在目标形状同一环境复测，本文不宣称全局最优配置。

尾部257×509×127：有用FMA=16,613,251，T16 shared padding模型17,825,792，R4模型20,971,520；R4在M方向覆盖320行而只有257有效。shared和register都不能把有用GFLOP/s当成真实已执行FP32吞吐；冗余计算要单独报告。所有输出仍只写有效位置，NaN与全量参考验证尾部完整性。

## 第4轮：NSYS修正小矩阵结论

[512³ trace](../../results/20261023T171106Z_nsys_qevfxuc6/timeline.nsys-rep)、[32³ trace](../../results/20261023T171138Z_nsys_wu9vj48f/timeline.nsys-rep)各25个kernel，五variant各5个，与`1+1*(1+2+1)`一致。[二维grid/block及时间戳解析](../../results/20261023T171800Z_stage7-timeline-analysis_l9fgy1nm/status.txt)逐case核对，正式batch仅取两次kernel，不把初验/warmup/E2E混入正式均值。下面单位us，kernel active为正式batch均值；Event是同次插桩CSV，不能替换普通表。

| variant | 512³ active / Event | 32³ active / Event |
| --- | --- | --- |
| naive | 3854.749 / 3929.232 | 12.240 / 48.496 |
| coalesced | 1226.020 / 1300.464 | 11.856 / 77.104 |
| shared_tiled | 1024.966 / 1085.888 | 5.008 / 40.704 |
| register_tiled | 657.210 / 705.280 | 6.576 / 10.624 |
| unrolled | 323.037 / 328.688 | 4.736 / 15.616 |

32³给出明确反例：shared→register的kernel活跃时间变长，而Event表反而变短。R4的BM64大于M32，grid只有2块（shared4块），源级padding FMA翻倍；更小grid和无效输出计算可以抵消复用收益，两个因素同时变化，不能仅归为寄存器压力。register→unrolled的活跃时间变短，Event却变长，进一步说明短任务启动/调度间隙会改变排序。

512³插桩active时间多数接近Event包围范围，优化方向与普通本次五版一致，但绝对时间显著不同，不能给普通115.302 us套用插桩323.037 us的成本分解。unrolled所有5个kernel中的P95为1345.044 us，正式两次均值323.037 us；这种非正式阶段异常也保留于[逐case JSON](../../results/20261023T171800Z_stage7-timeline-analysis_l9fgy1nm/large.json)，不删掉以美化分布。

## 第5轮：A/B/B/A复测，区分收益方向与环境漂移

按照第2轮预先约定，A=register_tiled、B=unrolled，固定T16/R4、random输入、warmup5/repeats11/K20，八次运行使用同一二进制哈希。每次进程独立，表中完整保留全部四次，而不挑选较好的B。单位us；Event统计为批量span/op，E2E为另一组独立端到端样本的Median。

| M×N×K | 顺序 / variant | Event Min / Median / P95 | E2E Median | 原始CSV |
| --- | --- | ---: | ---: | --- |
| 512×512×512 | A1 / register_tiled | 1105.664 / 1343.334 / 1708.186 | 3082.735 | [数据](../../results/20261023T171746Z_run_460950p7/samples.csv) |
| 512×512×512 | B1 / unrolled | 115.712 / 116.224 / 127.334 | 1251.015 | [数据](../../results/20261023T171758Z_run_g7c6urww/samples.csv) |
| 512×512×512 | B2 / unrolled | 125.952 / 192.768 / 218.573 | 3464.901 | [数据](../../results/20261023T171842Z_run_6o2f93es/samples.csv) |
| 512×512×512 | A2 / register_tiled | 963.072 / 1059.379 / 1304.883 | 3976.021 | [数据](../../results/20261023T171854Z_run_ns0_etyu/samples.csv) |
| 128×1024×256 | A1 / register_tiled | 751.155 / 1240.781 / 1695.488 | 11892.594 | [数据](../../results/20261023T171915Z_run_5m6kqy96/samples.csv) |
| 128×1024×256 | B1 / unrolled | 340.736 / 484.045 / 1204.224 | 6437.510 | [数据](../../results/20261023T172056Z_run_c28kw43v/samples.csv) |
| 128×1024×256 | B2 / unrolled | 34.202 / 45.414 / 51.098 | 893.036 | [数据](../../results/20261023T172342Z_run_n993d2hv/samples.csv) |
| 128×1024×256 | A2 / register_tiled | 67.482 / 96.819 / 112.077 | 990.835 | [数据](../../results/20261023T172522Z_run_hv5jg79l/samples.csv) |

512³中两个B的Median都低于两个A，支持本工作点展开后的收益方向；但A1/A2、B1/B2本身已有明显漂移，且A的绝对时间远高于第3轮279.962 us。不能把这些独立进程拼成一个固定加速比，也不能把差异全归因于SASS中减少的循环控制。静态证据还显示shared访问宽度/调度和寄存器变化，动态stall、发射与cache指标仍缺失。

矩形反例更强：A1/A2的Median相差约12.8倍，B1/B2约10.7倍，B1的484.045 us又高于A2的96.819 us；跨进程分布交错。相邻A1→B1和B2→A2的比较方向支持B，但同时存在显著环境变化，无法确定稳定收益大小，更无法把初轮约5%的回退解释成展开的固有代价。因此按事先判定条件撤回矩形的固定排序；保留初轮和这四次数据，结论是该环境未提供足够稳定的性能比较。

排查到的事实是：参数、输入、二进制及正确性相同，Event和host E2E都可出现较大变化；没有持续时钟、功率或GPU外部负载时间线，不能将原因具体定为降频、WSL调度或某个后台程序。ABBA只是顺序控制，不是消除噪声的保证。下一次若要得到可迁移的速度比，应在可控稳定环境重复相同配置、记录持续环境信号，并同时检查独立计时与相同工作点NSYS活动；不通过删样本或擅自修改系统状态获得漂亮数字。

## 第6轮：用数值诊断检验误差判据

正确性测试已包含随机、单位矩阵和相消输入；另外保存两个可直接检查误差字段的普通运行，warmup1/repeats2/K2仅用于数值诊断，不用于性能排序。

| 形状与输入，五版共同结果 | 最大绝对误差 | 最大误差/容差 | 最大相对误差（分母下限1e-12） | 原始CSV |
| --- | ---: | ---: | ---: | --- |
| 37×65×32，cancellation | 1.34844853505e-7 | 0.00309143957945 | 134844.853505 | [相消](../../results/20261023T174011Z_run_tm_yp5co/samples.csv) |
| 37×65×33，identity | 0 | 0 | 0 | [单位矩阵](../../results/20261023T174013Z_run_zs4vw81k/samples.csv) |

相消输入让相邻k的精确乘积互相抵消，FP64参考为0，但FP32 FMA每一步舍入会留下很小残差。此时按C的相对误差会巨大，不能单独作为失败依据；绝对误差仍远低于根据绝对积和推导的界。单位矩阵验证了另一类语义，随机及矩形/尾部测试继续承担索引覆盖，不能用易相消输入代替完整正确性。五版都按相同k顺序FMA，本次误差统计相同并不保证任意重排求和的实现也逐位相同。

## 数据复核、阶段验收与适用边界

[独立数据审计](../../results/20261023T174711Z_stage7-data-audit_0aguvimk/status.txt)检查21份普通CSV、67组variant：2份初版两variant、9份五variant扫描、8份单variant ABBA、2份五variant数值诊断。由raw重新计算三类时间的Min/Median/P95和速率，独立复算矩阵容量、global请求、FMA/padding、首warp sector、shared及AI，并核对命令成功、原始源码与二进制哈希；[清单](../../results/20261023T174711Z_stage7-data-audit_0aguvimk/manifest.json)保留每份CSV摘要与哈希。NSYS两份插桩CSV单独用于时间线分析，没有混入上述普通基线。

审计初次因旧两版CSV没有后来新增的`reg_rows_requested` / `k_tile_unroll_factor`字段而失败，见[原始失败](../../results/20261023T174645Z_stage7-data-audit_4mw88uw5/status.txt)。根据旧源码确认当时只有direct两版，再按明确的历史schema核对；没有回填或改写旧CSV，也没有放松当前五版字段要求。二维grid/block解析扩展后，对S5一维branch trace的[回归复核](../../results/20261023T174602Z_final-source-evidence-audit_27b0s0_j/status.txt)也通过。

| 验收项 | 结果及证据 |
| --- | --- |
| 五版、两种T/R、随机/单位/相消、小/矩形/尾部、非法输入 | [44个GPU CTest及CLI/CSV通过](../../results/20261023T170753Z_verify_3gj2g1ou/verification.txt) |
| memcheck/initcheck/racecheck/synccheck | [8次运行通过](../../results/20261023T170841Z_sanitizer_b6j9yqvv/status.txt)，每次五版；不是全部输入的证明 |
| PTX/SASS、FFMA/shared/barrier、资源、无Tensor替代 | [静态通过](../../results/20261023T170839Z_static_5seajc5p/status.txt)与上文14实例解析 |
| 独立性能、模型、误差及元数据 | 21 CSV / 67组审计通过；保留漂移，不承诺固定倍数 |
| NSYS正式batch、二维grid/block、单op单kernel | [50个kernel时间戳核对通过](../../results/20261023T171800Z_stage7-timeline-analysis_l9fgy1nm/status.txt) |
| 全项目集成与无CUDA工具路径 | [15程序构建及16个host/help通过](../../results/20261023T174000Z_build_zlhfx6wo/status.txt)，[host-only通过](../../results/20261023T174006Z_host-test_f_l2ed03/status.txt) |
| 当前源代码与GPU验证快照一致 | [15程序及本地传递依赖核对通过](../../results/20261023T174602Z_final-source-evidence-audit_27b0s0_j/source_evidence.json) |
| NCU动态计数器 | [Not Tested：工具不可用](../../results/20261023T174015Z_ncu_havqc_b_/status.txt) |

一次集成构建在cmake build成功后因输出管道`Broken pipe`中断，留下[失败记录](../../results/20261023T172526Z_build_7_4e0r_4/status.txt)；上述174000Z完整重跑才作为集成通过证据。没有将局部编译成功等同于整个动作完成。

**Stage 7基础验收完成，限定本次原生SM89环境。** 完整优化链已建立：地址映射假设→复用模型→资源/指令核对→普通计时→NSYS反例→ABBA降低结论强度→数值与证据审计。该链说明怎样验证和修正判断，不证明所有优化在所有规模上更快。DRAM字节/带宽、L1/L2命中、bank冲突、achieved occupancy、issue/stall均未测；静态requested和理论驻留不能充当动态指标。SM7x、其他SM8x及Toolkit13.1组合未实测，已记录的当前Toolkit/driver强制PTX JIT失败仍未消除。

下一步属于学习者复现或环境可用后的机制补测：先完成[综合练习](exercises.md)，用自己的环境建立相同证据链，再按当前NCU查询的指标验证上述竞争解释；可选cp.async不在本次必做五版GEMM中。项目整体交付按[逐项完成审计](../../docs/completion-audit.md)核对，个人能力是否掌握则需独立完成练习评估。
