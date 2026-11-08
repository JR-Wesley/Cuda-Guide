# Stage 5：指令与控制流实验记录

## 第0轮：采集前设计（2026-10-24，UTC+8）

Stage 4发现余数条件的SASS含倒数/商余修正，等价mask删除了该序列，但大输入的时间几乎不变。本阶段先分别隔离指令选择、循环控制和lane分布；以下在实现与Stage 5采样之前记录，不是实测结论。

### Integer Division

每线程以uint32状态执行依赖递推：计算一次商q，再混合state与q生成下一轮state，保留最终输出，避免常数除法结果反复变成0。比较同一正除数d下的runtime和编译期常量7/8；d8另有manual shift路径。支持unsigned与signed输入解释，signed转换定义清楚，除法向0截断；只允许正除数，避免INT_MIN/−1溢出和除零。CPU逐轮整数reference必须位级一致，输入含0、1、边界和负数位模式。

模型：逻辑division=N*rounds；global requested=8N。uint32混合允许模2^32回绕，signed路径仅执行不溢出的除法，之后回到unsigned混合。预期runtime除法比常量需要更多指令，unsigned/8可化为shift，signed/8还需负数向0截断处理；错误地直接对负数算术右移不是合法优化。常量/runtime在同d下才构成等价对照，不把d7与d8的不同递推结果当同输入数学运算。

静态判定：PTX有div不保证最终SASS有一条divide；检查虚拟操作、SASS序列、register/code size，循环中的工作不能被整个消去。全设备批量吞吐不能直接读成单条除法延迟，依赖链下界与吞吐下界分别写为符号模型，不编造周期数。

### Loop Unrolling

固定uint32递推`x=(x*a+b); x^=x>>13`，相同rounds与独立chain数，对比pragma unroll1/4/8；每个chain输出都可观察，CPU精确比较。因xor存在，不假设编译器会把整个递推合并成闭式表达式。检查非4/8整倍数的rounds，以验证编译器生成的尾部。

每op有效update=N*chains*rounds，global requested=4N*(1+chains)。固定chains下仅改unroll factor；chains1/4比较要明确总工作量、线程数和依赖结构变化，不把4倍工作解释成纯unroll收益。预期减少回跳或增加调度空间，也可能增长代码/register；若最终SASS已相同，就不能把源码pragma名字当作不同硬件方案。若无性能收益，也保留结果。

### Branch Layout

同一kernel与同一逻辑任务集合，改变输入排列：uniform每32个任务同一路径，interleaved在64任务组内交错两路。尾部只对现有任务重排，数量/分支比例不变；CPU先算逻辑任务reference再按排列检查输出。两路使用不同的uint32变换，各轮有相同数量的源级核心更新，但不假定机器指令吞吐完全相同。

提供短body（一次if）与loop body（每路径多轮递推）作为两个单独工作量场景；只在同body/rounds内比较layout。预期短body可能被predication/select代替，长body可能保留分支；只有SASS确认后才称为对应形式。warp内交错的长分支可能让两路串行，时间不能保证恰好2倍；同kernel、同资源与同输入工作量减少其他混杂，仍需考虑尾部、调度和频率。

判定：若编译器合并了两条路径，重新分析最终实现，不能凭源码if宣称divergence已测。真实branch efficiency、active lanes、stall依赖NCU；缺失时仅报告地址/flag模型、静态形式及时间观察。

## 实测状态

第0轮尚未采集结果；后续追加验证、静态发现、实测与下一轮决策。

## 第1轮静态与首轮计时后的补测设计（实施前记录）

静态检查已确认：runtime unsigned/signed的MUFU.RCP分别在PC0x170/0x180，而主循环从0x1f0/0x210开始，倒数准备不在每轮内。unsigned constant8和manual_shift的列出指令相同；signed两者略有不同。4-chain u1/u4/u8为20/17/17 registers，不能假定展开越多register越多。short body无输入flag控制的BRA，存在@P0/@!P0谓词算术；loop body保留两路循环及BSSY/BSYNC。

首轮unroll数据显示rounds131的各版本反而比rounds128更快，虽然逻辑工作多约2.3%。这提示跨进程状态或执行路径变化不能忽略；不能把两个进程直接拼成“每轮耗时”模型。补测固定N65536/chains4/rounds128，只比较u1(A)与u8(B)，按A/B/B/A运行，检查同配置漂移是否接近优化差值。此补测不改变源码或重新定义工作量。

short body首轮uniform/interleaved Event Median约6.49/10.48 us，但Min约5.57/5.38 us，分布明显重叠。因SASS没有数据分支，不能解释为分支串行。补采NSYS，按已知collect顺序区分两个layout，检查kernel活跃时长是否同样不同，还是差异主要存在于batch间隙/提交范围。profiler结果只用于这个歧义，不替换未插桩基线。

## 第2轮：环境、正确性与证据

日期2026-10-24（UTC+8）。每次重新记录RTX 4060 Laptop GPU、CC8.9、24 SM、driver596.21、nvcc13.4.92、Runtime13040；报告L2=32 MiB。C++17、Release、原生SM89、`-O3 -lineinfo --ptxas-options=-v`；确切命令、源码和二进制哈希在各运行目录。未锁频或连续监测clock，跨进程变化的具体原因仍未隔离。

普通计时固定block256、warmup5/repeats21/iterations100，pageable host、重复device数据，无profiler。时间为Event span/op，Host E2E另含上传、kernel和回读，全部统计在CSV。每次正式采样后逐字比较uint32结果，无容差；带宽列是逻辑I/O，不是硬件访存流量。下表的Gdiv/s或Gupdate/s也只是语义工作量率，不是SASS指令率。

| 验收 | 原始证据 |
| --- | --- |
| Integer Division：12组含signed/unsigned边界、d1/7/8/INT_MAX，CSV/非法参数/覆盖保护 | [PASS](../../results/20261023T161822Z_verify_k0uyvur1/verification.txt) |
| Loop Unroll：18组含展开尾部、chains1/4、N1/33/4099，CSV/CLI | [PASS](../../results/20261023T161839Z_verify_0gh124pr/verification.txt) |
| Branch：24组含warp/64任务组边缘、short/loop，重排模型/CSV/CLI | [PASS](../../results/20261023T161851Z_verify_nx9iueac/verification.txt) |
| Division八次Sanitizer：四工具×signed d8与unsigned d7 | [PASS](../../results/20261023T161832Z_sanitizer_iu2qj38r/status.txt) |
| Unroll四次Sanitizer：chains4/rounds9，全部展开因子 | [PASS](../../results/20261023T161848Z_sanitizer_npna0o7y/status.txt) |
| Branch八次Sanitizer：四工具×short/loop，N65两种排列 | [PASS](../../results/20261023T161902Z_sanitizer_fukuqnlz/status.txt) |
| 15份普通CSV、31组variant统计/单位与source/binary哈希复核 | [PASS](../../results/20261023T162345Z_stage5-audit_6c8o2uqn/status.txt) |
| 每kernel SASS静态计数、非填充回跳及列出地址范围 | [派生记录](../../results/20261023T162345Z_stage5-audit_6c8o2uqn/002_python3.log) |

20次Sanitizer日志无错误/hazard，范围限所测路径；本阶段无跨线程通信，不将racecheck/synccheck无报告泛化为任意同步算法正确。

## 第3轮：Integer Division，从语义到循环体

[静态记录](../../results/20261023T161830Z_static_6h6omwly/)的PTX有runtime `div.u32/div.s32`，SASS则是准备、乘法/移位和修正序列。以非自跳转的回边定义循环体，列出指令数包括mix和循环控制，不是“除法单独指令数”。

| 语义 / variant | registers | SASS列出字节（含填充） | 主循环区间条目数 | 静态观察 |
| --- | ---: | ---: | ---: | --- |
| unsigned runtime | 13 | 1024 | 15 | MUFU.RCP在0x170，循环从0x1f0开始 |
| signed runtime | 18 | 1152 | 21 | MUFU.RCP在0x180，循环从0x210开始 |
| unsigned/signed constant7 | 11 / 11 | 768 / 768 | 10 / 10 | 常量的乘法/移位替代序列，非runtime通用除法 |
| unsigned constant8/manual_shift | 10 / 10 | 640 / 640 | 7 / 7 | 两者列出指令完全相同 |
| signed constant8/manual_shift | 10 / 10 | 768 / 768 | 9 / 10 | 负数向0截断的处理不同，不能直接用unsigned模型 |

所有kernel static shared/local/stack均0，普通block256下理论驻留6 blocks/SM。寄存器数字不同但没有跨驻留台阶。`MUFU.RCP`位于循环外说明编译器已利用了除数的循环不变量；若把一次准备成本乘rounds，会明显高估runtime工作。源级逻辑FP32 FLOP记0不意味着最终没有浮点辅助指令，不能将这个字段当硬件浮点计数器。

| N / rounds | signed / d | variant | Median / Min / P95 (us/op) | 逻辑Gdiv/s | 证据 |
| --- | --- | --- | --- | ---: | --- |
| 65536 / 128 | 0 / 7 | runtime | 31.774 / 31.713 / 34.056 | 264.01 | [CSV](../../results/20261023T162017Z_run_9t1__l6_/samples.csv) |
| 65536 / 128 | 0 / 7 | constant | 20.439 / 20.375 / 23.777 | 410.42 | [CSV](../../results/20261023T162017Z_run_9t1__l6_/samples.csv) |
| 65536 / 128 | 0 / 8 | runtime | 27.975 / 27.905 / 32.133 | 299.86 | [CSV](../../results/20261023T162019Z_run_dr551nbt/samples.csv) |
| 65536 / 128 | 0 / 8 | constant | 14.582 / 13.789 / 15.319 | 575.28 | [CSV](../../results/20261023T162019Z_run_dr551nbt/samples.csv) |
| 65536 / 128 | 0 / 8 | manual_shift | 14.715 / 13.783 / 17.633 | 570.08 | [CSV](../../results/20261023T162019Z_run_dr551nbt/samples.csv) |
| 65536 / 128 | 1 / 7 | runtime | 35.932 / 35.901 / 38.349 | 233.46 | [CSV](../../results/20261023T162020Z_run_w18osf20/samples.csv) |
| 65536 / 128 | 1 / 7 | constant | 18.084 / 18.039 / 22.794 | 463.87 | [CSV](../../results/20261023T162020Z_run_w18osf20/samples.csv) |
| 65536 / 128 | 1 / 8 | runtime | 35.961 / 35.891 / 38.400 | 233.27 | [CSV](../../results/20261023T162021Z_run_7exltolq/samples.csv) |
| 65536 / 128 | 1 / 8 | constant | 17.939 / 17.869 / 20.531 | 467.63 | [CSV](../../results/20261023T162021Z_run_7exltolq/samples.csv) |
| 65536 / 128 | 1 / 8 | manual_shift | 18.237 / 18.196 / 20.261 | 459.97 | [CSV](../../results/20261023T162021Z_run_7exltolq/samples.csv) |
| 4096 / 16 | 0 / 7 | runtime | 7.199 / 5.527 / 8.438 | 9.10 | [CSV](../../results/20261023T162023Z_run_c7pp4vle/samples.csv) |
| 4096 / 16 | 0 / 7 | constant | 6.659 / 5.335 / 11.052 | 9.84 | [CSV](../../results/20261023T162023Z_run_c7pp4vle/samples.csv) |

**代入与反推。** 主组N65536×128=8388608次逻辑商计算。unsigned d7中runtime31.774→constant20.439 us，循环条目15→10，与减少循环工作方向一致；时间比约1.55接近但不能据此认定每条指令有相同吞吐或延迟。signed runtime有更多符号/修正处理，相对constant的差距更大；q序列也随signed解释改变，所以不能把signed/unsigned差值当纯一条符号指令的成本。

unsigned d8中constant和manual_shift列出指令相同，时间14.582/14.715 us，未出现值得宣称的手写shift收益。这支持优先保留清晰的等价常量表达式；仅“写了位运算”不是优化证据。signed manual方案正确却多一个循环条目，观测也未优于compiler constant；这是合法等价不保证最佳实现的例子。

N4096/rounds16的小组仅65536次商计算、16 blocks；runtime/constant Median7.199/6.659 us，但P95反而constant更高，不能给稳定小任务收益。这里改变N与rounds形成单独场景，不用它和主组作纯N缩放拟合。下一轮若要量化除法延迟，应另建受控单warp依赖链和时钟模型，而不是从批量Event除以N×rounds。

## 第4轮：Unroll，计算控制成本而非数pragma

[静态记录](../../results/20261023T161847Z_static_jyppf244/)表明三个factor都产生真实循环，u4/u8还含尾部循环。SASS列出字节按最后PC+16计算，包含NOP和末端填充，表示本次反汇编列出的代码范围；不是instruction cache流量或miss率。

| chains / factor | registers | 列出字节 | 主循环条目/每组轮数 | 尾部循环条目 |
| --- | ---: | ---: | --- | ---: |
| 4 / u1 | 20 | 1024 | 15 / 1 | — |
| 4 / u4 | 17 | 2048 | 52 / 4 | 15 |
| 4 / u8 | 17 | 2816 | 100 / 8 | 15 |
| 1 / u1 | 9 | 640 | 6 / 1 | — |
| 1 / u4 | 11 | 1024 | 15 / 4 | 6 |
| 1 / u8 | 10 | 1152 | 28 / 8 | 6 |

对4-chain整8轮，u1循环区间需要8×15=120个条目，u8主循环100个；约少1/6，但仍有循环外准备、input/output和调度。这是基于实际循环体的工作模型，不是把整个函数静态字节当动态次数。4-chain展开后register从20降到17，说明后端分配与调度不是源级变量数的单调函数；均无stack/spill，B256驻留均6，不能把性能差异归因于更多occupancy。

| N / rounds / chains | variant | Median / Min / P95 (us/op) | 逻辑Gupdate/s | 证据 |
| --- | --- | --- | ---: | --- |
| 65536 / 128 / 4 | u1 | 27.853 / 27.750 / 30.413 | 1204.71 | [CSV](../../results/20261023T162024Z_run_c5dxueid/samples.csv) |
| 65536 / 128 / 4 | u4 | 24.204 / 24.072 / 26.522 | 1386.31 | [CSV](../../results/20261023T162024Z_run_c5dxueid/samples.csv) |
| 65536 / 128 / 4 | u8 | 23.542 / 23.470 / 25.508 | 1425.32 | [CSV](../../results/20261023T162024Z_run_c5dxueid/samples.csv) |
| 65536 / 131 / 4 | u1 | 25.661 / 25.129 / 30.075 | 1338.23 | [CSV](../../results/20261023T162025Z_run_kc90crwp/samples.csv) |
| 65536 / 131 / 4 | u4 | 22.088 / 21.944 / 25.590 | 1554.75 | [CSV](../../results/20261023T162025Z_run_kc90crwp/samples.csv) |
| 65536 / 131 / 4 | u8 | 21.634 / 21.381 / 23.890 | 1587.34 | [CSV](../../results/20261023T162025Z_run_kc90crwp/samples.csv) |
| 262144 / 128 / 1 | u1 | 39.003 / 38.871 / 41.472 | 860.30 | [CSV](../../results/20261023T162027Z_run_696wbpa9/samples.csv) |
| 262144 / 128 / 1 | u4 | 25.436 / 25.385 / 27.144 | 1319.16 | [CSV](../../results/20261023T162027Z_run_696wbpa9/samples.csv) |
| 262144 / 128 / 1 | u8 | 23.839 / 23.429 / 25.938 | 1407.56 | [CSV](../../results/20261023T162027Z_run_696wbpa9/samples.csv) |

**预测与结果。** rounds128、chains4下u1→u8约27.853→23.542 us，减少约15.5%，与整组控制成本下降的方向相容。chains1、N262144保持总update=33554432，但输入输出字节和grid改变，u8观测23.839 us；它不能独立证明4条chain的ILP优势。该组u1到u8差距更大，也符合每轮控制成本占比更高的可能解释，具体调度/依赖stall仍未测。

rounds131本应多工作，实测三版本却比rounds128的另一进程更快；因此不建立“每轮固定耗时”的跨进程线性模型。按第1轮计划，固定rounds128做A/B/B/A：

| 顺序中的variant | Median / Min / P95 (us/op) | 证据 |
| --- | --- | --- |
| u1 | 27.812 / 27.720 / 30.730 | [CSV](../../results/20261023T162207Z_run_a14u7z8z/samples.csv) |
| u8 | 23.603 / 23.480 / 27.721 | [CSV](../../results/20261023T162208Z_run_xquxkmla/samples.csv) |
| u8 | 25.774 / 23.528 / 30.515 | [CSV](../../results/20261023T162210Z_run_ffd_ehvp/samples.csv) |
| u1 | 27.758 / 27.718 / 30.444 | [CSV](../../results/20261023T162212Z_run_duffqh5w/samples.csv) |

A=u1两轮约27.812/27.758 us；B=u8约23.603/25.774 us。B均低于A，但B自身变化约9.2%，P95也有交叠；当前支持本组展开有收益方向，不把首轮约15.5%写成稳定常数。跨进程频率、功耗或后台负载未隔离，rounds131异常的具体来源仍未知。保留该反例及补测，不选择只符合模型的轮次。

优化决定：当前u4/u8值得作为后续实现候选，仍保留u1基线。代码范围增长至2–2.75倍，但没有测instruction-cache指标，既不能说已受代码缓存惩罚，也不能断言更大展开始终有效；不为凑单调曲线继续扩大factor。

## 第5轮：Branch与Predication，先辨认最终控制流

[branch SASS](../../results/20261023T161901Z_static_mt5q3vck/branch_flow.sass.txt)显示loop body在0x00f0取flag、0x0110产生predicate、0x0120据此BRA到另一分支，BSSY/BSYNC包围两路，各有一个真实回边（0x1e0→0x190及0x2a0→0x250）。所以长body的任务重排确实改变了同一warp在该分支处的路径分布。

short body则在0x0130–0x01b0出现@P0/@!P0控制的算术，没有数据flag控制的BRA。尾部0x1e0自跳转属于填充，不能作为divergence证据。short14 registers、列出640 B；loop11 registers、列出1024 B；均无local/shared/stack，B256理论驻留6。

源码路径B的左移xor也发生了代数变换：`(a*x+b)<<7`在uint32模运算中等于`(128*a)*x+128*b`，因此SASS可用两次IMAD和LOP3表达该更新，未必保留一条显式左移。路径A用IMAD、SHF、LOP3；两路loop区间各6条（含counter/比较/BRA），但这些指令不是相同流水线工作。不能仅按源码“各一次更新”推断两路硬件耗时完全一样。

| N / body / rounds | layout | 模型mixed warps | Median / Min / P95 (us/op) | 逻辑Gupdate/s | 证据 |
| --- | --- | ---: | --- | ---: | --- |
| 65536 / short / 1 | uniform | 0 | 6.490 / 5.571 / 11.263 | 10.10 | [CSV](../../results/20261023T162028Z_run_cg_fn8e6/samples.csv) |
| 65536 / short / 1 | interleaved | 2048 | 10.476 / 5.376 / 12.121 | 6.26 | [CSV](../../results/20261023T162028Z_run_cg_fn8e6/samples.csv) |
| 65536 / loop / 128 | uniform | 0 | 14.335 / 12.869 / 17.828 | 585.17 | [CSV](../../results/20261023T162029Z_run_1ym82qpi/samples.csv) |
| 65536 / loop / 128 | interleaved | 2048 | 21.788 / 21.719 / 23.777 | 385.02 | [CSV](../../results/20261023T162029Z_run_1ym82qpi/samples.csv) |
| 1048576 / loop / 128 | uniform | 0 | 150.149 / 132.198 / 152.934 | 893.90 | [CSV](../../results/20261023T162030Z_run_nha084qj/samples.csv) |
| 1048576 / loop / 128 | interleaved | 32768 | 260.086 / 257.678 / 262.234 | 516.05 | [CSV](../../results/20261023T162030Z_run_nha084qj/samples.csv) |

**长body的有限归因。** N65536下同一kernel、同数据任务集合、同两路数量、同资源，interleaved约为uniform的1.52倍；N2^20约1.73倍。完整warp混合两路与更慢方向一致，比换kernel/改变工作量的对照更直接。但固定global工作、循环准备、调度和lane执行情况仍有影响，不能把比例反演为branch efficiency，也不要求恰好2倍。活跃lane和真实branch/stall计数器仍未测。

**短body的反例与NSYS补测。** 普通运行6.490/10.476 us的Median似乎有差，但Min分别5.571/5.376、P95分别11.263/12.121，分布重叠；SASS又没有数据分支，因此不解释为“两条branch串行”。按第1轮计划单独[采集时间线](../../results/20261023T162213Z_nsys_wb5xurpn/timeline.nsys-rep)，每个layout按collect顺序有`1+3*(5+100+1)=319`个kernel，合计638。[只读时间戳分析](../../results/20261023T162325Z_stage5-nsys-analysis_h3o7j0uf/002_python3.log)核对数量、每case单stream与grid/block。

本次插桩两layout的kernel活跃Median同为1.600 us、P95同为1.632 us；正式batch的active mean也约1.606–1.611 us。可是[插桩Event](../../results/20261023T162213Z_nsys_wb5xurpn/instrumented_samples.csv)为47.8–129.1 us/op，按kernel首末时间戳计算的包围span也达到46.1–128.6 us/op。说明**本次profile运行**的大部分batch范围落在kernel间隙，而不是kernel活跃；这个运行受插桩/系统状态的影响明显，不能用它替代6–10 us的普通基线，或据此精确分解普通运行。

该补测没有支持“short interleaved kernel本身稳定更慢”，因此保留普通结果但撤回稳定排序；并未声称任何predication工作负载都无布局成本。下一步若需要更小差异的可信比较，要增加可观测工作以减弱提交影响、保留独立基线和顺序控制，再查询动态指标。

## 结论、边界与阶段交接

Stage 5基础验收完成：三个可独立运行的程序、signed/unsigned除法语义、常量/shift对照、unroll尾部与多chain、同任务重排的short/loop分支、CPU逐字验证、54个GPU CTest、20次Sanitizer、逐kernel静态证据、15份基线数据审计、A/B/B/A和独立NSYS补测，以及原理、练习和独立答案齐备。

分析保留了两次必要修正：除法的准备成本在循环外；短body是predication且其普通排序不足以归因。还保留展开131轮反而更快这一跨进程反例，不改写原始CSV，也不冒称已定位clock原因。

当前仅原生SM89/Toolkit13.4，其他SM7x/8x、精确13.1及强制PTX JIT兼容性仍未通过本阶段复现。NCU不可用，动态instruction/active-lane/branch效率、stall、cache/DRAM、achieved occupancy均Not Tested；静态条目数不替代它们。

下一阶段S6研究host传输与并发，继续把API异步性、单stream顺序、kernel活跃时间、pipeline包围span和host端到端分别测量。必须用时间区间的实际交集证明copy/compute重叠，不能仅凭使用cudaMemcpyAsync或多条stream宣称发生并发。
