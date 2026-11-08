# Stage 3：从地址模型、指令到性能解释

日期：2026-10-23（Asia/Hong_Kong）。本报告的实测数据、公式复算与事后证据复盘分别注明；后者不冒充采样前已登记的预测。所有原始记录保持不变。

## 环境、计时与验收

RTX 4060 Laptop GPU，SM89，24 SM，driver 596.21，nvcc 13.4.92，Runtime 13040，API报告L2=32 MiB；C++17、Release、`-O3 -lineinfo --ptxas-options=-v`、原生sm89。各CSV保留环境、输入、资源和编译字段，记录目录保留源码/二进制哈希及确切编译命令。未锁频，未持续采集功耗或频率；沿用S2已经发现的漂移警惕，不把小幅差异解释成稳定收益。

全部测量使用block=256、warmup=5、repeats=21、iterations=100；未插桩，预分配pageable host缓冲区，重复使用device输入输出。Event为批量span/op，包含提交间隙；host E2E是上传+一次kernel+回读+同步，排除初始化、分配和CPU验证。表中GB/s仅为requested/time，不是DRAM流量。Min/Median/P95按各时间列独立统计。

| 验收项 | 本次结果与原始证据 |
| --- | --- |
| Transpose：9形状、边缘tile、CSV和非法参数 | [PASS](../../results/20261023T152131Z_verify_lf979g37/verification.txt) |
| Cache：6组T/W/S、CSV与非法参数 | [PASS](../../results/20261023T152137Z_verify_ohiop8sv/verification.txt) |
| Local：15组N/rounds，三个版本与非法参数 | [PASS](../../results/20261023T152922Z_verify_01b_hffk/verification.txt) |
| Transpose四类sanitizer，37×65 | [PASS](../../results/20261023T152239Z_sanitizer_8j0rzr5s/status.txt) |
| Cache四类sanitizer，T65/W128/S3 | [PASS](../../results/20261023T152245Z_sanitizer_567k5_3t/status.txt) |
| Local四类sanitizer，N33/rounds3，三个版本 | [PASS](../../results/20261023T152934Z_sanitizer_bhmojk4g/status.txt) |
| 9份性能CSV、19组variant的原始样本/统计/单位独立复算 | [PASS](../../results/20261023T153701Z_stage3-audit_0o58z5hd/status.txt) |

Sanitizer日志无错误/hazard报告；它只支持所测输入和检查范围，插桩耗时不进入下表。Transpose与Cache逐位比较；Local以float FMA reference、atol=rtol=1e-5检查，每次正式采样后校验。

## Transpose：先修复全局映射，再改变shared pitch

模型见[README](README.md)。naive读取`in[r*C+c]`连续，写`out[c*R+r]`跨步；tiled用shared交换访问方向，使global读写都连续，但shared转置读取的pitch32导致完整warp访问同一bank的不同地址。padded只将pitch改为33：`bank=(33*lane+col)%32`，32个lane落在32个bank；shared成本从4096增至4224 B。预测是tiled改善global映射，padding进一步改善shared冲突；小矩阵可能受固定成本影响，资源增加也可能抵消收益。

[PTX/SASS与资源](../../results/20261023T152024Z_static_ej0lhj14/)确认LDS/STS/barrier存在；naive18 registers，tiled/padded均22，均无local；Runtime理论驻留都是6 blocks/SM。因此本机没有观察到128 B padding导致理论驻留下降。真实bank-conflict计数器仍未测。

| 规模 | 版本 | Event Median / Min / P95 (us/op) | requested GB/s | Host E2E Median (us) | 证据 |
| --- | --- | --- | ---: | ---: | --- |
| 37×65 | naive | 14.039 / 8.745 / 22.284 | 1.37 | 110.12 | [CSV](../../results/20261023T153103Z_run_0dyyvlgk/samples.csv) |
| 37×65 | tiled | 11.356 / 8.274 / 15.247 | 1.69 | 94.92 | [CSV](../../results/20261023T153103Z_run_0dyyvlgk/samples.csv) |
| 37×65 | padded | 6.655 / 6.093 / 7.762 | 2.89 | 47.31 | [CSV](../../results/20261023T153103Z_run_0dyyvlgk/samples.csv) |
| 1024×1024 | naive | 237.117 / 232.786 / 237.486 | 35.38 | 1643.71 | [CSV](../../results/20261023T153118Z_run_eu2r40c2/samples.csv) |
| 1024×1024 | tiled | 69.509 / 69.375 / 74.373 | 120.68 | 1233.91 | [CSV](../../results/20261023T153118Z_run_eu2r40c2/samples.csv) |
| 1024×1024 | padded | 22.896 / 22.026 / 27.996 | 366.38 | 1342.44 | [CSV](../../results/20261023T153118Z_run_eu2r40c2/samples.csv) |
| 4096×4096 | naive | 3622.513 / 3608.883 / 3716.659 | 37.05 | 19604.64 | [CSV](../../results/20261023T153121Z_run_4zhz96vw/samples.csv) |
| 4096×4096 | tiled | 1054.749 / 978.289 / 1065.298 | 127.25 | 17052.83 | [CSV](../../results/20261023T153121Z_run_4zhz96vw/samples.csv) |
| 4096×4096 | padded | 655.727 / 651.274 / 660.613 | 204.69 | 16670.26 | [CSV](../../results/20261023T153121Z_run_4zhz96vw/samples.csv) |

**事后复盘与反推。** 4096² requested=134217728 B。naive→tiled的时间比约3.43；tiled→padded约1.61。模型的32路bank冲突不能预测整个kernel快32倍，因为shared读只占执行过程的一部分，还有global搬运、barrier、地址计算和调度。padding同样不能改变8RC的逻辑字节。当前数据支持这些改动在本次大矩阵上有效，不能由总时间倒推出具体shared stall百分比。

1024²工作集8 MiB时padded约366 GB/s，4096²工作集128 MiB时约205 GB/s。规模同时改变缓存复用、grid和固定成本，所以不能仅凭两个点量出L2命中率。小37×65只有6个tiled blocks，小于24 SM，且不同版本的Min/P95差距较大；这里的排序不足以证明bank冲突主导延迟。保留小矩阵用于边界与反例，不给稳定加速承诺。

**后续判定。** 在NCU可用环境比较tiled/padded的shared访问/冲突与global流量：若shared冲突下降而global工作量相近，才增强bank归因；如果没有预期差异，要检查最终指令宽度、活跃lane和指标含义。此项待测。当前决定保留padded作为大矩阵候选，同时保留naive/tiled以展示两个独立优化步骤。

## Cache Working Set：固定请求数后仍观察到时间变化

固定T=65536、S=256，读取16777216次并写65536个输出。requested=`4*T*(S+1)=67371008 B`，核心add=16777216；仅改变W。源输入分别64 KiB、1 MiB、16 MiB、64 MiB，另有256 KiB输出。访问表达式`(i+step*T)&(W-1)`保持每个warp的基本连续布局。预测小工作集可能复用缓存，大工作集可能需要更多较低层访问；不是预先认定某点就是L1/L2容量。

[静态证据](../../results/20261023T152026Z_static_rv8s_vlt/)确认LDG在真实循环回跳区间内，12 registers，无shared/local。若load被提到循环外，该实验就不能用于原定请求模型；这一检查排除了该编译器变换。它尚不能证明每次LDG到达哪个缓存层。

| 规模 | 版本 | Event Median / Min / P95 (us/op) | requested GB/s | Host E2E Median (us) | 证据 |
| --- | --- | --- | ---: | ---: | --- |
| 64 KiB | scalar | 127.150 / 126.392 / 132.096 | 529.85 | 249.31 | [CSV](../../results/20261023T153139Z_run_rdgj1tdp/samples.csv) |
| 1024 KiB | scalar | 128.174 / 126.996 / 132.966 | 525.62 | 504.71 | [CSV](../../results/20261023T153147Z_run_tm49prep/samples.csv) |
| 16384 KiB | scalar | 177.940 / 177.367 / 182.526 | 378.62 | 2312.05 | [CSV](../../results/20261023T153149Z_run_aqdufmbi/samples.csv) |
| 65536 KiB | scalar | 307.548 / 300.451 / 309.011 | 219.06 | 8421.68 | [CSV](../../results/20261023T153151Z_run_gyxhw15k/samples.csv) |

**事后复盘与反推。** 64 KiB与1 MiB时间相近（约127/128 us），16 MiB约178 us，64 MiB约308 us。在计算/循环次数固定后，工作集仍与明显时间差相关，比S2的stride扫描减少了一项混杂。64 MiB源加输出超过报告L2容量，requested速率约219 GB/s，与先前大Copy的约226 GB/s接近；这是较低存储层吞吐限制的支持线索，不是DRAM饱和证明。

不能由64 KiB/1 MiB两点相近反推二者有相同cache hit率：每线程累加有依赖链，循环控制和同址复用也可能限制时间。W<T时不同线程还复用输入地址；W增大也改变跨线程/跨步复用距离，这是本实验工作集干预的组成，不应声称只改变硬件容量。E2E随W明显增加，主要计时范围还包含更多输入传输，不能用它与固定请求的device时间混为一谈。

**下一轮最小实验（待测）。** 在固定W下独立扫S，检查时间是否近似线性以及固定成本；使用NCU查询实际L1/L2/DRAM指标；若要测依赖式访问延迟，应另建pointer-chase及其参考实现，不能给本实验换个名字当作latency。当前保留吞吐—工作集观察，不拟合缓存容量。

## Local Array：把可寻址数组与spill拆开

每线程32个独立值，每值执行32轮FMA；static_index允许标量化，dynamic_index用运行时旋转索引访问同一组值。旋转是双射，数学工作及最终求和顺序相同，但地址计算不同。仅比较这两者无法确认“慢是寄存器不够”；因此增加register_limited，与static_index使用同一helper，仅限制寄存器预算32。

| 版本 | Runtime registers/thread | Runtime local B/thread | ptxas stack | ptxas spill stores/loads | 理论blocks/SM |
| --- | ---: | ---: | ---: | --- | ---: |
| static_index | 39 | 0 | 0 B | 0 / 0 B | 6 |
| dynamic_index | 40 | 128 | 128 B | 0 / 0 B | 6 |
| register_limited | 32 | 48 | 48 B | 72 / 72 B | 6 |

依据：[最终静态输出](../../results/20261023T152544Z_static_m8tjze6p/)、[ptxas编译日志](../../results/20261023T152544Z_static_m8tjze6p/012_cmake.log)、对应CSV。PTX的动态数组有local load/store，受限kernel有`.maxnreg 32`；后端spill要看ptxas及SASS，不能要求所有spill已出现在前端PTX。cuobjdump资源中的LOCAL字段与STACK/Runtime localSizeBytes含义不同，不用其单独为0否认线程栈。ptxas的72 B是静态spill报告，不能直接乘循环次数当作实测运行流量。

| 规模 | 版本 | Event Median / Min / P95 (us/op) | requested GB/s | Host E2E Median (us) | 证据 |
| --- | --- | --- | ---: | ---: | --- |
| 4096 | static_index | 12.585 / 7.680 / 23.890 | 2.60 | 101.20 | [CSV](../../results/20261023T153155Z_run_nt9kyskx/samples.csv) |
| 4096 | dynamic_index | 91.443 / 91.320 / 95.683 | 0.36 | 175.37 | [CSV](../../results/20261023T153155Z_run_nt9kyskx/samples.csv) |
| 4096 | register_limited | 28.815 / 28.549 / 29.594 | 1.14 | 119.05 | [CSV](../../results/20261023T153155Z_run_nt9kyskx/samples.csv) |
| 65536 | static_index | 40.817 / 40.714 / 45.261 | 12.84 | 227.17 | [CSV](../../results/20261023T153156Z_run_b2vjinz4/samples.csv) |
| 65536 | dynamic_index | 1025.423 / 811.192 / 1314.918 | 0.51 | 1271.65 | [CSV](../../results/20261023T153156Z_run_b2vjinz4/samples.csv) |
| 65536 | register_limited | 156.744 / 151.757 / 162.744 | 3.34 | 335.69 | [CSV](../../results/20261023T153156Z_run_b2vjinz4/samples.csv) |

**事后复盘与优化决策。** N=65536时核心update为134217728 FLOP，static约3288 GFLOP/s、dynamic约131、limited约856，均不含初始化和最终sum算术。dynamic既有local访问又有更多索引操作，不能把约25倍时间比等同于单次local/register延迟比；其Min到P95也有明显漂移。

受限版本使寄存器39→32，却没有让理论驻留从6增加；同时产生spill，时间40.817→156.744 us。因而本次没有“用spill换更多驻留warp”的补偿，限制预算是负面优化。这里已经足以否定“寄存器越少越好”，但不代表所有限制预算都无用。下一阶段扫描block及更多预算，先预测是否跨驻留台阶，再验证spill与时间，而不是反复压低寄存器数。

N=4096只有16 blocks，小于SM数，static的7.680/12.585/23.890 us波动也大；不能根据这个点推导SM的峰值计算吞吐。global requested GB/s在本实验不是主要性能目标，因为每个输入经历大量FMA及潜在local操作；保留该列仅为统一逻辑I/O定义。

## 适用边界与阶段交接

Stage 3基础验收完成：三个程序、各对照、CPU reference、边界/同步检查、静态指令、真实样本、独立统计审计及分离练习/答案齐备。完成的是当前SM89原生路径的实验建设，不是所有机制均已被计数器证实。

NCU在本轮环境探测仍不可用：cache hit、实际sector/DRAM/local bytes、bank conflict、stall、achieved occupancy均Not Tested；本阶段没有另采NSYS时间线，不借用S1的时间线作为本次结果。其他SM7x/8x及精确Toolkit13.1未测；原有强制PTX JIT兼容性失败仍保留。后续S4将围绕归约、block与寄存器预算建立资源—并行度—时间的受控对照。
