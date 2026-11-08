# 项目进展

更新：2026-10-24，Asia/Shanghai（UTC+8；历史记录仍保留原时间标注）。

七阶段项目已完成课程架构、通用工具、15个独立实验及全部必做阶段的基础验收；以下验收记录采集时使用已移除的 `lab.sh`，原始命令保留在对应 `results/` 快照中，当前操作入口见 [根 README](../README.md)。实测范围限定原生SM89路径。主教材：[CUDA C++ Best Practices Guide 13.1](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-c-best-practices-guide/index.html)。逐项交付依据、当前源码与验证记录的对应关系，以及未测项见[完成审计](completion-audit.md)。下面保留各阶段交付时的验收记录，最后汇总最终状态。

## 已交付

- [七阶段课程地图](roadmap.md)：目标、章节、模型、控制变量、问题、证据、验收条件。
- C++17/CUDA/CMake 架构，独立实验 target；历史采集曾用 `lab.sh`，现按功能拆为 `commands/<动作>/run.py`；无 Toolkit 时仍支持 host-only 工具测试。
- 通用参数、CUDA 错误/资源检查、设备信息、Event/host 计时、Min/Median/P95 和全量数值验证。
- Stage 1 Vector Add：scalar/grid_stride；Memory Copy：scalar/grid_stride/Runtime D2D 参考。
- 每次独立证据目录，保存原始命令、退出码、源码与二进制哈希、构建参数、日志与 CSV，拒绝覆盖已有 CSV。
- [Stage 1 学习指南](../labs/01_measurement/README.md)、[分析报告](../labs/01_measurement/analysis.md)、[练习](../labs/01_measurement/exercises.md) 与分离的[答案](../labs/01_measurement/answers.md)。

## 验收与证据

| 项目 | 状态 | 证据 |
| --- | --- | --- |
| SM89 Release 构建，3 个 host 测试 | PASS，最终构建再次通过 | [初次](../results/20261023T140745Z_build_e8_x9zhd/status.txt) / [最终](../results/20261023T142446Z_build_l5kkd1x9/status.txt) |
| 18 个 GPU shape 测试、8 个 block 配置、CLI/CSV 检查 | PASS | [verify](../results/20261023T140811Z_verify_7xd76rnk/verification.txt) |
| 4 类 Sanitizer × 2 实验 | PASS，覆盖 N=4099 | [sanitizer](../results/20261023T141621Z_sanitizer_h19d9rnu/status.txt) |
| PTX/SASS 与资源静态检查 | PASS | [static](../results/20261023T141321Z_static_8dc7bvhl/status.txt) |
| 大小工作集、短 kernel K=1/100 | 已测，全部 CPU reference 通过 | [完整分析与原始 CSV](../labs/01_measurement/analysis.md) |
| Nsight Systems CUDA 时间线 | PASS；与基线独立 | [trace](../results/20261023T141752Z_nsys_o8egx3xz/timeline.nsys-rep) |
| 无 CUDA target 的 host-only 配置 | PASS | [host-test](../results/20261023T141730Z_host-test_d87_o8wy/status.txt) |
| Nsight Compute 计数器 | Not Tested：ncu 不可用 | [status](../results/20261023T141753Z_ncu_rhoaug_y/status.txt) |
| 强制 PTX JIT | 已测试，失败：unsupported toolchain | [原始错误](../results/20261023T141840Z_run_2lh0m_0c/014_vector_add.log) |
| SM7x、其他 SM8x、Toolkit 13.1 编译 | Not Tested：未在这些环境复现 | 当前仅实测 SM89 / Toolkit 13.4 |

**Stage 1 基础实验验收完成（限定原生 SM89 路径）。** CUDA 计数器归因尚未完成；强制 PTX JIT 兼容性失败未消除。没有将缺失工具或失败检查记作通过。S2–S7的阶段证据按顺序记录于下文。

Stage 1 交付时通过了 15 份 Markdown 的本地链接、脚本语法与已记录源码/二进制哈希核对。其最后一次构建仅改进编译元数据说明，分别标明 CXX 工具编译器与 CUDA host 编译器；内核、计时、校验逻辑未改变。报告继续引用采集时的原始快照与二进制，不改写历史 CSV。

## Stage 2 已完成

新增 `global_copy`：连续 scalar、float2/float4、stride、offset；gather/scatter/both；向量对齐/尾部保护；整个输出区域（包括洞和前缀）的逐位验证。增加通用 case/CSV 元数据工具，统一入口支持只选择一个实验构建、验证、静态和 sanitizer。

| 验收 | 结果与证据 |
| --- | --- |
| 编译、27 个 GPU CTest、8 组额外配置和非法参数 | [PASS](../results/20261023T145655Z_verify_n_qo85lc/verification.txt) |
| 16 次 sanitizer：4 工具 × 连续及3种布局 | [PASS](../results/20261023T145755Z_sanitizer___y1_d_0/status.txt) |
| 实际 64/128-bit 宽访存及资源 | [PASS](../results/20261023T150441Z_static_npsfrq7e/status.txt) |
| 零预热拒绝、覆盖保护、CSV 模型/统计复算 | [PASS](../results/20261023T150442Z_stage2-audit_fve49yp8/status.txt) |
| 大小向量对照、stride/offset 扫描和 A/B/B/A 复测 | [完整分析与 CSV](../labs/02_global_memory/analysis.md) |

学习材料：[指南](../labs/02_global_memory/README.md)、[练习](../labs/02_global_memory/exercises.md)、[独立答案](../labs/02_global_memory/answers.md)。基础验收完成，计数器归因未测。

关键发现：大规模向量化对照都约 225–227 GB/s，未出现向量宽度倍数加速；stride 扫描同时扩大地址跨度，不能只归因于 sector 利用率。offset 复测出现明显跨进程漂移（相同配置 Median 约 54→97 us），因此不宣称小幅对齐收益或稳定排序。保留反例，不修改原始数据。

## 已得到的结论

大工作集的 requested 有效带宽约为 226–232 GB/s；小工作集更高，但没有 cache/DRAM 指标，仍需区分缓存与调度等解释。短 Add 的 Nsight kernel 活跃时长 Median 为 0.992 us，而同一插桩运行的批量 Event span/op 为 7.058 us，说明两种计时不能混用。数据、参数、公式与范围见分析报告。

grid-stride 在本次大规模数据中有小幅观测差值，尚未以顺序交换和计数器建立稳定收益或具体机制。不能将“更多寄存器”或“较高带宽”单独写成瓶颈证明。

## Stage 3 已完成

新增三个独立程序及完整指南、分析、练习与答案：[Memory Hierarchy](../labs/03_memory_hierarchy/README.md)。9种Transpose形状、6组Cache参数、15组Local参数通过；三程序各四类Sanitizer无错误，静态检查确认shared/barrier、循环内global load及local/spill；9份CSV、19组variant独立复算统计。原始证据统一链接于[分析报告](../labs/03_memory_hierarchy/analysis.md)。

关键发现：4096²转置naive/tiled/padded Median约3623/1055/656 us；padding未降低本机理论驻留量。固定请求数的工作集扫描从64 KiB到64 MiB，时间约127→308 us，但cache hit未测。动态索引数组有128 B线程栈、无ptxas spill；预算32版本有48 B线程栈及72/72 B静态spill报告，理论驻留量却仍为6 blocks/SM，实测更慢。不能用动态数组直接代替寄存器不足的证据。

按用户要求增加[可复核的分析与决策记录方法](analysis-workflow.md)，更新模板；S1/S2补充事后证据复盘，S3完整记录代码模型、工具选择、结果偏差、竞争解释和下一轮判定。事后分析不冒充实验前预测。

## Stage 4 已完成

新增[执行配置实验](../labs/04_execution_config/README.md)：interleaved/interleaved_mask/sequential/warp多pass归约，B/I扫描；64累加值的unlimited/r32/r48/r64寄存器预算与block扫描。最终24个归约CTest、18个寄存器CTest、各四工具Sanitizer、PTX/SASS、14份CSV/35组variant的统计/单位与哈希核对通过；另有NSYS单stream的638个kernel对应319个两pass操作的时间戳核对。证据与全部数据见[分析](../labs/04_execution_config/analysis.md)。

记录了一轮完整干预：从SASS发现运行时余数条件有MUFU.RCP及修正序列，先提出等价位掩码假设，再实现/验证/采样；mask序列确实简化，N=2^20本次更快，但N=2^24几乎没有收益。没有将指令减少等价为全规模稳定加速。

寄存器反例：B256时unlimited实际71 registers、理论occupancy50%，Median约25.373 us；r32升至100%却有492/492 B静态spill报告，Median约519.158 us。这证明理论occupancy不能单独决定性能；真实local流量/stall/achieved occupancy仍未测。

统一入口已实际[构建全部8个已实现程序并通过9个host/help检查](../results/20261023T160617Z_build_less9x9h/status.txt)；[文档与脚本审计](../results/20261023T160658Z_stage4-doc-audit_qgrttztz/status.txt)核对26份Markdown的本地目标、四阶段必需文件和脚本语法。此检查不替代各实验的GPU验证或全文语义审查。

## Stage 5 已完成

新增[整数指令与控制流实验](../labs/05_instructions_control/README.md)：integer_division的signed/unsigned、runtime/constant/manual shift；loop_unroll的1/4/8展开与1/4 chains；branch_flow对同一逻辑任务集做uniform/interleaved排列，区分short与loop body。54个GPU CTest、20次Sanitizer、静态指令检查、15份普通CSV/31组variant的统计和哈希核对通过。[完整分析与证据](../labs/05_instructions_control/analysis.md)保留采集前预测与补测设计。

关键修正：runtime除法的倒数准备已移到循环外；unsigned常量/8与手写shift的列出SASS相同。4-chain u8比u1减少循环控制，A/B/B/A仍观察到收益方向，但u8自身有波动，未承诺固定倍数。short body最终是谓词算术，NSYS补测两种布局kernel活跃Median均1.600 us，而batch span大得多；不把普通短任务的排序归因于不存在的数据分支。loop body保留真实两路分支，交错layout本次更慢，动态branch效率仍未测。

统一入口已[构建全部11个实验并通过12个host/help检查](../results/20261023T162755Z_build__p_cpozg/status.txt)。[文档审计](../results/20261023T162756Z_stage5-doc-audit_q432nhpm/status.txt)通过29份Markdown的本地目标和新增脚本语法检查；各GPU验收仍以阶段证据为准。

## Stage 6 已完成

新增[异步实验与教学材料](../labs/06_async_execution/README.md)：双向pageable/pinned传输、独立buffer ring的chunk流水线、同jobs数量的serial/concurrent kernel组。34个GPU CTest、20次Sanitizer、CLI/CSV、静态检查通过；已[集成构建14个程序并通过15个host/help测试](../results/20261023T165321Z_build_s84u8nfs/status.txt)。21份普通CSV/33组variant已[独立复算统计/公式并校对源码与二进制哈希](../results/20261023T165703Z_stage6-data-audit_4uvtp9on/status.txt)。普通计时、五份NSYS trace及解析证据见[完整分析](../labs/06_async_execution/analysis.md)。

本阶段按实际证据推进了一轮优化：初版没有copy-compute重叠，先提出单引擎提交顺序假设，再增加phased分组提交并保留interleaved基线。ABBA支持当前工作点的整流水收益，但短计算仍无copy-compute交集；延长计算后，正式样本每op约5.952 us的copy-compute交集远小于984.854 us的kernel并发，因此没有把总收益全归因于传输隐藏。stream/buffer/chunk扫描保留控制变量、派生容量/launch数、漂移与负结果。独立小kernel组观察到最多4个kernel同时活跃。

指南包括事件依赖图、跨stream slot复用证明、host buffer生命周期、四种计时范围、参数/单位与独立练习答案。NSYS解析曾因phase/stage变量覆盖失败，原始错误保留，修复后重检所有相关trace；NCU仍Not Tested。当前平台确认为WSL2，只记录其适用范围，不把平台标签当作未测机制的答案。

[阶段6文档审计](../results/20261023T165803Z_stage6-doc-audit_8btwzamk/status.txt)通过32份Markdown本地链接、六阶段必需材料及脚本语法。

## Stage 7 已完成

新增[传统CUDA Core FP32 GEMM](../labs/07_gemm/README.md)：先验证Naive/Coalesced，再依次加入Shared Tiled、Register Tiled、Unrolled。五版共用全量FP64参考、基于绝对积和的前向误差界、NaN漏写检查；覆盖两种T/R、小/矩形/尾部、random/identity/cancellation。44个GPU CTest、8次Sanitizer、CLI/CSV及静态指令检查通过；21份普通CSV/67组variant的统计、容量、请求、FMA、sector、资源和哈希[独立复核通过](../results/20261023T174711Z_stage7-data-audit_0aguvimk/status.txt)。

[完整报告](../labs/07_gemm/analysis.md)保留预先假设、五步性能表、资源与代码体积、NSYS小/大矩阵补测、八次ABBA及数值诊断。关键修正：展开不仅减少循环控制，还改变shared加载宽度与调度；32³的Event排序与kernel活跃时间排序可相反；矩形ABBA同配置漂移超过十倍，撤回固定排序。相消参考接近0时相对误差很大，但绝对误差满足推导的界。未用单一耗时反推DRAM/cache或stall数值。

## 最终集成与交付核对

[集成构建](../results/20261023T174000Z_build_zlhfx6wo/status.txt)包含15个程序，16个host/help测试通过；[host-only](../results/20261023T174006Z_host-test_f_l2ed03/status.txt)也通过。先前一次集成动作因Broken pipe中止已保留为失败，未覆盖。

[最终证据审计](../results/20261023T175133Z_final-evidence-audit_sudrgn5o/status.txt)按15个程序分别核对当前翻译单元、各阶段verify/static/sanitizer原始日志、43组源码/二进制快照哈希与最终编译参数，累计249个GPU CTest、92次Sanitizer运行通过。该总数来自各阶段对应记录，未重复把历史同一测试算作新增覆盖。工具日志解析按实际CTest及racecheck输出格式修正，原始审计失败仍保留。

[传递依赖审计](../results/20261023T174602Z_final-source-evidence-audit_27b0s0_j/status.txt)确认15个程序的当前源码及本地依赖头文件与最新GPU验证快照一致；此前发现Global Copy公共CLI头文件后加了可选参数省略功能，已[针对当前源码重跑27个GPU测试及CLI/边界](../results/20261023T174203Z_verify_swkdgjrv/verification.txt)。还用S5旧trace回归检查了扩展二维grid/block后的解析器。

全项目按原始目标逐项复核见[完成审计](completion-audit.md)。[文档检查](../results/20261023T175558Z_final-doc-audit_r7ces91u/status.txt)核对35份Markdown的本地链接与标题、七阶段必需材料及Python/shell语法；它不代替内容审阅或GPU验证。课程工程交付完成；个人是否掌握需要独立完成练习和复现，不能以工程测试替代学习评价。

## 未测项与后续学习

1. NCU动态sector/cache/DRAM、bank conflict、issue/stall和achieved occupancy：Not Tested，工具不可用；命令和观察指标类别已提供，环境具备后另建证据记录。
2. SM7x、其他SM8x、精确Toolkit13.1：Not Tested；当前Toolkit13.4不支持sm70，不能用错误目标架构冒充兼容。强制PTX JIT已实际失败，需匹配的Toolkit/driver环境才能复现兼容路径。
3. 完成S1–S7练习，再尝试自己的受控优化与归因；可选SM80+ cp.async尚未实现，不属于本次必做交付。

当前未修改系统驱动、GPU状态或计数器权限；所有失败和原始实验数据保留。
