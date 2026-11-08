# 项目完成审计

审计日期：2026-10-24，Asia/Shanghai。以下是当时源码与历史取证的审计结论，不是当前脚本拆分后的全项目复验；新入口的局部验证见 [根 README](../README.md)。当时按目标核对构建配置、实际命令、原始日志、分析与教学材料，而不以阶段摘要或测试数量代替交付判断。

**必做七阶段、15个独立程序及教学材料已交付。** 实测验收限定RTX4060 Laptop / CC8.9、driver596.21、Toolkit13.4.92、WSL2的原生sm89路径；缺失的NCU计数器、其他架构/精确13.1环境和已失败的强制PTX JIT路径没有记作通过。项目工程完成不等于学习者已经掌握课程。

## 原始要求逐项对应

| 原目标部分 | 实现与证据 | 判断与边界 |
| --- | --- | --- |
| 一：以BPG13.1为教材，覆盖测量、内存、执行资源、指令、控制流、异步及归因 | [课程地图](roadmap.md)按归档章节定位，七阶段指南给出能力目标和机制；[分析方法](analysis-workflow.md)与各analysis保留完整实验循环 | 已交付；个人能力需独立做练习评估 |
| 二：单GPU、CC7.x/8.x≤89、传统CUDA Core，排除SM90+/多GPU/Tensor替代 | [CMake](../CMakeLists.txt)限定目标范围，[Device](../common/device_info/device.hpp)核对运行设备；源码仅使用当前课程范围内Runtime/SIMT实现；GEMM最终SASS为FFMA、无MMA类指令 | 范围符合；其他目标未实测。可选cp.async未实现，不作为必做要求 |
| 三：S1–S7全部课程实验 | 下表逐一列出15个程序、基线/对照及分析入口；包括所有五版GEMM | 七阶段基础验收均完成，非仅目录规划 |
| 四：每实验九类教学交付 | 下文九项矩阵对应七阶段README、analysis、exercises、answers和源码 | 已审阅具体内容，文件存在检查只是补充 |
| 五：预热/重复、Min/Median/P95、设备与host时间、规模/缓存/对齐、等量工作、元数据、不造数 | [方法](methodology.md)、[通用收集器](../common/benchmark/experiment.hpp)、[独立统计复算](../scripts/check_samples.py)；各阶段CSV保留raw及环境/输入/范围 | 已实现并实测；短任务Event与kernel活跃时间、S6整流水与单kernel明确分开；噪声大时撤回稳定速度排序 |
| 六：C++17/CUDA/CMake、少依赖、操作入口、独立target、O3/lineinfo/ptxas、容量上限与能力检查 | [按动作拆分的脚本](../README.md#操作按功能拆开)、各阶段CMake与CLI；[最终15程序构建](../results/20261023T174000Z_build_zlhfx6wo/status.txt)和[host-only](../results/20261023T174006Z_host-test_f_l2ed03/status.txt) | PASS；Python仅标准库，CPU工具无Toolkit配置可运行；默认容量保守，非法输入拒绝 |
| 七：独立项目、docs/common/labs/results及各Lab材料 | 当前目录结构与[根README](../README.md)一致，7个阶段各有README/analysis/exercises/answers/CMake/CUDA源码 | 已交付；未更改其他项目，不依赖业务项目 |
| 八：工程与教学并行、逐阶段实施、每阶段验证/分析/更新进度、明确验收 | [进展](progress.md)按S1→S7记录；独立results快照保留当时源码/文档；S7先完成两版验证再增加三版 | 已按顺序实施；S1–S3的补充推导标为事后复盘，S4起保留采集前假设与干预判据 |
| 九：先查环境，建立地图/架构/框架，完成S1并实际运行后继续 | S1分析保存初始构建、设备、Toolkit、工具、JIT失败及实测；后续按课程逐步扩展 | 已完成，最终交付包含全部必做阶段 |
| 用户追加：记录代码、结果、profile、反推及优化循环 | [分析方法](analysis-workflow.md)规定模型→假设→证据→竞争解释→干预→复测；每阶段analysis具体落地 | 已保留可复核技术推导、决策理由、反例和未解决机制，未只给最终表 |

## 课程实验与验收覆盖

GPU数为CTest用例数，一个用例可能运行多个variant；sanitizer数为实际工具调用数。两者不是全部输入组合的穷举。下表总计249个GPU CTest、92次Sanitizer；额外CLI、CSV、非法输入和覆盖保护检查不混入该计数。

| 阶段 / 程序 | 必做内容与对照 | GPU / Sanitizer | 完整分析与原始证据入口 |
| --- | --- | ---: | --- |
| S1：vector_add、memory_copy | scalar/grid_stride；Runtime D2D参考；Event、带宽、大小工作集、K扫描 | 18 / 8 | [S1](../labs/01_measurement/analysis.md) |
| S2：global_copy | coalesced、stride、offset；gather/scatter/both；float/float2/float4及尾部 | 27 / 16 | [S2](../labs/02_global_memory/analysis.md)；[当前CLI依赖补验](../results/20261023T174203Z_verify_swkdgjrv/verification.txt) |
| S3：transpose、cache_working_set、local_array | naive/tiled/padded；固定请求量的工作集扫描；静态/动态索引与真实spill | 30 / 12 | [S3](../labs/03_memory_hierarchy/analysis.md) |
| S4：reduction、register_pressure | interleaved/mask/sequential/warp多pass；block/items、register budget扫描 | 42 / 8 | [S4](../labs/04_execution_config/analysis.md) |
| S5：integer_division、loop_unroll、branch_flow | signed/unsigned、runtime/constant/shift；unroll/chains；同逻辑任务不同lane布局 | 54 / 20 | [S5](../labs/05_instructions_control/analysis.md) |
| S6：host_transfer、stream_pipeline、concurrent_kernels | pageable/pinned双向；serial/pipeline及提交顺序；同jobs数串行/并发 | 34 / 20 | [S6](../labs/06_async_execution/analysis.md) |
| S7：gemm | Naive→Coalesced→Shared Tiled→Register Tiled→Unrolled；T/R、方形/矩形/尾部及数值对照 | 44 / 8 | [S7](../labs/07_gemm/analysis.md) |

[最终证据清单](../results/20261023T175133Z_final-evidence-audit_sudrgn5o/evidence.json)分别列出每个程序的verify、sanitizer、static记录。审计实际读取退出码和成功摘要，按工具区分`ERROR SUMMARY`与`RACECHECK SUMMARY`；43组原始源码/二进制哈希复核通过。四类Sanitizer都有执行，host_transfer不包含应用kernel，其PTX/SASS标为Not Applicable，不能用二进制里未运行的kernel解释Runtime memcpy。

审计当日源码与当时证据之间另有两层核对：[传递依赖审计](../results/20261023T174602Z_final-source-evidence-audit_27b0s0_j/source_evidence.json)从每个`.cu`递归检查本地头文件，15个程序均与当时最新成功GPU验证快照相同；当时的最终证据审计确认翻译单元与对应static/sanitizer快照一致，最终集成的CMake和构建元数据模板未变化。这些比对不覆盖后来拆分的 Python 入口。生成的build-info不当作手写源码依赖，其实际编译器、flags和架构由各次compile_commands/CSV记录。

初次依赖审计发现Global Copy的公共CLI头文件有后续扩展，保留[失败](../results/20261023T174150Z_final-source-evidence-audit_j8_52eer/status.txt)，针对审计当日源码重新验证后才通过。集成动作因输出管道Broken pipe留下[失败记录](../results/20261023T172526Z_build_7_4e0r_4/status.txt)，完整重跑才作为构建通过依据。汇总审计还分别遇到[CTest摘要格式](../results/20261023T175051Z_final-evidence-audit_4afymd4j/status.txt)和[racecheck摘要格式](../results/20261023T175111Z_final-evidence-audit_fwmr59la/status.txt)假设错误；检查原始日志、按实际格式解析后通过，没有把解析失败当作性能证据。

## 九类教学材料的内容审阅

同一阶段共用的说明覆盖该阶段每个程序；不要求重复复制相同的计时规范。入口：[S1](../labs/01_measurement/README.md)、[S2](../labs/02_global_memory/README.md)、[S3](../labs/03_memory_hierarchy/README.md)、[S4](../labs/04_execution_config/README.md)、[S5](../labs/05_instructions_control/README.md)、[S6](../labs/06_async_execution/README.md)、[S7](../labs/07_gemm/README.md)。

| 必须包含 | 已核对的具体内容 |
| --- | --- |
| Learning Objectives | 每阶段给出BPG13.1章节、概念与完成后要解释/计算的能力；roadmap定义共同和各阶段验收门槛 |
| Theory | lane/warp/grid地址、sector/bank；cache与地址空间；资源/occupancy；整数与浮点语义；stream/event依赖；GEMM布局与同步 |
| Questions | 各指南或练习从定义、必要性、实际执行、性能因素到验证办法及反例；答案要求区分文档、静态与动态证据 |
| Performance Model | S1 12N/8N、AI与时间；S2地址集合/跨度；S3 bank/工作集/spill；S4 pass/资源；S5递推/链/布局；S6字节/chunk/jobs/时间交集；S7 2MNK、复用、padding、容量、AI和误差界 |
| Benchmark Implementation | 每个target有最小基线与对照；参数、派生工作量及改变因素明确；统一CLI、重复与CSV。改变tile/stride等导致的容量/grid变化如实报告 |
| Correctness Verification | 全量CPU/可信参考，浮点有限值与容差，copy精确比较，输出poison，边界/非法参数；CUDA API及launch/sync错误检查，任一variant失败不发布有效CSV |
| Profiling | 每阶段给出static/nsys/ncu入口与观察对象；NCU先查询sections/metrics；PTX与SASS区别、静态计数边界；NSYS活动/间隙/交集；插桩CSV与普通计时分离 |
| Analysis | 每组实测有参数、预测或明确标注的事后模型、原始CSV、统计与机制边界；保留与预测不同的结果及下一轮区分实验，不要求强行得出唯一瓶颈 |
| Exercises | 七阶段各有独立exercises.md与answers.md；要求手算、运行、验证、复测及识别不可推断内容，S7要求独立撰写完整优化报告 |

## 测量、复现和命令验收

旧验收采集时使用 lab.sh，随后曾改为 scripts/*.py；当前独立入口是 commands/<action>/run.py，详见根 README。必须先显式构建，再运行所选动作；日志在对应动作 log/，同动作覆盖，不自动创建快照/哈希/日期目录。run/nsys/ncu 要求明确实验名。历史日志与下面审计描述均属于旧采集，不代表本次入口迁移已通过 GPU 复验。

最终集成[完整日志](../results/20261023T174000Z_build_zlhfx6wo/commands.jsonl)包含15目标构建和16个host/help检查；[编译命令](../results/20261023T174000Z_build_zlhfx6wo/compile_commands.json)逐目标确认C++17、O3、lineinfo、ptxas资源统计及原生sm89。host-only配置实际通过，但没有借此宣称CUDA代码已在无GPU环境运行。

普通结果保留全部raw、独立列统计、GPU/CC/Runtime/driver API、输入、编译及scope；驱动发布号和Toolkit从当次环境日志读取。历史采集的每次results保留argv、退出码、日志、源码快照、哈希、构建配置和二进制。S7数据审计明确区分初版两variant schema与当前五variant schema，没有为通过审计回填旧记录。重度profile计时不作为普通性能基线，缺工具返回77并标Not Tested。

可复算数据与分析入口均在上表。最终GEMM的[21份CSV/67组审计](../results/20261023T174711Z_stage7-data-audit_0aguvimk/status.txt)重新计算统计、单位和地址/工作量/资源模型；S2–S6各自也保留数据审计。源码中的唯一工作集、requested bytes、DRAM流量、静态指令数、动态执行数和理论occupancy始终分开解释。

## 未测、失败与可选项的最终边界

| 项目 | 真实状态 | 为什么不伪装为通过 |
| --- | --- | --- |
| NCU计数器、cache/DRAM/sector、bank冲突、issue/stall、achieved occupancy | [Not Tested：ncu不可用](../results/20261023T174015Z_ncu_havqc_b_/status.txt) | 原目标允许工具缺失时标未测；已提供查询和复现方法，耗时与静态模型不能替代这些计数器 |
| SM7x、其他SM8x、精确Toolkit13.1组合 | Not Tested，只有当前SM89环境实测 | 教材版本与本机编译器版本分开；Toolkit13.4的可用目标列表不包含sm70，不改错误架构强行通过 |
| 当前Toolkit生成PTX的强制JIT | [已测失败：unsupported toolchain](../results/20261023T141840Z_run_2lh0m_0c/014_vector_add.log) | 原生cubin成功不等于PTX JIT兼容，失败没有静默回退或删除 |
| 长期性能稳定性及统一固定加速比 | 未建立 | 多阶段ABBA揭示显著漂移；保留分布、反例与条件化结论，没有锁频或持续负载/功耗证据 |
| SM80+ cp.async | 可选，未实现 | 本次必做传输、stream流水和五版CUDA Core GEMM均已完成，无需用可选新特性替代基础课程 |
| 学习者独立分析能力 | 待独立练习评价 | 代码、测试、模型和示例报告是教学交付，不是个人能力测试结果 |

以上是审计当日的结论；不能由这些历史数字推断拆分后的入口已通过 15 程序全量验证。后续仍需用当前入口逐阶段复现，并在工具/环境具备后检验未测机制。
