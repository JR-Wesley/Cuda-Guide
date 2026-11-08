# 七阶段课程地图

教材固定为 [CUDA C++ Best Practices Guide 13.1](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-c-best-practices-guide/index.html)。章节编号按该归档版本核对。下面是本课程的实验设计，所有预测都需要由当前设备证据检验。

## 学习流程与共同门槛

每次实验按 **Theory → Model → Hypothesis → Experiment → Analysis → Conclusion** 推进。建议每阶段投入 4–8 小时，但以验收为准。先写出可能推翻假设的观测，再运行；“变快了”不构成机制解释。

每个实验必须交付九项：目标与章节、理论、分层问题、可计算模型、最小实现及受控对照、正确性验证、Profiler/静态检查方法、分析、独立练习与答案。计划目录不算已实现。

阶段完成要求：代码已实现；CPU/可信参考和必要边界通过；当前可用环境中的构建、运行通过；测量可复现；解释包含证据、反例和限制；`progress.md` 更新。缺少 GPU 时只能标记“实现完成，GPU 验收 Not Tested”，不能标为整阶段完成。缺少某个 profiler 时可对已测内容验收，但其依赖的机制归因仍须明确保留，不以耗时替代指标。

| 阶段 | 章节 | 核心实验 | 当前状态 |
| --- | --- | --- | --- |
| S1 测量 | Ch.2、4、7、8、9 | Vector Add、Copy、Event、Bandwidth | 已实现，实测见进度 |
| S2 Global Memory | §10.2.1 | 合并/跨步/错位/向量化 Copy | 基础验收完成；计数器归因未测 |
| S3 层次 | §10.2.2–10.2.4，补充 §10.2.7 | Transpose、Padding、Cache、Spill | 基础验收完成；计数器归因未测 |
| S4 执行配置 | Ch.11 | Reduction、Block/Register Sweep | 基础验收完成；含mask干预与occupancy反例 |
| S5 指令与控制流 | Ch.12、13 | 整数除法、常量、Unroll、Divergence | 基础验收完成；含predication与漂移补测 |
| S6 异步 | §10.1、§10.2.3.4、§11.5 | Host Memory、Streams、Overlap | 基础验收完成；含提交顺序干预及时间线交集 |
| S7 综合 | 汇总前述章节 | CUDA Core FP32 GEMM 五个版本 | 基础验收完成；含尾部/数值模型、SASS、ABBA与NSYS反例 |

## S1：Performance Measurement

- 能力：解释 APOD；区分正确性、性能、性能归因；能从字节数和时间复算 GB/s；区分 Event span、host batch、host E2E。
- 理论与模型：FP32 Add 请求 `12N` 字节、`N` FLOP、AI=`1/12` FLOP/B；Copy 请求 `8N` 字节。`B_eff=bytes/(t_us×1000)` GB/s。硬件极限只作条件化下界，不用未测峰值填表。
- 实验：逐元素基线与受限 grid 的 grid-stride；Copy 加 Runtime D2D 参考。一次只变 N、block、iterations 或 variant。Runtime 内部策略不同，属于 API 参考而非单因素内核优化。
- 问题：为什么 host launch 返回不代表完成？重复 K 次平均与单次时间有什么差别？小工作集的有效带宽为何不能证明 DRAM 带宽？
- 证据：CPU 全量比较、原始样本、Median/Min/P95、ptxas/PTX/SASS、Nsight 时间线；计数器不可用则记 Not Tested。
- 验收：小 N、非整 warp/block、大 N 正确；CLI 拒绝非法输入；独立计时与插桩分开；能复算一个 CSV；能说明未测指标。入口：[S1](../labs/01_measurement/README.md)。

## S2：Global Memory Optimization

- 能力：从每个 lane 的地址预测请求覆盖的 32-byte sector 集合；区分 requested bytes、sector bytes 和实际 DRAM 流量。
- 最小实现：连续 scalar Copy；依次仅改变 stride、base offset、元素向量宽度（float/float2/float4）。向量路径必须检查对齐并处理尾部。
- 模型：一次 warp 指令的 `S=|union(floor(byte_address/32))|`，跨 sector 的元素需计入所有覆盖段；`efficiency=requested_unique_bytes/(32S)` 只描述该请求。读写分别建模，跨 warp 缓存复用另算。
- 控制：保持有效元素数和结果定义一致；stride 的更大地址跨度单独报告工作集；offset 必须为两端分配足够边界。不能把重复读取相同地址当作等量独立 Copy。
- 假设：增大 stride 增加 sector 请求；错位可能增加一个 sector，但缓存复用使耗时变化不按 sector 数线性缩放；更宽访问未必更快。
- 证据：查询当前 NCU 的 global request/sector、L1/L2、DRAM 指标；SASS 核验向量宽度；小/大工作集分别记录。
- 验收：对齐/尾部/stride 边界正确；手绘至少三种 warp 地址图；预测与指标比较，解释缓存和工作集混杂；向量化不成立的反例。后续入口：[S2](../labs/02_global_memory/README.md)。

## S3：Memory Hierarchy

- 能力：区分软件地址空间、物理存储和缓存；解释 transpose 两端的访问模式，建立 bank 映射；定位 spill。
- 实验顺序：naive transpose → shared tiled → padded shared；独立 working-set sweep；独立 register/local spill 控制实验。矩形、非 tile 整数倍必须验证。
- 模型：Transpose 请求 `8MN` 字节；共享空间 `tile_rows×pitch×4`；32-bit word 的 bank=`word_index mod 32`。同址广播与同 bank 不同址冲突分开。L2 容量只提供工作集拐点假设。
- 控制：tile、线程映射固定后再加 padding；缓存实验固定访问次数与访问链结构；spill 对照不同时改变算法工作量。
- 问题：padding 为什么可能降低 occupancy？局部数组是否必然在 local memory？大于 L2 是否就等于每次访问 DRAM？
- 证据：global sectors、shared conflicts、cache hit、local load/store；ptxas stack/spill 与最终 SASS；没有计数器时不给命中率结论。
- 验收：无越界/未初始化/同步错误；地址图、bank 图、资源变化都可复算；小矩阵与大矩阵分别解释。入口：[S3](../labs/03_memory_hierarchy/README.md)。

## S4：Execution Configuration

- 能力：计算 occupancy 上界并区别 achieved occupancy；说明 latency hiding、ILP、资源占用之间的权衡。
- 实验：朴素正确 reduction → 分块/warp reduction；在同一算法上扫描 block；另用明确活跃变量数控制 register pressure，保留无 spill 基线。
- 模型：`blocks/SM ≤ min(thread_limit, warp_limit, register_limit, shared_limit, block_slot_limit)`，每项按 block 资源和架构分配粒度计算；`occupancy=active_warps/max_warps`。使用 Runtime occupancy API 校核理论值，不能视作 achieved 值。
- 控制：固定 N 和归约次序策略；寄存器限制与源代码 ILP 分开试验；记录 block 尾效应与实际 grid 波数。
- 问题：更高 occupancy 为何可能更慢？`maxrregcount` 如何引入 spill？少数 block 的资源上界为何不能保证填满 GPU？
- 证据：register/shared、active/eligible warps、调度 stall、local 流量、时间；FP32 归约以高精度 CPU 参考和规模相关容差验证。
- 验收：至少一个“occupancy 增大但收益不增大”的对照或诚实未观察结论；资源模型与编译产物一致；归约尾部和同步检查通过。入口：[S4](../labs/04_execution_config/README.md)。

## S5：Instruction & Control Flow

- 能力：追踪 CUDA C++ → PTX → SASS；区分依赖链延迟与独立指令吞吐；判断 branch 与 predication。
- 实验：runtime 除数/编译期常量/2 的幂；相同有效循环工作量的 unroll；按 warp 一致/warp 内交错的分支。整数 signed/unsigned 与负数语义分别校验。
- 模型：总工作量=`线程数×迭代数×有效操作数`；理想吞吐下界与依赖链下界分开。只有反汇编确认后才统计实际指令，不能将一条 C++ 除法视作一条机器指令。
- 控制：保持结果可观测，避免编译器常量折叠或消去整个循环；分支两侧做可比工作；unroll 因子是唯一变化，记录代码尺寸、register/spill。
- 问题：编译器为什么可以用乘法替换除法？PTX 中有分支是否代表 SASS 也有？活跃 lane 少是否一定降低整个 kernel 吞吐？
- 证据：源码、PTX、SASS、资源统计与指令/分支指标相互对照；性能单独计时。
- 验收：被测指令仍存在；输入边界与符号语义正确；说明 register/指令缓存等可能抵消收益的条件。入口：[S5](../labs/05_instructions_control/README.md)。

## S6：Asynchronous Execution

- 能力：区分 API 对 host 的异步性、stream 顺序与实际引擎重叠；能用时间线证明 pipeline。
- 实验：pageable/pinned 传输 → 单 stream 串行 → 多 stream chunk 流水线；独立 concurrent kernel 对照。只使用本机一张 GPU。
- 模型：串行 `T≈H2D+compute+D2H`；理想稳定流水线每 chunk 周期受最慢阶段约束，但首尾、引擎共享、带宽竞争和 launch 成本均需计入。
- 控制：同一总数据量、同一数学工作；扫 stream 数、chunk size、buffer 数时一次变一个。用 events 表达依赖，记录 pin/unpin 是否在 E2E 范围内。
- 可选：仅 SM80+，global→shared `cp.async` 对照同步复制；编译和运行均检测能力，对齐、尾部、等待和可见性均要证明。SM7x 明确跳过可选项，保留同步版本。
- 问题：Async 后缀是否保证 host 不阻塞？有两条 stream 是否保证并发？理论 pipeline 最大值模型在哪些设备条件下不成立？
- 证据：Nsight Systems 时间线、引擎使用、E2E 吞吐；sanitizer 与结果检查；可选路径 PTX/SASS。
- 验收：无数据依赖竞态；时间线有可量化重叠或明确失败原因；传输/计算/总时间都有定义；fallback 可运行。入口：[S6](../labs/06_async_execution/README.md)。

## S7：Comprehensive Optimization — FP32 GEMM

- 能力：独立完成瓶颈分析、模型、假设、验证、回退与适用条件报告。
- 路径：Naive → Coalesced → Shared Memory Tiled → Register Tiled → Unrolled。每一步保留前一版；不使用 Tensor Core、cuBLAS、CUTLASS 替代实验。
- 模型：`2MNK` 常用近似 FLOP（说明是否把首项加法计入）；数学最低数据量不等于 naive 实际请求量；根据 tile 复用推导 requested bytes、AI、register/shared 和 occupancy 上界。
- 控制：统一布局、alpha/beta 定义、数据类型和输入；覆盖小/大、矩形、非 tile 整数倍。每步分析新增资源、同步与边界代价。
- 证据：FP64 CPU 参考（小规模全量，大规模可信分块验证需说明覆盖）；误差随 K 的容差依据；Event/host E2E；global/shared/compute/occupancy 指标与 SASS。
- 问题：优化改变了哪个限制？为什么小矩阵可能回退？unroll 收益是减少分支还是调度变化？没有计数器能得出哪些有限结论？
- 验收：五个版本正确且可重现；每一步给出预测、实测、差异、反例、适用范围；提交完整报告与未解决问题清单。入口：[S7](../labs/07_gemm/README.md)。

## 向下一阶段交接

保留原始数据和失败记录；更新 `docs/progress.md`；列出已支持的机制解释与尚缺证据。只有完成当前阶段的基础验收，才开始后续代码。课程设计允许优化不产生收益，不允许省略负面结果。
