# 测量与证据规范

## 先定义问题

一次实验写清：修改哪个变量、其他条件如何固定、预期影响哪个指标、什么结果会推翻假设。正确性 → 预热 → 重复测量 → 统计 → 归因。编译成功只说明生成了程序，sanitizer 无报告也不构成所有输入上的正确性证明。

## 三个计时范围

| CSV 列 | 起止与单位 | 不包含 | 解释边界 |
| --- | --- | --- | --- |
| `event_span_us_per_op` | start/stop Event 包围 K 次所选操作，总毫秒×1000/K | 分配、CPU reference 与校验；传输是否包含由实验定义 | 设备时间线上的批量平均，可能含活动间空隙和 host 供给不足；不是单 kernel 活跃时间 |
| `host_batch_us_per_op` | CPU steady_clock：record start 前至 stop Event 同步完成后，除 K | 分配、校验、EventElapsedTime 查询；传输范围与该批操作一致 | 包括提交、错误检查、Event 与等待开销 |
| `host_e2e_us` | 已分配缓冲区上一次完整操作，CPU 计时至依赖工作完成 | 分配、初始化、CPU reference、校验、输出 poison | 上传/回读、host 内存类型、重叠方式由实验定义；不是整个进程的耗时 |

S1–S5 与 S7 的 Event 批次使用已驻留输入，不包含 H2D/D2H；E2E 包括 pageable 输入上传、计算或 D2D、输出回读与同步。S4 reduction 的一个 op 包括所有 pass。S6 `host_transfer` 的 op 就是一次指定方向的传输，`stream_pipeline` 是完整 H2D→计算→D2H 流水，`concurrent_kernels` 则是已驻留数据上的一组 jobs；跨 stream 由协调 stream 的 Event 等待所有依赖后结束。S6 的 pinned/pageable、是否重叠及具体 E2E 范围见[异步指南](../labs/06_async_execution/README.md)和 CSV 的 `event_scope` / `e2e_scope`。每次比较先确认 op 定义，不能把通用列名理解成全部实验都只测一个 kernel。

本项目刻意把 Event 字段命名为 span。`--iterations 1` 是单次 Event 包围区间，仍有 Event 测量影响；精确的 kernel 活跃时长请单独从 Nsight 时间线读取，并注明插桩环境。CUDA Event 的行为见 [Runtime Event API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html)。`cudaMemcpyAsync` 对 pageable memory 可能在 host 阻塞，不能凭名称假定异步重叠，参见 [Runtime 同步行为](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html)。

通用收集器先上传、poison 输出并执行一次全量校验；每个样本先 poison、预热、计时 K 次、回读校验，再测独立 E2E 并校验。S1 另有批次之外的初始预热；S6 按传输/流水操作使用专门收集器，仍遵循先校验、再预热/计时、再校验。所有 variant 完成且 CUDA 资源释放成功后才写性能 CSV。失败返回非零，不写有效 CSV。每个 launch 检查 `cudaGetLastError`，同步点检查异步错误；所有返回状态的 CUDA API 都被检查。

S1 短 kernel 默认 K=100、21 个样本、每批预热 5 次，减少 Event 量化影响，但不会消除 dispatch 空隙；其他实验的默认值与实际采样参数见对应指南、命令及 CSV，例如 GEMM 默认 K=5、7 个样本、预热 3 次。可扫 K=1/10/100/1000 观察短任务稳定区间，不能把不断增加 K 当作证明。样本跨 variant 按固定顺序执行，温度/频率随时间变化可能混杂；顺序交换复测采用 A/B/B/A，但这种顺序本身也不能消除严重环境漂移。

## 统计与单位

保存全部 raw 样本；各列独立计算 Min、Median（偶数取中间两项平均）、nearest-rank P95（排序后的第 ceil(0.95m) 项）。m<20 时 P95 等于最大值；必须注明样本量，不能把它当成稳定尾延迟的估计。1–2个样本用于验证/工具插桩的smoke check；GEMM等实验7或11个样本的统计仅描述本次观测，结合顺序交换和漂移边界解释。不同列的 median/min/p95 行不一定来自同一个 raw 样本。

`effective_GBps_at_event_stat = requested_bytes/(event_span_us_per_op×1000)`，十进制 GB/s；MiB 使用 2^20 字节。summary 行的带宽是该行时间换算值，因此 min 时间行给出最大带宽；不要把它称作 min bandwidth。两组独立程序先统一参数和计时定义，再比较。Min 代表观测到的最好样本，不代表理论极限。

## 正确性和内存安全

Stage 1 使用固定 seed=20261023 的确定性有限 FP32 输入，包含正负值；CPU Add 用 double 加法后舍入为 float。Add 接受 `|actual-reference| ≤ 1e-6 + 1e-6×|reference|`，拒绝非有限值；Copy 逐位相同且必须有限。不是 GEMM/reduction 的通用容差，后续须按算法重新建立误差依据。

S1 的 N=1、31/32/33、255/256/257、4099、1048576 覆盖尾部；额外 block=1/31/128/1024 检查非整 warp 和上界。输出先填充 NaN，防止漏写被初值掩盖。非法参数、内存上限和 CSV 覆盖保护必须检查。Compute Sanitizer 的 memcheck/initcheck/racecheck/synccheck 分别执行；S1 没有 shared memory 协作，后两者适用范围有限，不能推论其他同步实验正确。后续阶段各自验证矩形/tile、归约 pass、整数语义、stream 依赖和 GEMM 误差界，覆盖范围见各阶段分析。

## 工作集与模型

S1 的 `working_set_bytes` 是同时活跃的互不重叠输入/输出设备数组总大小，恰与每次 requested bytes 相同；这个等式不适用于所有实验，例如 GEMM 每次重复请求、唯一矩阵数据量及 tiled 复用后的请求各不相同。重复访问相同缓冲区，策略是 **warm/reused buffers**，没有 L2 flush，也没有测量 cache hit。较小 N 的结果可能由缓存与调度开销决定；较大 N 也不能单靠规模宣称所有请求来自 DRAM。

条件模型：若实际数据全来自某层、字节量 Q 已知、带宽上界 B 已知，则该层传输时间下界是 Q/B。若有 F 次运算和吞吐上界 P，则计算下界 F/P。`max(Q/B,F/P)` 是简化模型，未计调度、依赖、尾效应等，不是实测值。Requested bytes 不直接等于 DRAM bytes。

## Profiler 与静态检查

```bash
CUDA_ARCH=sm_89 python3 commands/static/run.py
CUDA_ARCH=sm_89 python3 commands/nsys/run.py vector_add --n 1048576
CUDA_ARCH=sm_89 python3 commands/ncu/run.py memory_copy --n 16777216
```

`static` 用 cuobjdump 提取嵌入的 PTX/SASS 与资源信息，检查 load/store 仍存在；没有 PTX 或解析不符会失败，须调查。原生代码与嵌入 PTX 来自同一个构建；PTX 是虚拟 ISA，SASS 才是该 cubin 的目标机器指令。静态可看到寄存器和指令，但看不到缓存命中与实际流量。

`ncu` 先保存 `--list-sections`、`--query-metrics`，再用当前工具的 basic 集；增补指标时从实际查询选择，记录名称、单位、replay/cache/clock 设置。关注 duration、requested/global sector、L1/L2、DRAM throughput，后续才增加 occupancy/stall。工具缺失返回 77，status 写 `Not Tested`；权限/驱动失败保留原始错误，不修改系统设置。

`nsys` 使用 `--trace=cuda --sample=none --cpuctxsw=none`。关注 Event 边界、kernel 间隔、H2D/D2H、同步和 host 提交；在 GUI 中选放大后的单次活动查看时长。Profiler 内的 CSV 标记 `measurement_context=nsys/ncu`，只作插桩参考，不混入正常基线。[Nsight Compute Triage Guide](https://docs.nvidia.com/nsight-compute/ComputeTriage/) 是归因流程补充材料；当前版本可能含课程范围外特性，本项目不采用这些扩展。

## 可追溯与兼容性

每次动作将 argv、退出码、原始日志、输入参数和必要环境写在 commands/<action>/log/，同动作重跑覆盖。编译参数在共享 build/sm_XX/ 中；没有自动源码/二进制快照或 SHA256。GPU 型号/CC、内存/L2、CUDA Runtime/driver API 版本仍写入 CSV；驱动发布号请先运行 env 动作确认。历史 results/ 中的哈希与快照属于当时记录，不修改。

使用明确 CMake 架构和 Release 配置；官方教材版本不要求本机 Toolkit 必须等于 13.1，本机版本必须如实记录。原生 cubin 成功不代表 PTX JIT 成功。可单独使用 `CUDA_FORCE_PTX_JIT=1` 复现 JIT 条件，保留成功或失败证据；不将其错误静默替换成另一次普通运行。

笔记本功耗、温度、后台负载、调度、时钟都会影响结果。采集入口记录开始时 pstate/温度/功耗；没有锁频或持续功率采样，不据此推断稳定峰值。不能通过擅自重置 GPU、删除历史结果或修改驱动/计数器权限“修复”实验。

## 实验推导与优化决策

每个阶段遵循[分析与决策记录方法](analysis-workflow.md)，将代码模型、预期、工具证据、反例、解释边界及下一轮实验写入analysis.md。已采集数据的补充标为事后复盘；后续预测在采集前写入，保留原始失败和历史快照。
