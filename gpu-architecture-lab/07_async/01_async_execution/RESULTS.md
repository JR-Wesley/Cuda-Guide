# 实测结果与证据

> 当前复现入口已迁到 `commands/<action>/run.sh`：先按专题 README 显式 compile，再执行相应动作。下文命令是当前复现写法，历史精确命令仍以原日志为准；本轮未重新采集 GPU 数据。

2026-10-23 在 NVIDIA GeForce RTX 4060 Laptop GPU（`sm_89`）上测试，CUDA 编译器 13.4.92，Nsight Systems 2026.3.2，Compute Sanitizer 2026.3.0。程序报告 `asyncEngineCount=1`、`concurrentKernels=1`。以下数值来自该次历史运行，不是设备的固定性能指标；本次布局迁移没有重新编译或运行 GPU。

## 1. 验证记录

| 运行 | 原始记录 | 结果 |
| --- | --- | --- |
| 最终完整验收 | [run: status.tsv](results/sm_89-release.log) | 编译、流水线、事件/Graph、两种默认流、launch 错误检查全部退出 0 |
| 正常路径 memcheck | [sanitizer: status.tsv](results/sm_89-release.log) | 三个检查全部 `ERROR SUMMARY: 0 errors` |
| 故意错误定位 | [debug: status.tsv](results/sm_89-release.log) | 捕获预期非法写入；保留工具非零退出码，诊断验证通过 |
| Nsight Systems | [profile: trace_analysis.log](results/sm_89-release.log) | 三种配置均采集到 kernel 和 copy 活动 |

保留规则、逐动作参数及源码匹配边界见 [results/README.md](results/README.md)。四份历史记录的 CUDA 源码均与当前版本相同；debug / sanitizer 当时没有后来新增的 `analyze_trace.py`，不能据此宣称当前全套文件已经验证。证据统一保存在 [sm_89-release.log](results/sm_89-release.log)，按 `===== ACTION: <action> =====` 查找，再按原文件名定位 metadata、命令、日志和退出码；每段均标明 `HISTORICAL IMPORT`。时间线分析脚本是在历史 profile 结束后加入并对已有 SQLite 执行的，独立命令与哈希保留于 profile 段；当前入口已自动集成这一步，但新共享入口未经真实编译/运行验证。日期目录、源码/脚本副本及旧可执行文件已删除，唯一一套时间线报告移至 `build/sm_89-release/`。

日志位于脚本同目录 `commands/<action>/<variant>.log`，同动作同配置重跑覆盖（包括失败），其他动作日志不变。命令、实际参数、原始输出及退出码均记录；检查器的临时步骤文件在退出后清理。不生成自动快照、时间戳归档或 history。已有 `results/` 原始日志和历史报告保持原样，当前动作不写入。 新 profile 报告位于 `commands/profile/<variant>-timeline.sqlite` 和 `<variant>-timeline.nsys-rep`；重跑覆盖同配置的新报告，原 build 中的历史报告保持原样，本页旧数值不改链到新报告。

早期运行 `async_execution-20261023T132327Z-5ZmwAJ` 曾因错误码断言过窄失败：实际返回 `cudaErrorInvalidValue`，原代码只接受 `cudaErrorInvalidConfiguration`。修正后重新执行了上表的完整验收；没有把早期失败当作通过。该旧运行已被最新 run 取代，原日志已清理；这里保留历史说明，不以新日志替代旧失败证据。原因见 [DEBUGGING.md](DEBUGGING.md)。

## 2. 流水线：多流不等于加速

输入包含 1,048,613 个整数，分七块，每个配置预热一次、测量五次。所有输出逐元素验证通过，每次 host function 调用次数均为 1。完整结果见 [run: pipeline.log](results/sm_89-release.log)。

单位为 ms；中位数与范围均来自五个样本。

| 配置 | 提交时间中位数 | 完成墙钟中位数 | 完成墙钟范围 | GPU span 中位数 |
| --- | ---: | ---: | ---: | ---: |
| pinned，单流 | 0.339 | 1.512 | 1.461–1.680 | 1.207 |
| pinned，三流 | 0.460 | 1.588 | 1.464–2.740 | 1.192 |
| pageable，三流 | 2.132 | 2.212 | 2.079–3.509 | 1.987 |

本次 pinned 三流没有表现出稳定的墙钟优势。pageable 的提交耗时明显增加，与 staging/host 阻塞的可能性一致，但这里没有单独测量 staging，因此不能把全部差值归因于某一个内部步骤。

GPU span 的起点是 control stream 上的 start event，终点是等待全部 worker 后的 stop event；墙钟还包括 API 提交、CPU 轮询与 host function 完成等待。两者不相等是合理的。单次样本存在主机调度等波动，不应只挑最小值作为普遍结论。

## 3. 时间线：本次未观察到 copy/kernel 重叠

独立的 profile 运行包含每个配置一次预热与两次测量，因此各有 21 个 kernel、42 次拷贝。分析 CUPTI 活动的区间并集，得到：

| 配置 | Kernel 流数 | Kernel 活动并集 ms | Copy 活动并集 ms | Copy/kernel 交集 ms |
| --- | ---: | ---: | ---: | ---: |
| pinned，单流 | 1 | 0.191264 | 2.468509 | 0.000000 |
| pinned，三流 | 3 | 0.191936 | 2.622557 | 0.000000 |
| pageable，三流 | 3 | 0.191296 | 2.554911 | 0.000000 |

三流确实使用了三个 kernel stream，但该次记录没有 copy 与 kernel 活动的时间交集。这个结论来自实际活动时间线，而不是仅凭 event 区间或 API 名字推断。

本负载 kernel 很短，拷贝活动占比更高；提交节奏、设备资源和采集开销也可能影响结果。实验没有逐一隔离原因，因此不声称这块 GPU 无法重叠。下面的独立 kernel 顺序实验确实观察到了 kernel/kernel 重叠，这与当前 copy/kernel 结果并不矛盾。

原始文件：[profile: Nsight 报告](build/sm_89-release/timeline.nsys-rep)、[profile: SQLite](build/sm_89-release/timeline.sqlite)、[profile: 分析命令](results/sm_89-release.log)。可以在 GUI 中检查逐项活动，或执行 `python3 analyze_trace.py <SQLite路径>` 复核。

## 4. Event、默认流和销毁语义

[run: events_graph.log](results/sm_89-release.log) 显示：消费者 D2H 的 copied event 已完成、结果正确时，生产者仍为 `cudaErrorNotReady`。消费者只等待 produced event，因此不必等待生产者之后独立的尾部 kernel。

[run: legacy 日志](results/sm_89-release.log) 与 [run: per-thread 日志](results/sm_89-release.log) 给出以下观察：

| 组合 | 本次观察 |
| --- | --- |
| legacy + blocking worker + implicit/explicit legacy | A → B → C，重叠为 0 |
| legacy + nonblocking worker | 观察到独立 kernel 重叠 |
| PTDS + blocking worker + implicit default | 观察到重叠 |
| PTDS + blocking worker + explicit legacy | A → B → C，重叠为 0 |
| PTDS + nonblocking worker | 两种 B 流均观察到重叠 |

仅对规范要求的顺序设置断言；允许并发的组合即使未观察到重叠也不应判错。日志中的两两交集之和可能大于整体时间跨度，不能当作总重叠墙钟时间。

优先级查询结果是低优先级 `0`、高优先级 `-5`；实验只验证工作完成，不把一次调度顺序解释成抢占保证。

正常运行中，`cudaStreamDestroy` 返回后 done event 查询仍为 NotReady。memcheck 插桩运行中也出现了查询已成功的情形。两者都合法：销毁不提供 host 等待完成保证，而实际工作可能在查询前完成。缓冲区一直保留到 done event 完成。

## 5. Graph：重复提交成本与设置成本分开

捕获图包含 10 个节点，capture/instantiate 本次耗时 3.747 ms。以下每个样本包含 20 次完整链执行，共五个样本，不包含捕获设置成本。原始记录在 [run: events_graph.log](results/sm_89-release.log)。

| 方式 | 提交中位数 ms | 墙钟中位数 ms | 墙钟范围 ms | GPU span 中位数 ms |
| --- | ---: | ---: | ---: | ---: |
| 普通重复提交 | 3.107 | 5.745 | 3.236–14.524 | 5.598 |
| Graph replay | 0.925 | 3.220 | 2.266–3.582 | 3.068 |

本次 Graph 的重复提交耗时更低，完成耗时也较低；但普通路径波动很大，不能据此给出普适的固定加速倍数。设置成本需要由足够多的 replay 摊销；对长 kernel 或大拷贝主导的负载，收益也可能不同。两条路径的运算、数据和同步边界相同，结果均已验证。

## 6. 错误定位与内存检查

- [debug: launch 错误](results/sm_89-release.log)：Peek 与 Get 均返回 InvalidValue；Get 后 Peek 为 Success，之后合法 kernel 可正常运行。
- [debug: 默认异步模式](results/sm_89-release.log)：非法写 kernel 的即时检查为 Success，`cudaDeviceSynchronize` 报 IllegalAddress。
- [debug: CUDA_LAUNCH_BLOCKING=1](results/sm_89-release.log)：即时检查已经报告 IllegalAddress；同步也失败。
- [debug: 故意故障 memcheck](results/sm_89-release.log)：定位 `errors.cu:6` 的四字节全局非法写，地址为 0，thread/block 均为 0。插桩后同步返回 Unknown（999），程序退出 1；脚本保留该退出码，并要求出现非法写诊断才认定这项调试演示成功。

正常路径另行检查：[sanitizer: pipeline](results/sm_89-release.log)、[sanitizer: events/Graph](results/sm_89-release.log)、[sanitizer: ordering](results/sm_89-release.log) 均为零错误。故意故障进程的非零退出不是正常功能测试通过的替代品；两组记录分别保留。

这些 memcheck 结果验证了所执行配置下的内存访问，不能证明所有线程交互均无竞争。跨流正确性还依赖代码中的 event 边和互不重叠的数据区间；不要移除同步后仅靠一次数值正确判断安全。
