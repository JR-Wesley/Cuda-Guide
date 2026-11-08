# 02 · Memory Explorer

对应 [M1 全局内存](../LEARNING_PATH.md)与 M4 共享内存。已有缓存与 intrinsic 专题和下方微架构规划并存；完整的带宽/stride/bank 扫描仍待实施，不把迁入等同于新实测。

## 已有可执行专题

| 入口 | 内容 | 证据边界 |
| --- | --- | --- |
| [01_cache_operators](01_cache_operators/README.md) | PTX load/store 提示、依赖地址环、流式访存、mapped 系统内存 | 逻辑吞吐不是硬件 DRAM 流量；历史计时与本次运行分开 |
| [02_eviction_priority](02_eviction_priority/README.md) | L1 驱逐提示、热点与干扰、探测循环 | 不承诺锁定缓存；周期数不是命中率，不能直接推广到 L2 |
| [03_load_store_intrinsics](03_load_store_intrinsics/README.md) | C++ intrinsic、类型/对齐、只读生命周期、PTX/SASS 映射 | 功能和编译探针不是性能排名；`__ldg` 需满足只读约束 |

每项的命令、设备条件、`RESULTS.md` 和静态检查以对应 README 为准，没有主题统一构建入口。缓存提示、`volatile` 和内联 PTX 编译器约束不建立线程间同步；先读 [Memory Model](../04_sync/01_memory_model/README.md) 再分析并发访问。

## 合并访存：CPU 模型与单元测试

[04_coalescing](04_coalescing/README.md) 从 `ld.global.v4.u32` 区分指令、lane 地址和内存事务，覆盖标量/向量数据归属、32B sector、对齐、部分活跃 warp，以及 request/wavefront 与缓存下层流量的边界。独立于 Cache Operators：合并访存研究地址布局，缓存算子研究策略。

本专题只有 CPU 地址模型及单元测试，未新增 GPU 可执行实验或性能数据。[test 动作](04_coalescing/commands/README.md) 保存原始日志，[结果说明](04_coalescing/RESULTS.md) 标明验证范围；不增加上方已有 GPU 专题的计数。

## 存储模型与生命周期

先区分地址空间与物理层次：local 是线程私有地址空间，可能承载 spill/栈并访问设备内存层次；shared 是 block 协作存储；global 不代表每次访问 DRAM。寄存器、L1/shared、L2 与 DRAM 的教学模型和事务推导见 [ARCHITECTURE.md](../ARCHITECTURE.md)。

分配和传输学习还应覆盖 `cudaMalloc`、pinned/mapped host memory、managed memory 的生命周期与支持条件；统一地址不等于自动同步或无迁移。已有 mapped 示例是顺序交接，不证明 CPU/GPU 并发原子；有条件的 managed 握手与 shared carveout 见 [Thread Scopes](../04_sync/03_thread_scopes/README.md)，host 传输对照见 [Async Execution](../07_async/01_async_execution/README.md)。完整统一内存迁移性能实验仍为扩展方向。

## 保留的微架构规划

先研究 global memory，再研究 shared memory 和 bank。

| 子题 | 计划中的对照 | 测量前必须先画出 |
| --- | --- | --- |
| `dram_bandwidth` | 连续复制/读写，扫描工作集大小。 | 每线程有用字节与计时范围。 |
| `cache_latency` | 依赖式读取，比较不同工作集与缓存策略。 | 访问链、预期复用、潜在预取影响。 |
| `coalescing` | 已有 [CPU 地址模型与单元测试](04_coalescing/README.md)；GPU stride `1→128`、固定有用工作量的对照仍待实施。 | 每条指令的活跃 lane 地址、宽度与 sector 集合；区分跨指令重复次数和去重覆盖。 |
| `shared_memory` | shared 与 global 访问和同步的最小实验。 | 线程到地址的映射。 |
| `bank_conflict` | `[32][32]` 与 `[32][33]`，其他条件相同。 | 每个 lane 的 shared 地址和 bank 映射。 |

输出区分**有效带宽**与硬件 DRAM 流量；缓存命中、事务和 bank conflict 要待硬件指标采集后才能下结论。[SDK 转置程序](../../cuda-sdk-lab/examples/vector_transpose.cu)与 [当前运行动作说明](../../cuda-sdk-lab/experiments/execution/README.md)可作实现与工具使用示例。原小规模运行记录已不可用，不能继续作为实测证据；当前 README 不是替代记录，也不能代替本专题的 stride、bank 单变量对照或 NCU 硬件指标。[CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)可用于核对访存模型。
