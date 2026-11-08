# 07 · Streams / Async Copy / Pipeline

本主题连接 host 提交、stream/event 依赖、Graph 与 kernel 内流水线。host 侧异步基础可在理解 launch 和生命周期后学习；深入 kernel 内 tile 流水线则晚于基础访存、同步与 GEMM tile 模型。

## 已有可执行入口

[01_async_execution](01_async_execution/README.md) 已有 pinned/pageable、单流/多流流水线、默认流两种模式、event、host function、优先级、Graph capture/replay 和错误定位。运行、sanitizer、debug、profile 各动作以专题 README 为准；没有统一构建入口。

API 已返回、GPU 操作已完成、不同活动真实重叠是三个问题。event 只标记它之前的工作，不保证整个 stream 或其后 host function 完成；资源释放必须覆盖全部使用者。Graph 已有线性 capture 示例，但跨流 capture、update、动态节点和多 GPU 不在该示例覆盖范围，不能从“有 Graph”推断完整高级功能已实现。

## 异步的层次与关联专题

| 层次 | 已有材料 | 需要验证 |
| --- | --- | --- |
| Host 与 device | [Async Execution](01_async_execution/README.md) | 提交耗时、GPU span、端到端完成、缓冲区生命周期 |
| Stream / Graph 依赖图 | 同上 | 默认流语义、跨流 event、capture/instantiate 成本与 replay；时间线上的真实重叠 |
| Kernel 内 global→shared | [Async Barriers](../04_sync/02_async_barriers/README.md) | 拷贝完成绑定、phase、参与者、可读与可覆盖分开 |
| 多阶段 pipeline | [Thread Scopes](../04_sync/03_thread_scopes/README.md) | producer acquire/commit、consumer wait/release、预填与尾部排空 |
| 高级数据移动与矩阵流水线 | [GEMM](../06_gemm/README.md)、[寄存器配额](../03_instruction/01_setmaxnreg/README.md) | 架构专用条件与资源竞争；GEMM/Tensor Core 仍是规划 |

常见 `cp.async` 硬件路径要求 SM80+，SM90 transaction、cluster、架构专用寄存器/矩阵指令另有条件。高层 API 的可用或软件回退不等于所期望的硬件路径生成，更不等于本机实际重叠。完整层次模型见 [ARCHITECTURE.md](../ARCHITECTURE.md)。

## 保留的性能扩展规划

继续比较同一 stream 的顺序执行、多 stream 的拷贝/计算重叠、异步 global→shared 搬运，以及 load-next / compute-current 的软件流水线。已有 host 侧历史记录可作方法起点；kernel 内功能示例并没有因此完成 tile 大小、stage 数、资源用量的系统性能扫描。

通过 Nsight Systems 时间线先确认**有没有重叠**，再研究 kernel 内的等待、barrier 和指令；只看到异步 API 调用不代表实际重叠。保留同工作量的顺序基线、正确性和控制变量。目标架构的具体支持应以本机设备属性、Toolkit 版本及 [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)为准。

现有 [SDK 实验](../../cuda-sdk-lab/README.md)有 CUDA event 计时和 Nsight Systems 报告；新增导航中的多流/Graph 对照来自迁入专题，而不是此次文档整合的新实测。按 [实验规范](../EXPERIMENT_PROTOCOL.md) 区分普通基线与 profiler 扰动，保留原始记录和未测项目。
