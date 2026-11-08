# 06 · 从朴素 GEMM 到 Tensor Core

在 M0～M6 的基础机制可解释之后进入 GEMM。计划依次保留 CPU 参考、朴素 CUDA、合并访存、shared tile、register blocking、向量化、双缓冲、异步流水线及 Tensor Core 版本。

每个版本只引入少量可辨识改动，保存源文件或可追溯版本，并在**同尺寸、同精度定义、同正确性标准**下比较时间和吞吐。TFLOPS、有效带宽、实际 DRAM 流量和 occupancy 的口径分开记录；对 Tensor Core 还要注明输入格式、累加格式和允许误差。观察 PTX/SASS 与 NCU 指标后再解释瓶颈。

## 从体系结构模型制定优化假设

用约 `2MNK` 的 FLOP 与所选存储层级的字节数建立 [roofline](../ARCHITECTURE.md)，不要把算法最少字节直接当作实际 DRAM 流量。global 合并访问减少无用事务，shared/register blocking 提高不同层级的复用；更大的 tile 或更多 stage 同时占用寄存器/shared，可能降低驻留并引入 spill。每步都要写出地址布局、资源预算与同步依赖，再用静态和动态证据核对。

边界校验至少覆盖非 tile 整数倍的 M/N/K、不同长宽比与小矩阵。对 FP16/BF16/TF32 等路径写清输入表示、乘法精度、累加格式和容差；不能拿不同数学/精度定义的吞吐直接宣称实现加速。

## 高级特性在本主题内学习

- **Tensor Core**：区分数学操作与具体 MMA 接口，逐项核对架构支持的数据类型、布局、对齐、参与线程集合和完成机制。并非所有矩阵接口都异步，也不能把某代的 tile 或吞吐推广到所有 GPU。
- **异步 tile 流水线**：先读 [Async Barriers](../04_sync/02_async_barriers/README.md) 和 [Thread Scopes](../04_sync/03_thread_scopes/README.md) 的 ready/filled、acquire/commit/wait/release 协议，再设计 GEMM 的填充、消费、复用与排空。它们的功能校验不证明 GEMM 中获得性能收益。
- **warpgroup 寄存器与专用数据移动**：[setmaxnreg](../03_instruction/01_setmaxnreg/README.md) 是静态/动态配额案例，不是已实现的 WGMMA GEMM；SM90 transaction 示例也不覆盖全部 TMA 布局。具体架构、Toolkit、驱动和原生目标条件分别核对，`sm_90a` 不等于普通 `sm_90`。

本目录当前只有规划，没有 GEMM/Tensor Core 代码或测量结果，也没有统一构建入口。已有相关示例只按其 README 的支持范围引用；相关学习门槛见 [学习路径](../LEARNING_PATH.md)，证据要求见 [实验规范](../EXPERIMENT_PROTOCOL.md)。
