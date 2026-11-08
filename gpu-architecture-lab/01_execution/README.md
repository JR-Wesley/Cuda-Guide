# 01 · Execution / Warp / Occupancy

对应 [M2 Warp Explorer 与 M3 Latency Hiding Explorer](../LEARNING_PATH.md)，同时承接 CUDA 编程模型入门。下面的独立性能扫描仍是规划；已有功能专题通过链接复用，不宣称此次生成了新 kernel 或测量。

## Host/device 与 kernel 启动

Host 负责资源生命周期和工作提交，device 执行 kernel。典型流程是：分配并初始化 → 传输输入 → 按 grid/block 启动 → 等待所需完成点 → 校验与释放。kernel launch 通常相对 host 异步，提交后不能立即读取尚未完成的结果或释放输入；stream/event 的依赖和默认流规则见 [Async Execution](../07_async/01_async_execution/README.md)。

`kernel<<<grid, block, dynamic_shared_bytes, stream>>>(...)` 指定逻辑线程布局、动态 shared 大小与提交流，而不是指定某个 SM。先证明每个输出由正确线程处理，再优化布局：

- 一维 `i = blockIdx.x * blockDim.x + threadIdx.x` 配合 `i < n`；grid-stride loop 的步长为全 grid 的线程数，大输入要防索引乘法溢出。
- 多维 block 先线性化：`t = x + blockDim.x * (y + blockDim.y * z)`，再求 warp/lane。物理尾 warp 与数据尾部是两种边界。
- 普通 block 可独立、任意次序调度；不能用 block 编号推断驻留次序，不能让一个超额 grid 的所有 block 忙等来实现全局 barrier。

## 从逻辑线程映射到执行资源

Block 在 SM 上占用线程、寄存器与 shared 等资源；SM 可驻留多个 block，warp 调度器在就绪指令之间安排执行。thread 不永久绑定某个算术单元，stream 不拥有固定 SM；block 数等于 SM 数也不保证一一对应。分歧、谓词化、独立线程调度和资源限制的模型见 [ARCHITECTURE.md](../ARCHITECTURE.md)。

已有 [Warp Functions](../04_sync/04_warp_functions/README.md) 提供 mask、vote/match/reduce/shuffle 与 CUB 的数值/静态检查，应先学参与契约；它没有独立的延迟或吞吐测量。跨 block 协作与 cluster 的条件示例见 [Thread Scopes](../04_sync/03_thread_scopes/README.md)：cooperative grid 与 cluster 是不同机制，cluster 要求 CC 9.0+ 及合法配置，不能由历史 `sm_89` 记录宣称支持。

## 保留的单变量实验规划

先验证 grid/block/warp/lane 索引和边界，再研究分支、warp 内通信与驻留资源。

| 子题 | 单变量对照 | 要解释的机制 |
| --- | --- | --- |
| `thread_block` | block size 与 grid 覆盖方式。 | 每个线程处理哪些元素，边界线程是否正确。 |
| `warp` | warp 内 shuffle、ballot、active mask。 | lane 间通信与参与线程集合。 |
| `divergence` | warp 内交错条件与按 warp 分组条件。 | 分支路径、活跃 lane、谓词与控制流。 |
| `occupancy` | 固定工作量，改变 block 大小或寄存器压力。 | 驻留 block/warp、eligible warp 与吞吐之间的关系。 |

每次把正确性、资源用量、PTX/SASS 和动态指标分开记录。**高 occupancy 不保证高性能**；只有在同工作量、同架构和控制变量下才讨论因果。现有 [向量和转置源码](../../cuda-sdk-lab/examples/vector_transpose.cu)展示线程索引，但没有上述独立扫描。
