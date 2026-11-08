# 05 · 基础并行原语

先完成 [M6 Reduction](../LEARNING_PATH.md)，再扩展 `vector_add`、`scan` 与 `transpose`。归约将依次比较原子累加、朴素分层归约、共享内存树、warp shuffle，以及 warp + block + multi-block 组合。

## 已有素材与规划边界

[Warp Functions](../04_sync/04_warp_functions/README.md) 已包含 warp reduce/scan、shuffle 与 CUB 对照，可学习 mask、源 lane、逻辑分组和临时存储契约；这不是完整 block/grid reduction、scan 优化系列。先结合 [执行模型](../01_execution/README.md)、[访存](../02_memory/README.md) 和 [同步](../04_sync/README.md) 推导正确性，再组合算法。

本目录仍是综合原语规划，没有新增统一构建脚本。实现时覆盖非整 warp/block、单元素与多 block 边界；多阶段归约明确中间缓冲区、kernel 依赖和最终结果由谁写入。shuffle 不会自动同步 shared memory，多 block 也不能靠普通 block barrier 协作。

## 对照与评价

每一版保留相同输入、数值误差标准和原始记录；对 floating point reduction，结果顺序不同可能导致舍入差异，正确性判断不能简单要求逐位相等。分析每次改动是否真的减少全局访存、同步、争用或分支成本。

现有 [向量加法与两种转置](../../cuda-sdk-lab/examples/vector_transpose.cu)可作为入门对照；这些程序不算本目录新生成的实验。当前没有本目录独立的 reduction/scan 性能系列结果。使用 [roofline 与延迟/吞吐模型](../ARCHITECTURE.md) 区分数据搬运、依赖链、争用和提交开销，并按 [实验规范](../EXPERIMENT_PROTOCOL.md) 先验证再计时。
