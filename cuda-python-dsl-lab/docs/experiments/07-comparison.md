# 07 — 第 10 周：小规模跨 DSL 功能覆盖

当前独立动作：从项目根目录执行 python3 commands/compare/run.py commands/learn/log（也可传多个只读历史输入根目录）。先用 learn all 获得多 DSL 结果；compare 不运行 GPU，只写并覆盖 commands/compare/log/，不重新计算文件哈希或快照。下文数字为历史记录。

本轮优先功能，不做性能比较。读取四份独立记录，使用 [learning_compare.py](../../commands/compare/run.py) 在 CPU 上核对共享输入 ID 和功能状态。

证据：[覆盖汇总](../../runs/20261105-functional-comparison/SUMMARY.md)、[结构化结果及原记录 SHA256](../../runs/20261105-functional-comparison/comparison.json)、[命令与退出码](../../runs/20261105-functional-comparison/run.log)。15 个跨后端算子/shape 组通过记录一致性检查；没有重新运行 GPU。

| 算子 | Triton | Gluon | CuTe | cuTile |
| --- | --- | --- | --- | --- |
| Copy/Add | 小向量功能通过 | 小向量功能通过 | 小向量功能通过 | 小向量功能通过 |
| Transpose | 规则/尾块通过 | 规则/尾块通过 | 规则/尾块通过 | 规则/尾块通过 |
| Row Sum | 通过 | 两个布局通过 | 未实现（首轮非必需） | 通过 |
| Softmax | 通过 | 未实现（扩展） | 未实现（扩展） | 未实现（扩展） |
| FP16 GEMM | 规则/尾块通过 | 未实现（扩展） | 规则/尾块及最小 MMA 通过 | 规则/尾块通过 |

GEMM 三套工具链的输入、输出 dtype 和参考一致；FP32 累计、FP16 输出，不涉及 TF32，未融合 epilogue。输出为独立连续数组，Transpose 不是 stride view。

## 六个比较维度：当前能说什么

| 维度 | 当前依据与边界 |
| --- | --- |
| 编程复杂度 | 源码显示 Triton/cuTile 以逻辑 Tile 为主；Gluon 额外描述分布布局；CuTe 显式划分 Copy/MMA fragments。未以代码行数判定开发效率 |
| Layout 控制 | Gluon 对寄存器/lane/warp 所有权显式控制；CuTe 有坐标→偏移的 layout algebra 及 thread partition，二者不是相同抽象 |
| 线程/内存映射 | Gluon 有 10 组 GPU `%tid.x` probe；CuTe 有索引/partition 验证；cuTile 映射仍由编译器选择，不能从 Tile 大小推出线程数 |
| 生成指令 | [静态报告](06-static.md) 覆盖 Triton/Gluon/CuTe 的 PTX/SASS；CuTe/Triton GEMM 观察到 MMA；cuTile 缺实际机器码 |
| 性能随 shape 变化 | **未测量**。规则与尾块仅验证正确性，首次 JIT/同步时间不是性能排名依据 |
| 架构移植成本 | 本轮仅 SM89；CuTe suite 显式限制该目标；H20/SM90 需重新采集环境、核对 API/同步并验证，未给出移植成本数值 |

## 第 10 周未完成项

- CUDA C++ 基线尚未实现；未借用其他项目的历史结果。
- cuBLAS/明确调用路径的框架 GEMM 性能参考尚未实现。
- 未测试计划中的性能尺度、稳定计时、温度/负载控制、交替配置、独立 component 耗时。
- 未运行完整 Profile，未测 DRAM、occupancy、缓存命中率、bank conflict。

因此本页是**功能覆盖与表达方式对照，不是完整第 10 周性能报告**；学习计划相应复选框不勾选。
