# 合并访存地址模型：单元测试记录

## 验证范围

本次只验证 [coalescing_model.py](coalescing_model.py) 的 CPU 地址集合模型，不编译或运行 CUDA，不采集 PTX/SASS、Nsight Compute 或 GPU 性能。模型的预设是每条指令最多 32 个活跃 lane、每 lane 自然对齐的 4/8/16B 读取；32B sector 与 128B 地址段仅用于覆盖计算，不模拟缓存状态、wavefront 数或物理事务。

## 命令与原始证据

在工作区 `cuda/` 根目录执行：

```bash
bash -n gpu-architecture-lab/02_memory/04_coalescing/commands/test/run.sh
bash gpu-architecture-lab/02_memory/04_coalescing/commands/test/run.sh
```

脚本语法检查退出码为 0。测试运行环境为 Linux、Python 3.14.9；[原始日志](commands/test/cpu.log) 保存实际 Python 命令、14 个测试用例的 stdout/stderr、各步骤退出码及 `action_exit_code=0`。结果为 **14 tests，OK**。源码与测试分别见 [模型](coalescing_model.py) 和 [测试](test_coalescing_model.py)；此记录不是硬件测量，不引用其他专题的历史 GPU 结果。

## 已验证的模型结论

| 对照 | 单元测试核对结果 |
| --- | --- |
| 满 warp 连续 4/8/16B 读取 | 分别覆盖 4/8/16 个 sector |
| 一条 16B 向量读取 | 16 个 sector |
| 四条同数据归属的 4B 标量读取 | 每条 16 个，总次数 64，去重后 16 个 |
| 四条跨 lane 连续的 4B 标量读取 | 每条 4 个，总次数 16，去重后 16 个；lane 数据归属不同 |
| 4B 读取、32B stride | 32 个 sector |
| 32 个 lane 读取同一 4B 地址 | 1 个 sector，不重复数据只有 4B |
| 连续 3 个 / 仅偶数 lane 读 16B | 分别 2 / 16 个 sector |
| 无活跃 lane | 空集合 |
| 向量基址偏移 0/16/32B | sector 数 16/17/16；128B 地址段数 4/5/5 |
| 标量基址偏移 4B | 连续 128B 数据覆盖 5 个 sector |
| 自然对齐、lane 数及模型参数 | 非法输入抛出 `ValueError`，不发起非法 GPU 访问 |
| 段末端边界与逐字节参照 | 不多计相邻段；多个宽度、偏移、stride 与独立逐字节集合结果相同 |

这些结果证明模型实现符合教学中的地址算术，不证明 C++ 实际生成一条 `LDG.128` 或四条 `LDG.32`，也不证明 L1/L2 命中率、HBM/NVLink 字节数、耗时或加速比。GPU 对照仍未实施。

## 重跑与证据保留

`commands/test/run.sh` 的普通重跑会覆盖 `commands/test/cpu.log`。若修改模型/测试或需要保留本次独立记录，应在重跑前将该日志保存为有意义的命名并维护本文链接；不能让旧结论静默指向不同测试范围的新日志。本次没有修改其他专题的日志、构建入口或 GPU 源码。
