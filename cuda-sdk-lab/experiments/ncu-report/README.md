# ncu 报告阅读：不重新运行程序

## 目标

采集报告与阅读报告是两个动作。本节读取已有基本报告，解释启动配置与吞吐字段；不会编译、重新启动 kernel，也不会自动触发 ncu 采集。

前置：在 [ncu](../ncu/README.md) 成功生成 `log/basic.ncu-repz`。若重跑采集，本节旧输出也应显式重跑更新，不能把旧读数挂在新报告上。

## 1. 底层命令与入口

当前目录：`cuda-sdk-lab/experiments/ncu-report/`。

```bash
ncu --import ../ncu/log/basic.ncu-repz --page details
bash run.sh
```

第一条展示底层命令，第二条是记录同一动作的替代入口，不必重复执行。`--import` 读取文件；`--page details` 展示已有 section 的易读摘要。入门不默认导出包含数百设备属性和指标列的 raw CSV；需要完整指标时可自行理解 `--page raw --csv` 后使用，但本轮报告以 details 为准。

缺报告时提示先执行相应前置动作并返回非零。只覆盖本动作同名日志，不覆盖输入报告。

## 2. 本次观察

[命令](log/details.sh)、[原始输出](log/details.log)、[状态](log/status.tsv) 均对应当前基本报告，退出 0。

| 真实字段 | 本次值 | 如何理解 |
| --- | --- | --- |
| Block Size / Grid Size | 256 / 2 | 两块，每块 256 个线程，与 `ceil(257/256)` 一致。 |
| Threads | 512 | 启动线程数，不是有效数组元素数。 |
| # SMs | 24 | 设备执行单元数量；两块不能让所有 SM 都有工作。 |
| Registers Per Thread | 16 | 本次分析器报告口径；不把它混作编译日志里的全部静态字段。 |
| Duration | 1.76 us | 分析器报告的 kernel 时间；不是程序端到端时间或 execution 的 event 批平均值。 |
| Memory / Compute Throughput | 1.74% / 0.09% | 相对于工具定义峰值的指标，小网格不适合据此判断峰值能力。 |

这里报告的每线程 16 个寄存器与 [构建日志](../build/log/build.log) 中 `vector_add` 的 12 个不同；差异原因尚未定位，不能据此断言加载选择、JIT 或性能变化。工具提示网格太小，只产生约 `0.01 waves/SM`。本课程故意如此以方便手算；不根据 `Est. Speedup` 提示宣称存在已经实现的提速。

## 3. 练习

解释 `512 threads`、`257 elements` 和“两次程序启动但只采集一次”的区别。先读数据形状与采集范围，再读指标，避免把名字相似的时间相加。
