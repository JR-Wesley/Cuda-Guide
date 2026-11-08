# ncu 内存表格：从 257 个元素解释 sector

## 背景

基本采集回答启动与总体吞吐，本动作只选择 `MemoryWorkloadAnalysis_Tables`，学习缓存层级和访存计数。输入与 [普通运行](../execution/README.md) 相同：向量加法、257 个 float、一次计时启动；不换算法或做大型带宽基准。

**sector** 是这里按 32 字节统计的数据单位，不等于一个线程，也不等于完整 cache line。先算算法数据量：两次读、一次写，共 `257×4×3=3084` 字节；它不能直接当作实测 DRAM 流量。

## 1. 独立采集命令

当前目录：`cuda-sdk-lab/experiments/ncu-memory/`，先在 [build](../build/README.md) 显式编译并运行正确性基线。

```bash
ncu --export ./log/memory --force-overwrite \
  --kernel-name vector_add --launch-skip 1 --launch-count 1 \
  --section MemoryWorkloadAnalysis_Tables --page details --print-details all \
  ../../build/sm_89/vector_transpose --case vector --n 257 --repeat 1
```

`--section` 只选内存主题；`--page details --print-details all` 要求此次采集同时打印 section 表格，不是再运行一个 import 动作。过滤、跳过预热、报告覆盖等语义见 [基本采集](../ncu/README.md)。脚本替代入口：

```bash
CUDA_ARCH=sm_89 bash run.sh
```

脚本先创建 `log/`，复用共享二进制；输出是本机 ncu 2026.3 的 `.ncu-repz` 格式。只覆盖自己的报告、命令与日志，不采集所有 section。

## 2. 本次结果及解释

[命令](log/profile.sh)、[完整输出](log/profile.log)、[状态](log/status.tsv)、[报告](log/memory.ncu-repz)。本次退出 0，`correct=yes`，checksum `3508.375000`，工具为一个目标 kernel 进行了 **32 passes**，因此程序 event 耗时不作普通基线。

| 观察 | 本次值 | 解释边界 |
| --- | --- | --- |
| L1TEX Sectors | 99 | 日志还列出 66 个读 sector、33 个写 sector。 |
| Average Bytes Per Sector，global loads/stores | 31.15 | 对齐数组长度不是 32 字节倍数，最后一个 sector 不满；不能看到低于 32 就断言源码用了 stride。 |
| L2 Hit Rate，loads/stores | 53.26% / 100% | 当前工具重放与缓存条件下的观测，不泛化为所有输入或普通运行缓存命中率。 |
| DRAM Throughput | 1.19% | 小网格结果，不是设备峰值带宽测试。 |

手算最后一项边界：每个输入数组 `257×4=1028` 字节，覆盖 `ceil(1028/32)=33` 个 sector；两个读数组共 66 个，输出 33 个，总计 99。`1028/33≈31.15` 字节/sector 与本次表格相符。这个对照帮助解释 L1TEX 数据，不能据此将所有缓存层的计数强行等同；DRAM 表还可能受其他活动与测量条件影响。

工具提示可选 `inst_executed` 指标缺失，原始 warning 保留；本动作未为消除提示而扩大采集。自动规则给出的“可能有 stride”只是候选解释，当前顺序下标和尾部不满已经提供更具体的分析。

## 3. 练习

只在纸上把长度改成 256，预测理想覆盖将变为两个读数组 64 个 sector、一个写数组 32 个 sector。若要实测，保持源码和编译条件不变、显式修改输入，并另命名保留需要对照的记录；不要把这个预测当作已测结果。
