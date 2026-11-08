# Compute Sanitizer：一次只观察一种故障

## 背景与前置知识

GPU 程序没有崩溃，不代表它没有错误：越界写可能暂时没被发现，两个线程同时写一个位置可能碰巧得到想要的值。Compute Sanitizer 是运行时检查工具，不是性能分析器。本专题用最小错误输入，学习“程序意图、工具报告、退出码”三者的区别。

先认识：`cudaMalloc` 分配 GPU 全局内存但不初始化数据；`__shared__` 是同一 block 内共享的存储；线程间没有约定顺序的冲突访问是竞争；warp 通常含 32 个 lane（线程编号）。屏障用于约束参与线程的执行和内存可见性，不能随便写掩码。

`cases.cu` 是唯一故障源码，保留四个 kernel 和简单 `main`。每次程序调用只选择一个 kernel。**旧 `sanitizer/run.sh` 四检查器万能入口已删除，本目录仅放源码和导航**，不再要求在这里执行 `bash run.sh`。

## 1. 先预测错误，再构建

| 独立动作 | 最小输入与代码 | 要找的特定诊断 |
| --- | --- | --- |
| [sanitizer-memcheck](../sanitizer-memcheck/README.md) | 分配一个 `int` 却写 `p[1]`，1 线程 | `Invalid __global__ write` |
| [sanitizer-racecheck](../sanitizer-racecheck/README.md) | 2 线程无序写同一个 shared `slot` | `Race reported` 或 `WAW hazard` |
| [sanitizer-initcheck](../sanitizer-initcheck/README.md) | 1 线程读取尚未初始化的 `in[0]` | `Uninitialized __global__ memory read` |
| [sanitizer-synccheck](../sanitizer-synccheck/README.md) | lane 0–16 执行只列出 0–15 的 `__syncwarp(0xffffu)` | `Barrier error detected. Invalid arguments` |

所有输入都比大数组更容易逐个地址核对。每个 `int` 为 4 字节；越界例中只有 `[p,p+4字节)` 属于分配区域。racecheck 例中，写后的 `__syncthreads()` 不会让之前两个无序写变成有序写。

先到 [sanitizer-build](../sanitizer-build/README.md) 看底层 `nvcc` 命令，再执行 `CUDA_ARCH=sm_89 bash run.sh`。只生成一个共享 debug 产物 `build/sm_89/sanitizer/cases`。debug 构建用 `-G` 保留设备调试信息并关闭常规设备优化，不适合性能比较。不复制四份源码或构建目录。

## 2. 运行一个检查器

手动进入上表任一动作目录。先看其 README 的底层 `compute-sanitizer` 命令，再用记录入口，例如在 `experiments/sanitizer-memcheck/`：

```bash
CUDA_ARCH=sm_89 LD_LIBRARY_PATH=/opt/cuda/lib64 bash run.sh
```

`CUDA_ARCH` 必须显式设置并匹配产物；`LD_LIBRARY_PATH` 用于找到共享 Runtime 库。`COMPUTE_SANITIZER` 可指定工具路径。每个脚本只执行自己的检查器，不切目录、不编译，不调用旧入口；缺文件提示先构建。

`--tool` 选择检查器，`--error-exitcode 86` 让发现错误时使用可区分的非零退出码。脚本捕获但**不改写**此原始返回值，然后分别记录三项断言：

1. `expected_exit`：原始返回值是否为 86。
2. `diagnostic`：是否包含本例的特定诊断，不能用任意报错代替。
3. `summary`：错误汇总是否非零（racecheck 使用自己的 `RACECHECK SUMMARY`）。

只有三项都退出 0，动作脚本才退出 0，含义是“预期故障已检出”，不是“CUDA 程序正确”。原始检查器行始终为 86。缺工具、加载失败、错误未检出或诊断不匹配都会使动作失败。

## 3. 本次真实证据

2026-11-08，RTX 4060 Laptop、CC 8.9、驱动 596.21、nvcc 13.4.92、Compute Sanitizer 2026.3.0.0。共享 debug 构建与 [SASS 导出](../sanitizer-static/README.md) 均退出 0。运行环境为 `LD_LIBRARY_PATH=/opt/cuda/lib64`。

| 动作 | 检查器原始退出码 | 实际汇总 | 三项断言 |
| --- | --- | --- | --- |
| [memcheck](../sanitizer-memcheck/log/memcheck.log) | 86 | 2 errors：越界写及同步时 API error 999 | 均为 0 |
| [racecheck](../sanitizer-racecheck/log/racecheck.log) | 86 | 1 hazard，1 error，0 warnings | 均为 0 |
| [initcheck](../sanitizer-initcheck/log/initcheck.log) | 86 | 1 error | 均为 0 |
| [synccheck](../sanitizer-synccheck/log/synccheck.log) | 86 | 17 errors | 均为 0 |

不要要求故障数在不同工具版本下恒定；这里验证的是故障类别与非零汇总。memcheck 的同步错误原样保留，不以“预期故障”为由关闭 API 检查。`cases.cu` 仅补充了第二次分配和正常清理的错误检查，没有改故障算法。

## 4. 日志和静态观察

每个动作 `source ../record.sh`，仅初始化本动作 `log/status.tsv`。实际命令写为 `log/<label>.sh`，原始合并输出为同名 `.log`。重跑覆盖同名记录，不创建 history 或快照；以本次状态表为准，不把遗留文件当成本次执行。日志不是集中放在本源码目录。

可单独在 [sanitizer-static](../sanitizer-static/README.md) 导出同一二进制的 SASS（GPU 机器指令文本），观察错误操作仍存在；这不能代替动态故障检测。正常程序的检查另外在 [validation：memcheck](../validation/README.md) 和 [validation-racecheck](../validation-racecheck/README.md)，不能用这些故障程序当作正确性基线。

## 5. 练习

一次只改一个故障：先预测把越界的 `p[1]` 改成 `p[0]` 会怎样，再显式重编译并只运行 memcheck。预期检测脚本应因“不再发现故障”而失败，这正是在验证负例断言。练习完成后恢复故意错误、重编译并重新采集；改源码或覆盖日志后，上表不再代表新记录，须同步更新说明。

参考：[Compute Sanitizer 文档](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/)。
