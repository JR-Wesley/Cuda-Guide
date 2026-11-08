# `setmaxnreg` 实验结果

> 当前复现入口已迁到 `commands/<action>/run.sh`：先按专题 README 显式 compile，再执行相应动作。下文命令是当前复现写法，历史精确命令仍以原日志为准；本轮未重新采集 GPU 数据。

历史记录日期：2026-10-23（Asia/Hong_Kong），原记录名中的时间为 UTC。本页引用已按配置导入单一日志的历史证据，不代表本次迁移执行了编译或 GPU 验证。语义与操作步骤见 [README.md](README.md)，失败与修正见 [DEBUGGING.md](DEBUGGING.md)。

## 1. 环境与验证边界

| 项目 | 本次值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| Compute capability | 8.9 |
| SM 数 | 24 |
| 驱动版本 | 596.21 |
| `cudaDriverGetVersion` | 13020 |
| CUDA Toolkit / runtime | 13.4.92 / 13040 |
| Compute Sanitizer | 2026.3.0.0 |
| 动态版本编译目标 | `sm_90a`，仅原生 cubin |
| 本机基线编译目标 | `sm_89`，仅原生 cubin |
| 核心选项 | `-std=c++17 -O3 -lineinfo --maxrregcount=128` |

环境来自 [baseline: device.log](results/sm_89-baseline.log)、[baseline: run.log](results/sm_89-baseline.log) 和 [baseline-sanitizer: sanitizer_version.log](results/sm_89-baseline.log)。驱动 API 与 runtime 版本分别记录，没有把二者当作必须相同的值；这里只证明这次原生基线能运行，没有测量新 PTX 的 JIT 兼容性。

## 2. 各动作最新记录索引

| 动作 | 配置日志（按 action 查找原文件名） | 历史结果 |
| --- | --- | --- |
| `run` | 不保留旧源码证据 | **未验证当前版本**；不能用其他动作预检替代 |
| `static` | [static: 134436Z-afGZ1p](results/sm_90a-dynamic.log) | 编译、PTX/SASS 和资源检查通过 |
| `baseline` | [baseline: 134440Z-CeWbsg](results/sm_89-baseline.log) | 24576 项校验通过，完成普通运行计时 |
| `baseline-sanitizer` | [baseline-sanitizer: 134454Z-n2KqF6](results/sm_89-baseline.log) | 预检通过；memcheck / synccheck 各 0 errors |
| `debug` | [debug: 134455Z-ydYwwx](results/sm_90a-dynamic.log) | 8 个编译探针符合预期 |
| `sanitizer` | [sanitizer: 134458Z-9JGpcr](results/sm_90a-dynamic.log) | 动态版本预检返回 77，未执行 sanitizer |

保留规则、参数与核对边界见 [results/README.md](results/README.md)。旧 `run` 的 `main.cu`、`check_static.py` 与当前不同，因此已删除整个 action 证据，不能拿后来的 sanitizer 预检替代当前版本的 run 验证。其余五份记录导入前 CUDA/Python 源码与迁移前版本匹配；本轮改动构建/记录入口、无缓存探针执行及静态检查器的日志参数，没有改 kernel。每个 action 段标有 `HISTORICAL IMPORT` 和旧记录名，原 metadata、命令、输出、`status.tsv` 及静态 PTX/SASS 以文件名分界保留；源码/脚本副本及旧二进制已删除。`static` / `debug` 当时在沙箱中进行，设备查询的非零退出码保留，但不妨碍离线编译。基线与预检当时在可访问设备的环境执行；新共享入口未经过真实编译/运行验证。

## 3. `sm_90a` 静态观察

以下表格来自同一次最终静态构建，不混用 Ada 基线的编译统计。

| 指标 | `fixed128` | `roundtrip` | `redistribute` |
| --- | ---: | ---: | ---: |
| cubin `REG` | 128 | 128 | 128 |
| stack frame 字节 | 56 | 56 | 0 |
| ptxas spill stores 字节 | 96 | 108 | 0 |
| ptxas spill loads 字节 | 108 | 120 | 0 |
| SASS `USETMAXREG` 条数 | 0 | 2 | 4 |
| 最高显式通用寄存器编号 | R125 | R125 | R164 |
| SASS `LDL` 静态条数 | 27 | 30 | 0 |
| SASS `STL` 静态条数 | 24 | 27 | 0 |

证据：[static: build.log](results/sm_90a-dynamic.log)、[static: resources.log](results/sm_90a-dynamic.log)、[static: static_check.log](results/sm_90a-dynamic.log)、[static: 完整 SASS](results/sm_90a-dynamic.log)。

`redistribute` 的四个调整是 `DEALLOC 0x40`、`TRY_ALLOC 0xc0`、`DEALLOC 0x80`、`TRY_ALLOC 0x80`，对应 `64/192/128/128`。最高编号 `R164` 不是“每个线程始终使用 165 个寄存器”，也不是瞬时活跃寄存器数测量；它证明生成代码确实引用了初始 128 范围以上的寄存器。

`ptxas` 的 spill 字节和 SASS opcode 条数是静态编译统计，不能乘一个线程数就当作实际 DRAM 流量。`REG:128` 与 `R164` 同时出现，恰好展示静态初始资源配置与动态分支配额的区别。

相对于第一版重分配实现，最终版 stack/spill 降为 0，说明调整代码结构达到了静态目的。未在 Hopper 上执行，不能把该变化转换成加速比、功耗收益或 occupancy 提升。

## 4. 本机 `sm_89` 基线实测

只编译并执行 `fixed128`。程序明确输出 `BASELINE_ONLY`，没有模拟或执行动态寄存器指令。

```text
blocks=32 threads=256 rounds=3 steps=8 repeats=20
numRegs=128 localSizeBytes=56 binaryVersion=89
PASS fixed128 checked=24576
```

同一二进制完成校验后预热 3 次，再测 5 组、每组 20 次启动，结果为：

| sample | 批量启动平均时间（μs/launch） |
| --- | ---: |
| 0 | 15.923201 |
| 1 | 15.872000 |
| 2 | 16.112000 |
| 3 | 16.076801 |
| 4 | 15.974399 |
| 中位数 | 15.974399 |

原始数据见 [baseline: run.log](results/sm_89-baseline.log)。时间包括每次 kernel 中的三轮工作、屏障及输出，排除分配和结果回传；测量范围是 events 包围的整批启动，可能含提交间隙。这是 Ada 上的固定版本基线，不能与 `sm_90a` 的静态结果组成跨架构加速比。

最终 Ada 基线编译的 spill stores / loads 为 **100 / 112 字节**，与上表 Hopper 的 **96 / 108** 不同，见 [baseline: 基线 build.log](results/sm_89-baseline.log)。比较时必须同时固定源代码、目标架构和编译参数。

## 5. 正确性检查与缺失指标

`baseline-sanitizer` 首先运行 `blocks=1, rounds=1, steps=1, repeats=0`，校验 256 项。随后 memcheck 与 synccheck 分别运行 `blocks=2, rounds=3, steps=2, repeats=0`，各校验 1536 项，均报告 0 errors，见 [baseline-sanitizer: memcheck.log](results/sm_89-baseline.log) 和 [baseline-sanitizer: synccheck.log](results/sm_89-baseline.log)。这些结果仅覆盖基线。

动态版本预检输出 `target_cc=90 dynamic=1`，发现实际设备为 CC 8.9 后返回 **77 / SKIPPED**，见 [sanitizer: preflight.log](results/sm_90a-dynamic.log) 和 [sanitizer: status.tsv](results/sm_90a-dynamic.log)。该 action 段没有 memcheck/synccheck 结果，不能写成“动态版本检查通过”。

| 指标 | 状态与原因 |
| --- | --- |
| 动态版本 GPU 数值正确性 | 未验证：没有支持目标指令的设备 |
| 多轮动态恢复、资源等待是否完成 | 未验证：动态 kernel 未启动 |
| 动态版本 memcheck / synccheck | 未执行：预检返回 77 |
| `setmaxnreg` 延迟、加速比 | 未测量：没有同设备的动态运行样本 |
| CTA 池实际空闲配额、occupancy | 未测量：无对应硬件观测 |
| DRAM 流量、缓存命中率 | 未测量：未采集分析器计数器 |
| PTX JIT | 未验证：本实验部署路径使用原生 cubin |
| Blackwell / family-specific 目标 | 未验证：本次只编译 sm_90a 并运行 sm_89 基线 |

后续拿到匹配设备，按 README 先 compile，再 static → run → sanitizer，核对各动作日志。历史数值只对应原始证据，不用新日志替代。
