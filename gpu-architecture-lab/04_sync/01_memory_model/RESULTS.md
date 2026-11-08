# 实测记录

> 当前复现入口已迁到 `commands/<action>/run.sh`：先按专题 README 显式 compile，再执行相应动作。下文命令是当前复现写法，历史精确命令仍以原日志为准；本轮未重新采集 GPU 数据。

> 记录整理（2026-11-06）：历史证据按真实编译配置合并为 `results/<variant>.log`，各 action 仅保留最新一节；见 [记录索引](results/README.md)。无日期目录或源码/脚本副本。本次未编译、未运行 GPU。下文数值仅对应标有 `HISTORICAL IMPORT` 的原记录；链接进入配置日志后按 `ACTION` 与 `FILE` 标界查找。本轮迁移保留这些历史证据；新日志写入独立动作目录，旧结论不自动套到新日志。

历史采集日期：2026-10-23（UTC+08:00）。原 record 名中的时间使用 UTC，现仅作为配置日志的历史来源标识，不再保留日期目录。

## 环境

| 项目 | 本次记录 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| compute capability / SM 数量 | 8.9 / 24 |
| 编译器 | CUDA Toolkit 13.4，nvcc V13.4.92 |
| 编译选项 | C++17、`-O3 -lineinfo`、sm_89 及 compute_89 PTX |
| 驱动版本 | `nvidia-smi` 查询为 596.21；`cudaDriverGetVersion` 返回 13020 |
| Runtime 版本 | `cudaRuntimeGetVersion` 返回 13040 |
| cooperativeLaunch | 1 |
| managedMemory / concurrentManagedAccess | 1 / 0 |
| pageableMemoryAccess / hostNativeAtomicSupported | 0 / 0 |

设备属性来自下方运行日志；驱动软件版本来自单独执行的 `nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader`。本次没有运行 CPU/GPU 并发 managed-memory 原子协议。system 组使用设备内存，并发访问者仅来自同一 GPU。

## 功能校验

命令：从实验目录执行 `bash commands/run/run.sh`，默认 `ROUNDS=1000`、`SPIN_LIMIT=100000`。

| 测试 | 结果 |
| --- | --- |
| relaxed 原子计数 | 409600，等于期望值 |
| block：6 种协议 × 1000 轮 | 全部通过 |
| device：6 种协议 × 1000 轮 | 全部通过 |
| system_gpu_only：6 种协议 × 1000 轮 | 全部通过 |
| 所有协议的 stale / unexpected / timeout | 全部为 0 |
| 汇总 | `failed=0 skipped=0` |

证据：[运行日志](results/sm_89-release.log)、[命令与配置](results/sm_89-release.log)、[退出码](results/sm_89-release.log)。

有序协议的观察结果与 happens-before 推导一致。弱序组此次未出现旧值，不能由此推出其他架构、编译器或运行条件下也不会出现；本例也没有测量同步延迟或吞吐。

## 工具检查

命令：`ROUNDS=100 bash commands/sanitizer/run.sh`。

- [memcheck](results/sm_89-release.log)：19 项校验通过，`ERROR SUMMARY: 0 errors`。
- [synccheck](results/sm_89-release.log)：19 项校验通过，`ERROR SUMMARY: 0 errors`。
- [退出码记录](results/sm_89-release.log)：编译及两个检查均为 0。

每项工具运行中的 19 项校验指 1 项计数加 18 组消息传递。工具检查不是跨线程内存模型的形式化证明，也没有验证文档中的未定义行为反例。

## 静态指令观察

命令：`bash commands/static/run.sh`。完整证据为 [PTX](results/sm_89-release.log) 和 [SASS](results/sm_89-release.log)。

| 源码配置 | 本次 PTX 中 flag 操作的观察 |
| --- | --- |
| block release/acquire | `st.release.cta.b32` / `ld.acquire.cta.b32` |
| device release/acquire | `st.release.gpu.b32` / `ld.acquire.gpu.b32` |
| system release/acquire | `st.release.sys.b32` / `ld.acquire.sys.b32` |
| seq_cst | 对应 scope 的 `fence.sc` 配合访存指令 |
| release/acquire fence | 可见对应 scope 的 `fence.acq_rel` |
| relaxed 原子 payload | 可见 `st.relaxed` / `ld.relaxed` |

这说明本次编译保留了可观察的顺序和作用域差异。grid barrier 也含有同步指令，不能把整个 kernel 的 fence 数量直接归因于消息协议；PTX/SASS 也不能替代运行数据或源码层面的正确性推导。
