# Warp Functions 实验记录

> 当前复现入口已迁到 `commands/<action>/run.sh`：先按专题 README 显式 compile，再执行相应动作。下文命令是当前复现写法，历史精确命令仍以原日志为准；本轮未重新采集 GPU 数据。

> 记录整理（2026-11-06）：历史证据按真实编译配置合并为 `results/<variant>.log`，各 action 仅保留最新一节；见 [记录索引](results/README.md)。无日期目录或源码/脚本副本。本次未编译、未运行 GPU。下文数值仅对应标有 `HISTORICAL IMPORT` 的原记录；链接进入配置日志后按 `ACTION` 与 `FILE` 标界查找。本轮迁移保留这些历史证据；新日志写入独立动作目录，旧结论不自动套到新日志。

历史执行于 2026-10-23（Asia/Hong_Kong），采集时间约为本地 22:13～22:19。原 record 名中的 UTC 时间仅作为日志内历史来源标识；命令、参数、源码哈希、输出和退出码原文已合并到配置日志，源码/脚本/二进制副本与日期目录已删除。以下分析仅对应标记的历史节。

## 环境与结论

| 项目 | 本次记录 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| compute capability / 编译目标 | 8.9 / `sm_89` |
| NVIDIA 驱动发行版本 | 596.21 |
| CUDA Driver API / Runtime | 13020 / 13040 |
| NVCC | CUDA 13.4，V13.4.92 |
| CUB | `CUB_VERSION=300500`，3.5.0 |
| Compute Sanitizer | 2026.3.0.0 |
| CUDA-GDB | 13.4，基于 GNU GDB 16.3 |
| 优化构建 | C++17、`-O3 -lineinfo`，包含 `sm_89` cubin 与 `compute_89` PTX |
| 调试构建 | `-O0 -g -G -arch=sm_89` |
| PTX | `.version 9.4`、`.target sm_89` |

普通 GPU 校验、PTX/SASS 检查、四项 sanitizer、逻辑错误复现和 CUDA-GDB 断点调试均通过。**强制 PTX JIT 失败**，不能把普通运行的成功推广到该驱动对新 PTX 的支持。此次未改动系统驱动或调试权限。

最初沙箱内 `nvidia-smi` 返回 `GPU access blocked by the operating system`；随后在可访问 GPU 的执行环境中完成设备查询和所有 GPU 检查。此访问限制不是 CUDA 算法结果。

## 逐项证据

下列动作均为在专题目录执行 `CUDA_ARCH=sm_89 bash commands/<动作>/run.sh`；完整历史命令存于对应配置日志的同名 `ACTION` 节、各 `.cmd` 的 `FILE` 标界内。

| 动作 | 状态 | 原始记录 |
| --- | --- | --- |
| build（旧独立记录已清理） | 当时编译退出 0，无编译告警；后续动作均重新构建 | `warp_functions-20261023T141304Z-LumyLd` 已由后续完整构建覆盖，旧日志已清理；此行仅为历史说明 |
| run | 156 组集合测试、120 组 shuffle、CUB/libcu++、共享内存交换通过；460,720 个字段校验 | [run.log](results/sm_89-release.log)、[status.tsv](results/sm_89-release.log) |
| static | 27 项按 kernel 的指令模式检查通过 | [static_check.log](results/sm_89-release.log)、[PTX](results/sm_89-release.log)、[SASS](results/sm_89-release.log) |
| sanitizer | memcheck、racecheck、synccheck、initcheck 均退出 0 | [status.tsv](results/sm_89-release.log) |
| debug | 9 个主机契约案例符合预期；GPU 的错误 width 产生预期 16 个逻辑差异，修复后全部正确 | [契约诊断](results/sm_89-release.log)、[GPU 对照](results/sm_89-release.log) |
| gdb | 命中 `width_debug_kernel`，观察 32 个线程，继续校验并正常退出 | [gdb.log](results/sm_89-debug.log)、[status.tsv](results/sm_89-debug.log) |
| jit | **退出 1，未通过**：`cudaErrorUnsupportedPtxVersion` | [force_ptx_jit.log](results/sm_89-release.log)、[status.tsv](results/sm_89-release.log) |

sanitizer 每个工具执行 quick 集合矩阵（48 组）和完整的 shuffle、库与共享内存矩阵，各验证 74,560 个字段：

- [memcheck](results/sm_89-release.log)：0 errors。
- [racecheck](results/sm_89-release.log)：0 hazards，0 errors，0 warnings。
- [synccheck](results/sm_89-release.log)：0 errors。
- [initcheck](results/sm_89-release.log)：0 errors。

## 实测能支持什么

1. **mask 与寻址**：不完整物理 warp、输入尾部、二维索引与稀疏成员集合的结果均与 CPU 对照一致。XOR 向前分组读取和向后分组返回自身的边界得到覆盖。没有把 `__activemask()` 快照等于预期集合列为断言。
2. **位比较**：`+0.0f` 和 `-0.0f` 的 match 结果按 bit pattern 分组，`match_all` 的 predicate 和 mask 同时校验。整数规约分别验证 add/min/max/and/or/xor。
3. **库与手写路径**：CUB 的 8-lane inclusive scan、32-lane reduction，libcu++ 的广播和 match 都实际执行通过，没有可选 API 跳过。
4. **共享内存**：组内反转使用互不相交的半 warp mask；跨半反转使用 full mask。1/2/101 轮的结果通过，racecheck 未观察到 shared-memory hazards。正确性仍依赖覆盖写者/读者的 barrier 和读后复用 barrier，而非零告警本身。
5. **调试对照**：材料中默认 width 的调用本身合法，错误在于它不符合“两个半 warp 各自广播”的意图。修复所用的 `src=0,width=16` 得到全部正确结果。mask 违规示例只在主机模型中诊断，不声称已在 GPU 上触发这些未定义行为。

## 静态观察

`collectives_kernel` 中确认 `activemask.b32`、三类 `vote.sync`、两类 `match.*.sync` 和六类 `redux.sync` 均存在；对应 SASS 中可见 VOTE、MATCH、REDUX 和 SHFL。`shuffle_kernel` 包含 idx/up/down/bfly 四种 PTX shuffle。

`shared_exchange_kernel` 的 PTX 中有 `bar.warp.sync`、`st.shared`、`ld.shared`，SASS 中有 WARPSYNC、STS、LDS。此次 PTX 的 barrier 文本有 30 处，SASS 的 WARPSYNC 文本有 1 处；编译器展开和控制流 lowering 后不应期待一条源码 barrier 对应一条同名机器指令。这里仅记录两级产物的不同静态形式，不由计数推出动态 barrier 次数或同步代价。

## PTX JIT 的实际失败

强制 `CUDA_FORCE_PTX_JIT=1` 后，在第一次集合 kernel 启动的错误检查中保留了：

```text
FAIL cudaGetLastError() line=30 cudaErrorUnsupportedPtxVersion: the provided PTX was compiled with an unsupported toolchain.
```

普通构建已携带匹配的 `sm_89` 机器码并成功运行；强制 JIT 则必须处理 Toolkit 13.4 的 PTX 9.4。结合驱动报告的 CUDA API 13.2 和上述错误，此次失败定位于 PTX 工具链兼容性。当前环境可复现普通 `run` 路径，不能宣称 `jit` 已通过。若后续另行安装匹配工具链或更新驱动，需要重新采集 JIT 结果；本任务没有进行这些系统变更。

## 未测量与适用范围

本例没有计时，也未采集性能计数器；指令延迟、吞吐率、缓存命中率、DRAM 流量均未测量。没有对 CUB 与 intrinsic 做性能排名。浮点规约、其他数据类型、其他 GPU 架构、多 GPU、跨 warp 内存通信及非法 collective 的 GPU 故障表现均未验证。测试覆盖与语义推导见 [README](README.md)，调试方法与工具边界见 [DEBUGGING](DEBUGGING.md)。
