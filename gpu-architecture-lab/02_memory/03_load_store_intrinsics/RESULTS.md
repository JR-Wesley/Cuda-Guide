# 本机验证与结论

> 当前复现入口已迁到 `commands/<action>/run.sh`：先按专题 README 显式 compile，再执行相应动作。下文命令是当前复现写法，历史精确命令仍以原日志为准；本轮未重新采集 GPU 数据。

本报告只陈述历史编译、执行与检查的结果。文本证据已合并为 [sm_89-release.log](results/sm_89-release.log) 的 `HISTORICAL IMPORT` 动作节，日期目录、旧二进制及源码/脚本副本已删除。本次没有真实编译或运行 GPU；新独立动作脚本与 `check_types.py` 的无缓存探针流程不由这些历史结果验证。源码匹配边界及复现配置见 [记录索引](results/README.md)。

日志位于脚本同目录 `commands/<action>/<variant>.log`，同动作同配置重跑覆盖（包括失败），其他动作日志不变。命令、实际参数、原始输出及退出码均记录；检查器的临时步骤文件在退出后清理。不生成自动快照、时间戳归档或 history。已有 `results/` 原始日志和历史报告保持原样，当前动作不写入。

## 1. 环境

| 项目 | 本次记录 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，24 SM，`sm_89` |
| Toolkit | CUDA 13.4，nvcc V13.4.92 |
| Driver API / Runtime API | 13020 / 13040 |
| nvidia-smi | NVIDIA-SMI 615.71.09；KMD 596.21；CUDA UMD 13.2 |
| 编译选项 | C++17、`-O3 -lineinfo`；主程序含 sm_89 cubin 和 compute_89 PTX |
| mapped host memory | `canMapHostMemory=1`，实际执行 |
| Compute Sanitizer | 2026.3.0.0 |

环境原始证据位于 [配置日志](results/sm_89-release.log)：run 节的 `gpu.log`、`compiler.log` 以及 sanitizer 节的 `sanitizer_version.log`，均以文件名标界。本次未强制 PTX JIT，运行成功不等价于额外验证了驱动对嵌入 PTX 的 JIT 支持。

## 2. 可复核的结果

| 验证 | 配置 | 结果 | 单日志中的位置 |
| --- | --- | --- | --- |
| 普通运行 | n=4099 | 415 passed，0 failed，0 skipped | [日志](results/sm_89-release.log)：run / `run.log` |
| 极小边界 | n=1 | 415 passed，0 failed，0 skipped | [日志](results/sm_89-release.log)：run / `edge.log` |
| memcheck | n=257 | 415 passed；`ERROR SUMMARY: 0 errors` | [日志](results/sm_89-release.log)：sanitizer / `memcheck.log` |
| 编译类型探针 | 6 类型 × 4 种写法 | 9 项编译成功，15 项按预期缺少重载；24 项验证通过 | [日志](results/sm_89-release.log)：types / `types.tsv` |
| 静态检查 | 16 个具名 kernel | 16 passed，0 failed | [日志](results/sm_89-release.log)：static / `inspect.log` |

每次主程序的 415 项为：34 类型 × 12 变体 = 408；跨 kernel 更新后 `__ldg` 读取 = 3；mapped 系统内存读取/写入 = 4。mapped 部分始终是 n=1027。每种类型另外验证输入未被修改，但没有给这个附加验证单独增加计数。

普通运行与边界运行共 830 项通过；memcheck 下的 415 项单独记录。它们检验不同规模下的搬运与边界，不是 1245 个彼此不同的算法。

完整配置与源码哈希保留在 [配置日志](results/sm_89-release.log) 对应 action 节的 `metadata.txt` 原文；已删除源码和脚本快照，旧绝对路径不能直接执行。导入前核对各动作 CUDA 源码一致；types 当时未调用其旧静态检查器，具体边界见记录索引。

## 3. 历史编译器的指令映射

下表取自只包含单个目标访问的命名 probe，读取和写入指令分别定位，没有从整个文件里模糊搜索后混合归因。完整文本位于 [配置日志](results/sm_89-release.log) static 节的 `probe.ptx`、`sass.log` 文件标界内。

| C++ 访问（`unsigned`） | PTX | sm_89 SASS |
| --- | --- | --- |
| 普通 `*src` | `ld.global.u32` | `LDG.E` |
| `const __restrict__` 输入的普通读取 | `ld.global.nc.u32` | `LDG.E.CONSTANT` |
| `__ldg` | `ld.global.nc.u32` | `LDG.E.CONSTANT` |
| `__ldca` | `ld.global.ca.u32` | `LDG.E.STRONG.SM` |
| `__ldcg` | `ld.global.cg.u32` | `LDG.E.STRONG.GPU` |
| `__ldcs` | `ld.global.cs.u32` | `LDG.E.EF` |
| `__ldlu` | `ld.global.lu.u32` | `LDG.E.LU` |
| `__ldcv` | `ld.global.cv.u32` | `LDG.E.STRONG.SYS` |
| 普通 `*dst=value` | `st.global.u32` | `STG.E` |
| `__stwb` | `st.global.wb.u32` | `STG.E.STRONG.SM` |
| `__stcg` | `st.global.cg.u32` | `STG.E.STRONG.GPU` |
| `__stcs` | `st.global.cs.u32` | `STG.E.EF` |
| `__stwt` | `st.global.wt.u32` | `STG.E.STRONG.SYS` |

与已有 PTX cache operator 专题相符，显式缓存 intrinsic 生成预期操作符。本机普通读取和 `__ldca` 的 SASS 形式并不完全相同；即使 PTX 定义默认 cache operation 为 `.ca`，也不能要求两个不同源码形态的后端编译结果逐字相同。

`const __restrict__` 的 probe 与显式 `__ldg` 相同，说明在这个足够简单的 kernel 中编译器可以自动选择只读路径。这个结论不能推广为“任何加了 const 的程序都会这样”，也不能当作所有 workload 的性能结论。

还有三项类型观察：

| probe | PTX | SASS | 可得结论 |
| --- | --- | --- | --- |
| `__ldca(float4*)` | `ld.global.ca.v4.f32` | `LDG.E.128.STRONG.SM` | 该受控访问保留为 16-byte vector load |
| `__ldg(__half2*)` | `ld.global.nc.b32` | `LDG.E.CONSTANT` | 两个 half 以 32 位整体搬运 |
| `__stcs(__nv_bfloat162*)` | `st.global.cs.b32` | `STG.E.EF` | 两个 bfloat16 以 32 位整体搬运 |

这些访问宽度不提供跨线程原子性保证。SASS 后缀也不应解释为 C++ acquire/release 或 `__constant__` 声明；详细说明见 [调试文档](DEBUGGING.md)。

## 4. 类型探针为何需要独立记录

`unsigned` 的普通访问和三种 intrinsic 共 4 项成功。其余五类候选的普通访问也成功，但 `__ldg`、`__ldca`、`__stwb` 均被 nvcc 以缺少匹配重载拒绝：`bool`、`long double`、`int1`、`float3`、自定义 `Pair`。合计 9 个正向成功、15 个预期拒绝。

因此，这里的 34 种运行类型是经过实际编译与执行的覆盖列表；文档里的概括性 `T` 不能视为无限制的模板接口。不把 `float3` 的失败归咎于 GPU capability：同一编译命令中，普通 `float3` 赋值可编译，而 intrinsic overload resolution 失败发生在 C++ 类型检查阶段。

## 5. 历史失败与未测项

首次静态检查由于误期待 `__half2` 使用 `.u32` 而失败 1 项，实际 PTX 为正确的 `.b32`。该次 `inspect=1` 的旧记录已被后续完整静态检查取代，旧日志及快照已清理；仅保留 [定位过程](DEBUGGING.md) 作为历史说明，不将它记为成功或用新日志冒充旧失败证据。独立的首次 build 记录也因后续 run/sanitizer 已完整构建同一主程序源码而清理。

本次没有测量耗时、L1/L2 命中率、DRAM bytes、PCIe 流量或 mapped 写穿的物理事务。跨 kernel 和 mapped 场景验证的是在合法依赖下数据搬运正确，不能用来区分“结果正确是因为某个缓存 hint”还是“依赖本来就保证了正确性”。后者正是这里采用的设计前提。

需要性能实验时，应在 [01 Cache Operators](../01_cache_operators/README.md) 已有依赖地址环和 streaming 控制条件下替换封装，再采集新数据。旧 PTX 实验的性能数字没有移植成当前 intrinsic 示例的实测结果。
