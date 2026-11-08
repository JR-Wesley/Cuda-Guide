# `setmaxnreg` 调试过程与故障定位

> 当前复现入口已迁到 `commands/<action>/run.sh`：先按专题 README 显式 compile，再执行相应动作。下文命令是当前复现写法，历史精确命令仍以原日志为准；本轮未重新采集 GPU 数据。

本记录区分实际遇到的问题和未执行的风险分析。历史时间为 2026-10-23，用户时区 Asia/Hong_Kong；原记录名使用 UTC。可保留证据已按配置合入唯一日志，各 action 的 `HISTORICAL IMPORT` 段保留旧记录名及原文件内容；日期目录、源码/脚本副本和旧二进制已清理。旧 `run` 源码不匹配，证据已删除，当前动态 run 未验证。以下早期失败仅作历史说明，不用不同条件日志替代。保留清单见 [results/README.md](results/README.md)，实现见 [main.cu](main.cu)。

## 1. 先确认能执行哪一部分

沙箱内 `nvidia-smi` 最初返回：

```text
Failed to initialize NVML: GPU access blocked by the operating system
```

这是设备访问失败，尚不能据此判断 GPU 型号或支持能力。随后在可访问 GPU 的执行环境查询到：

```text
NVIDIA GeForce RTX 4060 Laptop GPU, 8.9, 596.21
```

工具链是 CUDA 13.4.92。于是将验证分为两部分：`sm_89` 固定寄存器基线实际运行，`sm_90a` 指令版交叉编译、检查 SASS，并验证其设备不匹配时返回 77。没有尝试在 Ada 上强行执行 Hopper 专用指令，也没有修改驱动或 GPU 配置。

最终环境证据见 [baseline: device.log](results/sm_89-baseline.log)、[baseline: compiler.log](results/sm_89-baseline.log) 和 [sanitizer: preflight.log](results/sm_90a-dynamic.log)。

## 2. 第一个失败：检查器混淆 PTX 与 SASS 名称

第一版 `check_static.py` 搜索 SASS 中的 `SETMAXNREG`，虽然编译成功，检查仍报：

```text
AssertionError: Mode 1: SETMAXNREG missing in native code
```

历史记录 `setmaxnreg-20261023T134010Z-cOy8Jz` 已被最终 static 取代并清理；此处错误摘录仅保留故障背景，原日志和 SASS 已清理。

人工检查发现，实际指令名为：

```text
USETMAXREG.DEALLOC.CTAPOOL 0x40
USETMAXREG.TRY_ALLOC.CTAPOOL UP0, 0xc0
```

这不是被测指令消失，而是检查器用了错误名称。修正为按 kernel 分段检查 `USETMAXREG` 的动作、立即数及数量，并要求基线不含该指令。还增加了资源元数据和实际寄存器编号检查。`TRY_ALLOC` 周围的谓词/回跳是本次编译器为申请生成的等待实现，不能只找到一个助记符便忽略控制流。

最终 [static: static_check.log](results/sm_90a-dynamic.log) 通过。失败解析器未被忽略；首次记录未被改写为成功，现已按保留策略清理。

## 3. 第二个问题：有 `inc 192`，消费者仍然大量 spill

第一版大致结构是：

```cpp
if (donor) decrease<64>();
__syncthreads();
if (!donor) increase<192>();
__syncthreads();
if (donor) { /* light work */ }
else       { /* 128 accumulators */ }
```

`inc` 与高需求计算不在同一连续的角色分支内，中间发生控制流汇合。虽然 PTX/SASS 中有配额指令，`redistribute` 的首次编译却报告 472 字节 stack frame、1120 字节 spill stores 和 1120 字节 spill loads，这是初版历史摘录，所属 `setmaxnreg-20261023T134010Z-cOy8Jz` 的原日志已清理，不用最终日志替代这些旧数值的证据。

这提示不能把“请求了更多寄存器”与“编译器在目标计算区间使用了更多寄存器”等同。对编译器具体内部算法没有直接证据；“汇合后按较小配额保守分配”是根据代码形态及生成代码提出的解释。

修正时将消费者的 `inc 192 → 组内屏障 → 初始化/计算/写出 → 组内屏障 → dec 128` 放在同一个分支内。先写出结果并结束累加器生命周期，再缩减；全 CTA 只在必要的交换/恢复边界汇合。

| `redistribute` 静态项目 | 初版 | 最终版 |
| --- | ---: | ---: |
| cubin `REG` | 128 | 128 |
| stack frame 字节 | 472 | 0 |
| ptxas spill stores 字节 | 1120 | 0 |
| ptxas spill loads 字节 | 1120 | 0 |
| 最终 SASS 最高显式通用寄存器 | 此表不作统计 | R164 |

最终证据见 [static: build.log](results/sm_90a-dynamic.log)。这不是纯粹“移动一个 if”的严格性能消融：屏障组织也随分支调整改变了。能得到的结论是最终结构在本工具链下达到了需要的静态寄存器分配效果，不能给出初版到终版的 GPU 加速比。

## 4. 第三个问题：合法立即数不代表合法方向

最初上界探针写为 `.maxnreg 128` 后执行 `dec 256`，原意只是测试立即数上界，实际触发了方向矛盾。编译器返回 255，并给出 C7406，称减配目标高于程序中的最大 temporal register count。旧记录 `setmaxnreg-20261023T134132Z-n3YWJN` 已由最终 debug 取代，原日志已清理；此处保留历史诊断说明。

该诊断内部把操作数显示为 255；不根据诊断措辞擅自把 ISA 上界改成 255。用户材料与正式条目规定的立即数上界仍是 256。

最终将上界语法探针改成 `inc 256`，另外保留 `wrong_direction_dec256` 作为预期失败。**所有探针只组装，从不执行**：特别是 `inc 256` 虽能组装，当前探针没有提供足够的 CTA 寄存器预算，不能当作合法运行示例。

最终 [debug: compile_probes.log](results/sm_90a-dynamic.log) 包含 8 个检查：

| 探针 | 预期 | 本次退出码 |
| --- | --- | ---: |
| `dec 24` / `sm_90a` | 接受下界语法 | 0 |
| `inc 256` / `sm_90a` | 接受上界语法；不执行 | 0 |
| `dec 16` | 拒绝小于下界 | 255 |
| `dec 65` | 拒绝非 8 的倍数 | 255 |
| `dec 264` | 拒绝大于上界 | 255 |
| `dec 64` / `sm_89` | 拒绝目标架构 | 255 |
| `dec 64` / `sm_90` | 拒绝普通 Hopper 目标 | 255 |
| `dec 256` / 初始配置 128 | 拒绝这个可静态发现的方向矛盾 | 255 |

检查器同时确认退出状态和诊断类别，不会把工具缺失或任意失败当作“反例测试通过”。全部正负探针每次重新组装并匹配诊断；命令、输出和退出码写入 debug 动作日志，PTX 和成功 cubin 留在共享 build 中。本轮未做真实编译验证。编译器能拒绝这个方向错误，不代表能静态发现所有线程参与错误或循环等待。

## 5. 第四个问题：sanitizer 通过仍须复核屏障语义

第一次将计算移入消费者分支时，组内屏障写成了 `bar.sync 1, 128`。该版本的本机基线通过了 memcheck 与 synccheck，旧记录 `setmaxnreg-20261023T134355Z-BAB070` 已由最终 baseline-sanitizer 取代，原日志已清理；此处仅保留复核背景。

继续核对 [PTX barrier 规范](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar) 后发现，`bar.sync` 等价于隐含 `.aligned` 的形式，而文档要求其 CTA 线程满足一致参与条件。本例只有 WG1 进入这段分支，因此改为：

```cpp
asm volatile("barrier.sync 1, 128;" ::: "memory");
```

该形式没有 `.aligned`。barrier 1 专用于完整 WG1，参与数为 128；CTA 其余线程不参加。两次组内屏障均完成后才复用同一资源，没有其他角色向其计数。

最终重新执行静态检查、数值校验、memcheck 和 synccheck；见 [baseline-sanitizer: 最终 memcheck](results/sm_89-baseline.log)、[baseline-sanitizer: 最终 synccheck](results/sm_89-baseline.log)。旧版本的工具成功不再作为最终版本正确性的依据；这也说明工具无报错不能替代规范推导。

## 6. PTX 导出阶段的寄存器警告

`static` 的 `ptx.log` 中出现：

```text
ptxas warning : Too big maxrregcount value specified 128, will be ignored
```

它来自单独的 `-ptx` 导出步骤，保留在 [static: ptx.log](results/sm_90a-dynamic.log)。不能仅据此断言之前构建的 cubin 已忽略 128 配置，也不能忽略它后把 `.ptx` 当作独立携带完整资源设置的可部署产物。

本次最终判断依据是可执行文件构建日志中对 `maxrregcount=128` 的明确处理，以及从**该可执行文件**提取的 [static: resources.log](results/sm_90a-dynamic.log) 中三个 kernel 的 `REG:128`。本机基线的运行时函数属性也报告 `numRegs=128`。动态版本另有启动前检查，但因设备不匹配，本次未走到该检查。

导出文件未声明 `.maxnreg 128`；若单独交给 `ptxas`，必须随命令重新提供同样的寄存器约束，并重新检查产物。不要用无约束的重组装结果替代实验二进制。这里未确定导出警告的编译器内部成因，只确定它出现的步骤和实际 cubin 的配置。

## 7. 在受支持设备上出现挂起时怎样定位

以下是后续排查顺序，**不是本次已发生的 GPU 故障**：

1. 先运行 `bash commands/static/run.sh`，确认初始资源是 128、四种调整未消失，检查器失败则调查实际 SASS。
2. 用 `BLOCKS=1 ROUNDS=1 STEPS=1 REPEATS=0 bash commands/run/run.sh` 缩小问题。顺序输出为基线、roundtrip、redistribute，可根据最后完成的 case 缩小范围。
3. 对照每个阶段的配额表，检查总需求是否超过 `256×128`，以及 `.inc` 之前是否已存在可执行的释放路径。
4. 检查角色谓词是否 warpgroup 一致，是否错误改成 `threadIdx.x < 32`，是否有线程提前退出，是否改了固定的 256 线程启动布局。
5. 检查 barrier ID、参与数和到达次数。不要在一个角色等待寄存器时，让唯一能释放资源的角色等它到达另一个屏障。
6. 一轮通过后增加 `ROUNDS`；多轮才失败时重点检查恢复配额、屏障复用、缩减后的活跃值和重新申请后的初始化。
7. 再运行 `bash commands/sanitizer/run.sh`，结合 memcheck / synccheck 定位具体访问或同步问题；不能用工具结果证明寄存器池一定不会死锁。

非法 warpgroup 参与、过量申请及错误 barrier 等反例只用于上述分析，没有加入会故意挂起 GPU 的运行选项。host 侧参数上限也不是 `setmaxnreg.inc` 的超时保护；修改协议后仍需重新做资源和依赖推导。
