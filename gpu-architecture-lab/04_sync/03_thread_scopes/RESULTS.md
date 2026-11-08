# 本次验证结果

> 当前复现入口已迁到 `commands/<action>/run.sh`：先按专题 README 显式 compile，再执行相应动作。下文命令是当前复现写法，历史精确命令仍以原日志为准；本轮未重新采集 GPU 数据。

> 记录整理（2026-11-06）：历史证据按真实编译配置合并为 `results/<variant>.log`，各 action 仅保留最新一节；见 [记录索引](results/README.md)。无日期目录或源码/脚本副本。本次未编译、未运行 GPU。下文数值仅对应标有 `HISTORICAL IMPORT` 的原记录；链接进入配置日志后按 `ACTION` 与 `FILE` 标界查找。本轮迁移保留这些历史证据；新日志写入独立动作目录，旧结论不自动套到新日志。

2026-10-23，在 NVIDIA GeForce RTX 4060 Laptop GPU（SM89）上完成验证。nvcc 13.4.92，Runtime API 版本值 13040，Driver API 版本值 13020；Compute Sanitizer 2026.3.0。程序报告 cooperative launch=1、concurrentManagedAccess=0、hostNativeAtomicSupported=0。

**正常运行 47 项检查通过，九项分析器检查均为零错误，九项静态检查通过。CPU/GPU managed-memory 并发与 SM90 cluster 两项扩展仅编译并验证跳过路径，本机没有执行其 kernel。**

## 1. 原始记录入口

| 动作（配置日志内同名 ACTION 节） | 状态与配置日志 | 历史结论 |
| --- | --- | --- |
| run | [status.tsv](results/sm_89-release.log) | 全部退出 0 |
| sanitizer | [status.tsv](results/sm_89-release.log) | 三个程序 × 三种工具，全部退出 0 |
| debug | [status.tsv](results/sm_89-release.log) | 字节数错误被检出；缺少 opt-in 的错误被捕获且恢复成功 |
| static（修正检查器后） | [status.tsv](results/sm_89-release.log) | 全部退出 0 |
| system | [status.tsv](results/sm_89-release.log) | 编译/PTX 成功；执行返回 SKIP/77 |
| cluster | [status.tsv](results/sm_90-release.log) | 编译/PTX 成功；执行返回 SKIP/77 |

每配置单日志保留历史 SHA256、工具命令、输出及退出码原文；日期目录与源码/脚本/二进制副本已删除。初次编译与初次静态解析失败此前已被后续同动作记录覆盖，旧日志已清理；简短历史原因仍见调试文档。run/sanitizer/debug/system 的旧 `check_static.py` 没有被这些动作调用，不能据此声称当前 helper 已验证；最新 static 单独提供相应证据。

## 2. 作用域与消息发布

[scopes.log](results/sm_89-release.log) 中六种计数均满足最终值和票号连续唯一性：

| 模式 | 通信组数 | 每组增量 |
| --- | ---: | ---: |
| thread，各自 global counter | 512 | 7 |
| block，各自 global counter | 8 | 448 |
| block，各自 shared counter | 8 | 448 |
| device，共享 global counter | 1 | 3584 |
| system，仅单 GPU 访问 | 1 | 3584 |
| builtin device | 1 | 3584 |

block 的 libcu++ 与 builtin release/acquire 两组，各验证 8 个 block × 19 轮；device cooperative 消息验证 19 轮。没有超时哨兵或错误 payload。

[scopes_one.log](results/sm_89-release.log) 将计数操作改为每线程一次，票号范围随之缩小，消息发布测试仍为 19 轮。两次运行各有 9 项检查，共 18 项。

这些结果不测量不同 scope 的延迟；system_gpu_only 不涉及 CPU 与 GPU 同时访问该 counter。

## 3. 五种拷贝与 pipeline

[copies.log](results/sm_89-release.log) 验证同步、Cooperative Groups、thread primitives + block sync、barrier-bound、two-stage pipeline 共五种路径。

| n | padded 元素数 | tile 数 | 五种路径 |
| ---: | ---: | ---: | --- |
| 1 | 128 | 1 | 全通过 |
| 127 | 128 | 1 | 全通过 |
| 128 | 128 | 1 | 全通过 |
| 129 | 256 | 2 | 全通过 |
| 1027 | 1152 | 9 | 全通过 |

共 25 项。每个输出使用自身及邻居 tile 元素，因此验证包含跨线程搬运完成的要求。九个 tile 覆盖两阶段 pipeline 的多次缓冲复用，最后不足一 tile 的部分由补零输入处理。

没有采集时间线，不把 pipeline 的结构或指令存在性写成已证明实际重叠。

## 4. Dynamic shared 与 carveout

[carveout.log](results/sm_89-release.log) 报告：

| 属性 | 字节数 |
| --- | ---: |
| sharedMemPerBlock | 49152（48 KiB） |
| sharedMemPerBlockOptin | 101376（99 KiB） |
| sharedMemPerMultiprocessor | 102400（100 KiB） |
| 本次 dynamic shared launch | 65536（64 KiB） |

设置 kernel 最大 dynamic shared 为 65536 后，请求 `-1/0/50/100` 四种 carveout 偏好，查询到的 preference 与请求相同；四次均通过共享数据的邻居求和校验。静态 shared 为 0，occupancy API 对本次 64 KiB 请求预测每 SM 一个 block。

这里读取的是偏好属性，未读取实际硬件 L1/shared 分区。max L1 请求与 64 KiB shared launch 可以并存，不能从 preference=0 推断真实 shared 为零。四种配置构成 4 项检查；与前述 18+25 项合计 47 项。

## 5. Compute Sanitizer

| 程序 | memcheck | racecheck | synccheck |
| --- | --- | --- | --- |
| scopes | [0 errors](results/sm_89-release.log) | [0 hazards](results/sm_89-release.log) | [0 errors](results/sm_89-release.log) |
| copies | [0 errors](results/sm_89-release.log) | [0 hazards](results/sm_89-release.log) | [0 errors](results/sm_89-release.log) |
| carveout | [0 errors](results/sm_89-release.log) | [0 hazards](results/sm_89-release.log) | [0 errors](results/sm_89-release.log) |

工具运行中数值检查也全部通过；没有 kernel 过滤或 suppression。检查范围是本专题正常主程序，不包括被能力门控跳过的 system/cluster kernel，也不包括此前 barrier 专题的 parity/custom completion。

racecheck 不为 global mailbox 的 scope/order 提供完整证明，语义分析仍不可省略。

## 6. 实际错误与修复证据

- 初次编译（`thread_scopes-20261023T140033Z-Eeq9OT`，旧日志已清理，仅保留历史说明）：模板把 namespace constexpr 按引用绑定，设备代码报 Tile undefined；改为值临时量后编译成功。
- [少拷贝反例](results/sm_89-release.log)：需要 512 字节却传 128 字节，96 个元素仍为 poison，host 检查正确定位逻辑错误。
- [缺少 opt-in](results/sm_89-release.log)：64 KiB launch 返回 cudaErrorInvalidValue；清除 launch 错误并设置属性后，四种偏好全部执行成功。
- 第一次静态检查（`thread_scopes-20261023T140534Z-XYvmOD`，旧日志已清理，仅保留历史说明）：解析器假定 scope 修饰符在 add 之前，导致假失败；按 opcode 修饰符集合检查后通过。

调试步骤与原因详见 [DEBUGGING.md](DEBUGGING.md)。两个运行期反例都有限且可恢复，不依赖未初始化读取、数据竞争或死锁来制造输出。

## 7. PTX / SASS 观察

修正后的 [static_check.log](results/sm_89-release.log) 共九项通过。相关 [scopes.ptx](results/sm_89-release.log) 中分别出现：

```text
count_cta:     atom.add.relaxed.cta.u32
count_gpu:     atom.add.relaxed.gpu.u32
count_sys:     atom.add.relaxed.sys.u32
count_builtin: atom.add.relaxed.gpu.u32
```

thread-scope 的当前编译形式也使用 `.cta`，说明最终实现可比源代码契约更宽，并不允许应用擅自扩大参与线程集合。

[copies.ptx](results/sm_89-release.log) 含 cp.async、commit_group、wait_group；[copies_sass.log](results/sm_89-release.log) 含 LDGSTS。完整 [scopes SASS](results/sm_89-release.log) 也保留。静态形式不证明命中层级、延迟或带宽。

## 8. 两项扩展的验证边界

CPU/GPU 握手：[system.log](results/sm_89-release.log) 因 concurrentManagedAccess=0 返回 77。编译与 [system.ptx](results/sm_89-release.log) 生成成功；尚未在支持同时 managed 访问的系统上执行。

cluster：[cluster90.log](results/sm_90-release.log) 因本机 SM89 返回 77。[cluster90.ptx](results/sm_90-release.log) 包含 `.reqnctapercluster 2,1,1` 和 `atom.add.relaxed.cluster.u32`；未在 SM90+ 上运行票号校验。

这两项不计入 47 项已通过的正常功能检查。未测量 scope 性能、真实 copy/compute 重叠、实际 cache 分区、多 GPU、cluster DSM/barrier、STAS；不填零或借用其他专题结果。
