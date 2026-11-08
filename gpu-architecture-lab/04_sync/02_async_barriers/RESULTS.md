# 本机验证结果

> 当前复现入口已迁到 `commands/<action>/run.sh`：先按专题 README 显式 compile，再执行相应动作。下文命令是当前复现写法，历史精确命令仍以原日志为准；本轮未重新采集 GPU 数据。

> 记录整理（2026-11-06）：历史证据按真实编译配置合并为 `results/<variant>.log`，各 action 仅保留最新一节；见 [记录索引](results/README.md)。无日期目录或源码/脚本副本。本次未编译、未运行 GPU。下文数值仅对应标有 `HISTORICAL IMPORT` 的原记录；链接进入配置日志后按 `ACTION` 与 `FILE` 标界查找。本轮迁移保留这些历史证据；新日志写入独立动作目录，旧结论不自动套到新日志。

本次验证日期：2026-10-23。设备为 NVIDIA GeForce RTX 4060 Laptop GPU，compute capability 8.9；`cudaDriverGetVersion=13020`、`cudaRuntimeGetVersion=13040`，nvcc 13.4.92，Compute Sanitizer 2026.3.0。编译参数为 C++17、`-O3 -lineinfo`，主程序目标 `sm_89`，事务示例独立编译 `sm_90`。

**普通数值检查与完整 memcheck 通过；完整 racecheck、synccheck 未通过。SM90 示例编译成功，本机不支持执行。** 以下将这些状态分别列出，不用过滤后的结果代替完整检查。

## 1. 可执行功能

[sm_89-release.log](results/sm_89-release.log) 的 `ACTION: run` 节保留最终运行的 `status.tsv` 与 `metadata.txt`（参数、源码哈希）；本节以下数值仅对应其中标记的历史 record。

| 组别 | 验证内容 | 结果 |
| --- | --- | --- |
| `syncthreads_baseline` | 四个 block 各 128 线程，邻居值与独立计算累积 | PASS |
| `split_arrive_wait` | token 等待，发布/复用两个 phase | PASS |
| `explicit_parity` | 原生 PTX parity 等待，两个 phase 都翻转 | PASS（数值） |
| `arrive_and_drop` | warp 参与轮数 2/4/6/8，退出后减员 | PASS |
| `completion_once_per_phase` | 每个等待线程的累计和、completion 次数 | PASS（数值） |
| `sync_copy` | 普通 global→shared 基线，五种输入大小 | 全部 PASS |
| `async_copy_barrier` | 绑定到 barrier 的异步拷贝，五种输入大小 | 全部 PASS |
| `double_buffer` | 一个生产 warp、三个消费 warp、四个 barrier | 全部 PASS |

每次程序运行共 20 个 case。默认 17 轮、边界 1 轮、边界 2 轮均通过，共 60 项 case 检查。原始日志分别为 [run.log](results/sm_89-release.log)、[boundary_one.log](results/sm_89-release.log)、[boundary_two.log](results/sm_89-release.log)。

输入大小 `1/255/256/257/4099` 对应完整输入分配 `256/256/256/512/4352` 个元素。输出只含原始 n 个元素。4099 覆盖 17 个 tile，能够复用双缓冲多次，不只是验证第一轮交接。

## 2. 分析器：完整与过滤后的检查

完整检查使用 9 轮；drop 仍最多八轮。见 [状态表](results/sm_89-release.log)。

| 工具与范围 | 退出码 | 实际结果 |
| --- | ---: | --- |
| [memcheck，全部 kernel](results/sm_89-release.log) | 0 | 20 个 case 数值通过，0 errors |
| [racecheck，全部 kernel](results/sm_89-release.log) | 86 | 5 errors、1 warning；集中于 parity 与 custom completion，数值仍通过 |
| [synccheck，全部 kernel](results/sm_89-release.log) | 86 | parity 处 512 项 Missing wait；程序随后终止，后续 kernel 未执行 |

为覆盖受提前终止影响的 kernel，另执行显式过滤检查。见 [独立状态表](results/sm_89-release.log)：

| 工具与明确排除范围 | 结果 |
| --- | --- |
| [racecheck，排除 parity 和 completion](results/sm_89-release.log) | 0 hazards；其余 kernel 的检查通过 |
| [synccheck，仅排除 parity](results/sm_89-release.log) | 0 errors；其余 kernel 的检查通过 |

这些过滤结果仅补足其余范围，不将完整检查失败改写为成功。custom completion 的 synccheck 零报告也不能替代 racecheck 验证。

## 3. 已完成的定位与仍有限制的结论

[配置日志](results/sm_89-release.log) 的 `ACTION: diagnose` 节提供以下对照：

| Parity 等待实现 | 一轮 | 九轮 |
| --- | --- | --- |
| [原生 PTX](results/sm_89-release.log) | racecheck 0 | [racecheck 86](results/sm_89-release.log) |
| [库 `wait_parity`](results/sm_89-release.log) | racecheck 0 | [racecheck 86](results/sm_89-release.log) |

两种实现的单轮/九轮退出码表现相同。旧源码 SASS 已清理，不再用其偏移对照这些轨迹，也不据此断言当前构建中的等待循环形式。completion 单轮仍有 [2 errors、1 warning](results/sm_89-release.log)，涉及库软件原子协议。

这些证据支持继续调查分析器对 phase 复用和软件 barrier 的追踪，但尚不足以确定为工具缺陷。源码按 API 契约建立了发布与复用顺序；同时，本次没有获得 parity/custom completion 的完整分析器通过证据。详细推导、证据限制及过滤参数修正见 [DEBUGGING.md](DEBUGGING.md)。

## 4. 调试反例与恢复：当前未验证

旧 `debug` 记录使用的 `main.cu` 尚无 `LAB_API_PARITY` 分支，整节证据已清理；虽然 `debug.cu` 当时一致，也不保留混合版本记录作为当前配置验证。本次未重跑 `debug`，不再引用旧反例输出、hazard 数量或源码行号。

`bash commands/debug/run.sh` 仍提供缺失 arrival 的有界恢复与发布顺序错误反例。脚本要求故意竞态返回 86 且包含明确诊断；这只是预期检查条件，不代表当前执行通过。正常路径的 86 仍是未通过。

## 5. 指令与 SM90 路径

**当前主程序 static 未验证。** 旧 `static` 的 `main.cu` 过时，相关 PTX/SASS 和通过记录已一并清理；不再引用其检查数量、指令偏移或将其视作当前编译结果。需对当前源码重新执行 `bash commands/static/run.sh`。下述 `tx90` 是独立 SM90 配置的历史证据，不替代主程序 static。

[SM90 状态](results/sm_90-release.log) 显示 executable 与 PTX 编译均成功。[tx90.ptx](results/sm_90-release.log) 包含：

```text
cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes
mbarrier.arrive.expect_tx.release.cta.shared::cta.b64
```

[执行日志](results/sm_90-release.log) 在识别到 SM89 后返回 77，没有启动 SM90 kernel。事务追踪的实际完成与输出校验仍需 SM90+ 实机；当前不能宣称测试通过。

## 6. 未测量范围

本专题研究同步语义与调试，没有测量 barrier 延迟、吞吐、实际计算/拷贝重叠或双缓冲加速比。未验证 cluster scope、跨 block barrier、多 GPU 或 TMA tensor descriptor。不要将指令存在、数值通过或过滤后的零报告扩展为这些能力已被验证。
