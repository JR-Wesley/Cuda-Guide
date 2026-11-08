# 异步 barrier 调试过程

> 当前复现入口已迁到 `commands/<action>/run.sh`：先按专题 README 显式 compile，再执行相应动作。下文命令是当前复现写法，历史精确命令仍以原日志为准；本轮未重新采集 GPU 数据。

> 记录整理（2026-11-06）：历史证据按真实编译配置合并为 `results/<variant>.log`，各 action 仅保留最新一节；见 [记录索引](results/README.md)。无日期目录或源码/脚本副本。本次未编译、未运行 GPU。下文数值仅对应标有 `HISTORICAL IMPORT` 的原记录；链接进入配置日志后按 `ACTION` 与 `FILE` 标界查找。本轮迁移保留这些历史证据；新日志写入独立动作目录，旧结论不自动套到新日志。

本文记录实际运行中的问题、定位动作和结论边界。源码中的 `debug.cu` 含一个故意的共享内存竞态，只有 `bash commands/debug/run.sh` 才会在独立进程中执行。正常功能程序不会启动这个反例。

## 1. 先核对协议，再看数值和工具

对每个 barrier 写下四项：初始化线程及 bootstrap 同步、当前 phase 的 arrival 义务、谁等待哪个 token/parity、下一次复用之前的依赖边。

本实验的对应检查是：

| 问题 | 方法 |
| --- | --- |
| 正常输出是否符合数学预期 | 每个线程/元素对照 CPU 参考值 |
| 是否存在越界、错位访问 | memcheck，尤其覆盖 n=1/255/256/257/4099 |
| shared 数据读写是否有同步 | racecheck + 源码 happens-before 推导 |
| barrier 使用是否存在可检测违例 | synccheck |
| 是否真的发出异步指令 | PTX/SASS，检查 cp.async、barrier 绑定、LDGSTS |
| 是否实际计算/拷贝重叠或加速 | 本专题未测，不能由以上检查推出 |

正常构建使用 `-O3 -lineinfo`，保留优化与行号。分析器作用和局限参见 [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)。

## 2. 初次编译：shared barrier 动态初始化警告

初次编译曾报告（旧 build 目录 `async_barriers-20261023T134041Z-GOsLki` 已由后续完整构建覆盖，日志已清理） `static_var_with_dynamic_init`。这里不能依赖 shared 变量的自动动态构造；代码已由线程 0 调用 `init`，全 block 随后同步，自定义 completion 则使用对齐 storage 和 placement new。

确认这些初始化路径后，在对应 `.cu` 文件局部抑制该诊断，未关闭其他编译警告。不要把诊断抑制复制到没有显式初始化的代码中。后续构建日志保留以供核对。

## 3. 可恢复的 arrival 数量错误

运行：

```bash
bash commands/debug/run.sh
```

`missing_arrival<<<1,1>>>` 设置 expected=2，但先只 arrive 一次。使用 `try_wait_for(...,100us)` 检查当前 phase，避免让 GPU 永久挂在普通 wait 上。失败的 timed wait 保留 token 的使用资格；随后补上尚缺的 arrival，再用原 token 等待。下一 phase 使用 `arrive(2)` 完成两个义务。

这是当前源码的预期行为，不是本次实测。旧 `debug` 记录中的 `main.cu` 过时，整节证据已清理，当前动作未验证；不再展示旧输出作为当前证据。100us 是等待预算，不是精确计时结果；等待失败一般也可能来自调度，这个示例则由单线程控制到达义务。

这不是让生产代码随意补到达数的建议。真实协议必须找出缺失参与者：分支退出、少调用一次、错把 warp 数当线程数等。提前退出应使用 `arrive_and_drop`，而不是让别的线程猜测缺了几次 arrival。

## 4. 真实竞态：把发布写放在 arrive 后

`late_publish` 故意执行：

```cpp
auto token = bar.arrive();
data[t] = t + 1;                 // 错：发生在本线程发布点之后
bar.wait(cuda::std::move(token));
out[t] = data[(t + 32) % 64];    // 读取另一个 warp 的槽
```

虽然所有线程执行了 wait，但 barrier 只发布 arrive 之前的相关访问。最后到达者完成 phase 后，一个 warp 的读取可与另一个 warp 的后置写冲突。`volatile` 只让反例中的访问保留，不提供线程间同步。

旧反例的 racecheck 报告已随过时 `debug` 记录清理，不再保留其行号与 hazard 数值结论。当前动作未验证。程序不对竞态输出设置某个“必须错误”的数值，因为存在数据竞争时不能枚举可靠结果。

正确修复是在 arrive 之前写入：`data[t]=...; token=arrive(); wait(token); read;`。正常程序 `exchange<1>` 已采用这个顺序，并额外用第二个 phase 保护下一轮覆盖。

当前脚本要求退出码 86 与明确的 racecheck 诊断同时存在，才将“反例成功被检测”标记为通过。普通程序崩溃、超时或工具未启动都不会冒充成功；尚无本次执行证据。

## 5. 正常示例的报告：没有用数值正确覆盖分析器失败

首次正常路径检查中，memcheck 为零错误，所有数值校验通过，但 racecheck 返回 86：五组 error、一组 warning（旧目录 `async_barriers-20261023T134132Z-CXnKvX` 已被后续完整 sanitizer 记录取代，日志已清理；这里仅保留历史描述）。报告主要集中在 `exchange<2>` 的显式 parity，以及自定义 completion 的 shared 数据和库内部原子读写。

最初脚本遇到非零即停止，因此该目录没有 synccheck 结果。随后修改脚本，让三种工具都执行，并在最后继续返回非零汇总状态；没有添加 suppression 或把该报告当作预期故障。

### 5.1 显式 parity 的隔离对照

运行 `bash commands/diagnose/run.sh`：

- 过滤到具体 mangled kernel 子串 `kns=8exchangeILi2E`，防止把其他模板实例混进来。
- 原版使用 `cuda::ptx::mbarrier_*_wait_parity` 循环；对照版只把等待换成 `bar.wait_parity`，数据访问、arrival 和 phase 数不变。
- 两版各检查 1 轮与 9 轮，启用 `--racecheck-trace-sync yes` 保存同步轨迹。

[诊断状态](results/sm_89-release.log) 显示：单轮两版都是 0；九轮两版都是 86。轨迹中确实有 barrier init、arrive 与 wait 记录，因此不能解释为“工具完全没有看到 barrier”。

源码复核结果：共享写在发布 arrival 之前；读在发布 wait 后；读后第二次 arrive/wait 完成才允许下一轮覆盖。每个 phase 都有 128 个 arrival；parity 在两个阶段分别翻转，所有参与线程都等待，不会越过一个落后的线程完成两个 phase。当前构建是否保留相应指令需重新执行 static；旧 PTX 已清理。

这一对照排除了“只在手写 PTX 路径发生”的解释，报告表现与多 phase 追踪有关；**当前证据仍不足以把它确定为工具缺陷**。本专题保留语义正确的协议与原始报告，不通过添加多余 `__syncthreads()` 改变被研究的同步机制来换取零报告，也不声称 parity 已获得 racecheck 的无冲突验证。

### 5.2 Completion 单轮仍报告冲突

[completion 单轮报告](results/sm_89-release.log) 包含两组 error、一组 warning。单轮没有下一轮覆盖，因此“少了跨轮 reuse barrier”不能解释这些报告。

本机 Toolkit 的 `cuda/std/__barrier/barrier.h` 对自定义 completion 使用软件原子实现：arrival 通过 acq_rel 计数，最后参与者执行 completion，并以 release 发布 phase；wait 通过 acquire 观察。相比之下，空 completion 的 block shared barrier 有硬件特化。报告还涉及库内 `__cuda_atomic_store` / `__cuda_atomic_load`。

按 API 契约，arrival 前的槽写入应对 completion 可见，completion 的 total 更新应对 waiter 可见；下一次 completion 也被所有线程的下一次 arrival 限制。现象与分析器对该软件协议的跟踪不完整相符，这是**基于源码与报告的推断**，尚未得到工具维护者确认。不能将其写成已确认误报，也不能把本次完整 racecheck 标为通过。

后续若在其他 Toolkit/GPU 复核，应保留版本、源码版本说明、单轮/多轮对照、PTX/SASS 与同步轨迹；本次未修改驱动、全局工具设置或标准库实现。

### 5.3 Synccheck 与等待轨迹的差异

补跑的 [完整 synccheck](results/sm_89-release.log) 在 `exchange<2>` 的下一轮 arrival 处报告 `Missing wait`，共 512 项。随后同步 API 返回 Unknown，程序终止；该日志不覆盖后续 drop、completion 和拷贝 kernel。

旧 `static` 使用尚无 `LAB_API_PARITY` 分支的 `main.cu`，相关 SASS 已清理。此前依赖该 SASS 偏移建立的两处等待事件计数对照一并撤下；不能将不同源码记录放在同一配置日志中就视作一致。

[单轮 trace](results/sm_89-release.log) 与 [九轮 trace](results/sm_89-release.log) 原文仍保留在 diagnose 节，但当前 static 未验证。下一步需在同一源码/编译状态下重新取得静态产物与动态轨迹，再调查等待识别；现有证据不足以确认根因。没有通过更改硬件设置、屏蔽完整报告或添加额外同步来把失败变成通过。

为完成后续 kernel 的检查，另提供 `bash commands/isolate/run.sh`：racecheck 排除显式 parity 和自定义 completion，synccheck 仅排除 parity；程序仍执行全部数值校验。这个动作在终端和日志中标为 **SUBSET ONLY**，其通过只代表明确过滤后的范围，不抵消完整 `sanitizer` 的失败。最新状态见结果文档。

### 5.4 过滤参数也需要验证

第一次隔离命令误用了 `regex:...` / `kns:...`，工具在启动目标程序前报 `Filter ill-formed`，退出 1（旧目录 `async_barriers-20261023T134341Z-Eq6mvX` 的日志已清理；仅保留这段历史说明，不把后续正常过滤日志作为该错误的证据）。根据本机 `--help` 修正为 `key=value`，过滤对象为 mangled 名称。后续日志包含实际 kernel 的同步轨迹或 hazard，避免把“没有匹配任何 kernel”的零报告当作验证。

## 6. 静态与硬件边界调试

SM89 的显式等待使用 `mbarrier.test_wait.parity`，不能照抄仅支持 SM90 的 try_wait 指令。本例按编译目标选择；`static` 检查指令存在性，但不借此推断速度或实际重叠。

SM90 的 transaction count 示例与主程序分离编译。本机执行前检查 capability，返回 77；此状态只说明正确跳过，不能替代 SM90 实机测试。源与目的均 16 字节对齐，expected transaction count 为 `sizeof(data)=1024`，避免把 256 个元素误写成 256 字节。

最终日志索引与每项验证状态见 [RESULTS.md](RESULTS.md)。
