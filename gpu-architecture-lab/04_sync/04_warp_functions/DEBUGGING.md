# Warp Functions 调试

> 当前复现入口已迁到 `commands/<action>/run.sh`：先按专题 README 显式 compile，再执行相应动作。下文命令是当前复现写法，历史精确命令仍以原日志为准；本轮未重新采集 GPU 数据。

> 记录整理（2026-11-06）：历史证据按真实编译配置合并为 `results/<variant>.log`，各 action 仅保留最新一节；见 [记录索引](results/README.md)。无日期目录或源码/脚本副本。本次未编译、未运行 GPU。下文数值仅对应标有 `HISTORICAL IMPORT` 的原记录；链接进入配置日志后按 `ACTION` 与 `FILE` 标界查找。本轮迁移保留这些历史证据；新日志写入独立动作目录，旧结论不自动套到新日志。

先按 README 显式 compile，再执行 `CUDA_ARCH=sm_89 bash commands/run/run.sh`，确保 CPU 逐字段校验通过；换设备时调整架构。日志位于脚本同目录 `commands/<action>/<variant>.log`，同动作同配置重跑覆盖（包括失败），其他动作日志不变。命令、实际参数、原始输出及退出码均记录；检查器的临时步骤文件在退出后清理。不生成自动快照、时间戳归档或 history。已有 `results/` 原始日志和历史报告保持原样，当前动作不写入。

## 1. 复现有定义的逻辑错误

```bash
CUDA_ARCH=sm_89 bash commands/debug/run.sh
```

脚本先运行 `debug_contracts.py` 的主机模型，再启动 `width_debug_kernel`。若目标是半 warp 广播，默认 width 的版本在 lane 0～15 得到 16，修复为 `srcLane=0,width=16` 后得到 0；lane 16～31 始终得到 16。程序必须同时验证“错误版本恰有 16 个不符合目标的值”与“修复版本全部符合目标”，否则失败。

这里错误版本的 CUDA 调用本身合法，因此 sanitizer 未报告错误是合理结果：程序仍需要独立的期望值。主机模型则展示 0xb1111 拼写、遗漏 caller、命名未调用 lane、重叠不一致 mask、inactive source、非法 width、跨半组共享内存访问等契约错误。它只描述一次会合且所有 lane 尚未退出，不解析实际 CUDA 控制流，也不模拟允许忽略的已退出线程。不要把这些非法调用直接加入正常 GPU 校验，以免产生无定义结果或挂起。

## 2. Compute Sanitizer

```bash
CUDA_ARCH=sm_89 bash commands/sanitizer/run.sh
```

| 工具 | 本例主要检查什么 | 不能据此推断什么 |
| --- | --- | --- |
| memcheck | 尾部输入、输出和共享内存寻址 | 不验证算法目标 |
| racecheck | 共享内存发布及下一轮复用 | 不能证明任意 global-memory 协议正确 |
| synccheck | 工具支持范围内的同步误用 | 不能穷举所有 mask/source/控制流问题 |
| initcheck | 默认检查的 device global memory 未初始化读取 | 不证明所有局部寄存器/C++ 值初始化正确 |

所有工具均配置 `--error-exitcode 86`。某个工具失败后继续收集其他工具证据，最终整体返回非零；超时也视为未通过。工具范围依据 [Compute Sanitizer 文档](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)，零错误需要与代码的参与集合和 happens-before 推导一起解读。

## 3. CUDA-GDB 进入 kernel

```bash
CUDA_ARCH=sm_89 BUILD_MODE=debug bash commands/compile/run.sh
CUDA_ARCH=sm_89 bash commands/gdb/run.sh
```

该动作构建 `-O0 -g -G` 版本，通过 `set cuda break_on_launch application` 停在 `width_debug_kernel`，打印 `info cuda kernels` 和 `info cuda threads`，继续运行并核对正常退出与 PASS。它不自动改动系统调试权限。若驱动、沙箱或 ptrace 限制阻止调试，日志应保留为未验证，不是 GPU 算法失败的证明。

执行当前 `gdb` 动作建立调试构建后，需要逐步观察时使用同配置程序（以下为 `sm_89`）：

```text
cuda-gdb build/sm_89-debug/bin/warp_functions
(cuda-gdb) set cuda break_on_launch application
(cuda-gdb) run debug
(cuda-gdb) info cuda kernels
(cuda-gdb) info cuda threads
(cuda-gdb) next
(cuda-gdb) print lane
(cuda-gdb) continue
```

优化构建中变量可能被删除或移动；观察局部值使用本动作的 debug 构建，不拿它的执行速度与 `-O3` 比较。

## 4. 区分原生机器码与 PTX JIT

```bash
CUDA_ARCH=sm_89 bash commands/jit/run.sh
```

该动作设置 `CUDA_FORCE_PTX_JIT=1`，强制走 PTX。如果遇到 `cudaErrorUnsupportedPtxVersion`，检查生成 PTX 的 Toolkit 与驱动支持范围；保留错误，不删除校验或把之前 cubin 路径的 PASS 记为 JIT 通过。`main.ptx` 中的 `.version` 和程序打印的 `driver_api/runtime` 可帮助定位。驱动 API 数字与 `nvidia-smi` 显示的驱动发行版本是不同字段。

## 5. 从输出定位到代码

CPU 比较失败会输出 `field index got expected`。先由 index 和 block 形状算出 warp 起点、lane 与成员集合，再手算源 lane。检查 mask 时逐项询问：调用者是否被命名、每个被命名的未退出线程是否到达同一调用、mask 是否一致、shuffle 的目标源是否参加。检查 shared memory 时再确认 barrier 同时覆盖写者和读者，并保护循环下一轮的覆盖写入。

运行 `static` 后可读取 `build/<arch>-release/main.ptx`，并在 `results/<arch>-release.log` 的 static 节中找到 SASS 输出；动作中临时 `sass.log` 在 EXIT 合并后清理。在对应 kernel 中找源码行与指令。静态检查器解析失败必须调查，不能绕过后继续声称指令存在。
