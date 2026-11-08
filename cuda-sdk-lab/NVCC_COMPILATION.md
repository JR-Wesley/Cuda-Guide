# `nvcc`：一次只学一个编译阶段

## 为什么拆开学

`.cu` 同时包含 CPU 与 GPU 代码。`nvcc` 驱动主机编译器、设备前端、`ptxas` 汇编器和 `nvlink` 设备链接器完成不同工作。直接生成最终程序容易掩盖这些边界：**编译成功、镜像可加载、GPU 计算正确、运行快，是四种不同结论**。

本课程复用 [compile/ 中的三份源码](experiments/compile/README.md)，每个动作都有短小 `run.sh`、自己的 README 和就地 `log/`。先看源码手算，再读底层命令，再执行该动作；没有 `all` 调度器，没有编译后自动检查或运行。

## 分步入口

以下目录相对于 `cuda-sdk-lab/`。手动进入目标动作目录后执行 `CUDA_ARCH=sm_89 bash run.sh`；脚本不切换目录，`CUDA_ARCH` 没有默认值。每个 README 都按背景、底层命令、脚本、真实日志、边界练习组织。

| 顺序 | 动作 README | 本节只做什么 |
| --- | --- | --- |
| 1 | [compile-dryrun](experiments/compile-dryrun/README.md) | `--dryrun` 只打印编译计划，不生成二进制或大中间文件 |
| 2 | [compile-ptx](experiments/compile-ptx/README.md) | 从源码输出 `probe.ptx`，读虚拟指令与源码的对应关系 |
| 3 | [compile-cubin](experiments/compile-cubin/README.md) | 输出 `probe.cubin`，保留重定位信息供 `nvdisasm` 使用 |
| 4 | [compile-fatbin](experiments/compile-fatbin/README.md) | 输出原生代码与 PTX 的镜像容器 `probe.fatbin` |
| 5 | [compile-object](experiments/compile-object/README.md) | `-c` 生成含主机代码与设备镜像的 `probe.o` |
| 6 | [compile-host-link](experiments/compile-host-link/README.md) | 读取 `probe.o`，动态链接为 `probe` |
| 7 | [compile-rdc-compile](experiments/compile-rdc-compile/README.md) | `-dc` 分别编译两个跨文件调用的源文件 |
| 8 | [compile-rdc-device-link](experiments/compile-rdc-device-link/README.md) | 解析跨文件设备引用，输出 `rdc_dlink.o` |
| 9 | [compile-rdc-host-link](experiments/compile-rdc-host-link/README.md) | 复用三个目标文件，只主机链接为 `rdc_app` |
| 10 | [compile-lto-compile](experiments/compile-lto-compile/README.md) | `code=lto_89` 保存链接时优化的中间表示 |
| 11 | [compile-lto-device-link](experiments/compile-lto-device-link/README.md) | `-dlto` 执行 LTO 设备链接 |
| 12 | [compile-lto-host-link](experiments/compile-lto-host-link/README.md) | 只主机链接为 `rdc_lto_app` |

2–5 各自读取源码，没有相互依赖。6 依赖 5；7→8→9 和 10→11→12 是需要手动按顺序执行的链路。RDC 是“可重定位设备代码”，解决分别编译后的设备引用；LTO 是“链接时优化”，给链接器更多跨文件优化信息，不保证内联或加速。

## 产物、日志和重跑规则

所有输出继续共享在 `build/sm_89/compile/artifacts/`，不复制源码、产物或运行快照。常规产物路径保持为 `probe.ptx`、`probe.cubin`、`probe.fatbin`、`probe.o`、`probe`；RDC/LTO 也沿用原文件名。静态工具后续读取它们，不需要在分析阶段重新编译。

每个动作 `source ../record.sh`，记录 `log/<label>.sh`（命令）、`log/<label>.log`（原始输出）和 `log/status.tsv`（退出码）。命令使用 `timeout 60s`，超时返回 124；一般返回原始工具状态。缺输入返回 1；失败清理当前动作的部分编译输出。重跑只更新当前动作固定日志与产物，不删除其他动作结果。改源码后由使用者重做下游阶段，文件存在不等于最新。

不执行 `--keep`、`-E` 或 `--cuda` 大文本导出，不生成额外独立 `.ltoir` 或 PTX-only 程序；编译器内部临时文件仍按正常编译流程使用和清理。可执行文件统一 `-cudart=shared`，避免静态 Runtime 使小例子膨胀。

旧大入口 `experiments/compile/compile-trace.sh` 已被替代并删除，顶层导航已同步为上述独立动作。本轮开始时该目录没有旧 `log/`；没有创建历史目录。

## 本次真实观察与验证边界

本轮 [环境检查](experiments/environment/README.md) 确认 **RTX 4060 Laptop / sm_89**、驱动 **596.21**；实际 `nvcc --version` 为 **13.4.92**。本页列出的编译动作只做离线处理，不启动 GPU。

- 12 个脚本均通过 `bash -n`；当前正常路径 14 条编译/链接/计划命令全部退出 **0**。
- LTO 第一次编译因 `-dlto` 与 `-gencode ...code=sm_89` 冲突退出 **1**。按本版本诊断改用 `code=lto_89` 后通过；原始失败保存在 [LTO 动作的具名反例日志](experiments/compile-lto-compile/log/rejected_flags.log) 和 [退出码记录](experiments/compile-lto-compile/log/rejected_flags-status.tsv)，没有用成功结果掩盖它。
- 13 个产物共 **158,026 字节（约 154.3 KiB）**；`probe` 为 **22,528 字节**，两个 RDC/LTO 程序各 **28,680 字节**。完整逐文件统计见 [课程结果表](experiments/compile/README.md)。没有 `keep/`、持久化 `.ii` 或 dryrun 程序。
- `probe.ptx` 实际包含 PTX 9.4 与 `fma.rn.f32`；cubin 编译诊断报告 10 个寄存器、0 字节 spill。额外 `readelf -d` 静态核对确认三个程序动态依赖 `libcudart.so.13`，没有把这项检查混入构建脚本。
- 未验证 GPU 加载、PTX JIT、计算结果、LTO 内联或性能。相同程序大小不表示相同设备指令；生成 PTX 不保证驱动支持该 PTX 版本。本轮环境的 `LD_LIBRARY_PATH` 包含 `/opt/cuda/lib64`，但这些编译课程程序未启动验证加载；其他小型加载示例的运行证据见 [模块加载](experiments/module-loading/README.md)。

## 理解检查

1. 为什么 PTX、cubin、fatbin 都不等于 CPU 可启动程序？
2. 为什么普通 `.o` 与 `-dc` 产生的 `.o` 不能只凭扩展名区分设备链接需求？
3. 为什么修改 RDC 中一个源文件后，需要重新做设备链接和主机链接？
4. 要判断 LTO 是否消除了调用，下一步该读 SASS 还是只比较文件大小？

官方参考：[CUDA Compiler Driver NVCC](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)、[CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html)。
