# 从一份 CUDA 源码认识编译、设备链接与主机链接

## 学习目标与背景

源码不能直接在 GPU 上执行。`nvcc` 先把 CPU 与 GPU 部分分别处理，再把设备镜像嵌入主机程序。这里要弄懂三个问题：**当前生成的是什么文件、它由谁读取、成功生成它能证明什么**。

前置知识只需知道函数声明/定义、数组下标和命令行的输入输出文件；不要求先懂 ELF 或链接器。PTX 是虚拟指令文本，cubin 是指定 GPU 的机器码文件，fatbin 是设备镜像容器，`.o` 是等待链接的目标文件。主机指 CPU，设备指 GPU。

本目录只保留一份源码和课程导航，动作移到相邻的 `compile-*` 目录。旧 `compile-trace.sh` 已删除；不再有 `all/standard/rdc` 模式。编译、静态检查与 GPU 运行各自独立，本课程脚本**不运行 GPU、不调用分析器**。

## 1. 先读小例子并手算

- [compiler_probe.cu](compiler_probe.cu)：`i = blockIdx.x * blockDim.x + threadIdx.x` 将块号和块内线程号转换成元素下标；`i < n` 保证合法下标。CPU 准备 256 个 `x=1`、`y=1`，kernel 计算 `y=2*x+y`，所以预期每项为 3。源码已有分配、传输、启动和同步检查，保持原样。
- [rdc_main.cu](rdc_main.cu)：`rdc_kernel` 调用另一个文件定义的 `rdc_scale(3.0f)`；[rdc_device.cu](rdc_device.cu) 定义 `2*x`，预期结果为 6。这用于学习跨文件设备符号如何连接，不是性能基准。

这些是源码推导的预期，不是本轮运行结果。

## 2. 按阶段实践，每次只引入一种产物

先进入表中一个动作目录，阅读其 README 的底层命令，再运行：

```bash
CUDA_ARCH=sm_89 bash run.sh
```

每个目录的命令相同，职责不同；不要在本 `compile/` 目录执行。脚本不切换目录，也不自动调用前置动作。`NVCC` 可指定编译器，默认 `nvcc`。需要 Bash、GNU `timeout`、Toolkit 与主机 C++ 编译器，离线编译不要求 GPU 可用。

| 顺序与动作目录 | 学习内容 / 输出 | 必须先做 |
| --- | --- | --- |
| 1. [compile-dryrun](../compile-dryrun/README.md) | 打印完整编译计划；没有二进制输出 | 无 |
| 2. [compile-ptx](../compile-ptx/README.md) | `--ptx` → `probe.ptx` | 无 |
| 3. [compile-cubin](../compile-cubin/README.md) | `--cubin`、保留重定位信息 → `probe.cubin` | 无 |
| 4. [compile-fatbin](../compile-fatbin/README.md) | `--fatbin` → `probe.fatbin` | 无 |
| 5. [compile-object](../compile-object/README.md) | `-c` → `probe.o` | 无 |
| 6. [compile-host-link](../compile-host-link/README.md) | 只主机链接 → `probe` | 5 |
| 7. [compile-rdc-compile](../compile-rdc-compile/README.md) | `-dc` → `rdc_device.o`、`rdc_main.o` | 无 |
| 8. [compile-rdc-device-link](../compile-rdc-device-link/README.md) | `--device-link` → `rdc_dlink.o` | 7 |
| 9. [compile-rdc-host-link](../compile-rdc-host-link/README.md) | `--no-device-link` → `rdc_app` | 7、8 |
| 10. [compile-lto-compile](../compile-lto-compile/README.md) | `code=lto_89` → 两个 `*_lto.o` | 无；建议先读 7 |
| 11. [compile-lto-device-link](../compile-lto-device-link/README.md) | `-dlto --device-link` → `rdc_lto_dlink.o` | 10 |
| 12. [compile-lto-host-link](../compile-lto-host-link/README.md) | 只主机链接 → `rdc_lto_app` | 10、11 |

步骤 2–5 是从同一源码**独立开始**的停止点，不是顺次消费前一步文件。步骤 7–9 和 10–12 才是真正的产物依赖链：

```text
rdc_device.cu ─分别编译─ rdc_device.o ─┐
                                     ├─设备链接─ rdc_dlink.o ─┐
rdc_main.cu   ─分别编译─ rdc_main.o   ─┘                       ├─主机链接─ rdc_app
                       两个原始 .o ──────────────────────────┘
```

LTO（链接时优化）仍遵守这三个阶段，只是先保存更高层的中间表示，设备链接时再做跨文件优化。Toolkit 13.4.92 的选项实测差异见 [LTO 编译反例](../compile-lto-compile/README.md)，不要照搬旧脚本的组合。

## 3. 读日志，而不是只看“脚本没报错”

每个动作直接 `source ../record.sh`，不增加通用编译框架。以 `compile-cubin/` 为例：

- `log/cubin.sh`：实际命令，包含 `timeout 60s`；可在该动作目录重放，但直接重放不会更新状态或执行脚本的失败清理。
- `log/cubin.log`：原始 stdout/stderr；不额外伪造成功信息。成功的编译/链接经常没有诊断，空日志不表示未执行。
- `log/status.tsv`：本次命令标签和退出码。正常返回编译器退出码；超时为 124；输入缺失时记录 `input=1`。

每个编译命令最长 60 秒；两个源文件的编译动作各有两条命令。脚本 `set -euo pipefail`，遇错停止；失败会删除本动作可能残留的部分产物。重跑仅覆盖本动作同名日志与固定输出，不清理别人的产物。`status.tsv` 才是本次步骤清单，未列入其中的日志不能冒充本次结果。

本轮开始时 `compile/` 没有旧 `log/`，无需清理历史目录；没有生成自动归档。LTO 首次参数失败被具名保留为其动作内 `log/rejected_flags.*` 与 `log/rejected_flags-status.tsv`，其退出码为 1；它不是当前成功记录，也不会自动重跑。

## 4. 当前离线结果与体积

本轮 [环境检查](../environment/README.md) 确认 RTX 4060 Laptop、计算能力 8.9、driver 596.21；`nvcc --version` 确认 **Toolkit 13.4.92**。本编译课程动作不启动 GPU。所有脚本显式使用 `CUDA_ARCH=sm_89`，不把此架构当全工作区默认值。

全部 12 个脚本均通过 `bash -n`，当前正常路径的 **14 条命令退出码均为 0**。LTO 初次失败的 1 单独保留，未被隐去。产物共享位置固定为项目 `build/sm_89/compile/artifacts/`：

| 动作 | 产物 | 本次字节数 |
| --- | --- | ---: |
| dryrun | 无；`probe_dryrun` 不存在 | — |
| PTX | `probe.ptx` | 1,266 |
| cubin | `probe.cubin` | 5,416 |
| fatbin | `probe.fatbin` | 5,816 |
| 主机目标文件 | `probe.o` | 14,520 |
| 普通主机链接 | `probe` | 22,528 |
| RDC 编译 | `rdc_device.o` / `rdc_main.o` | 4,872 / 9,640 |
| RDC 设备链接 | `rdc_dlink.o` | 11,832 |
| RDC 主机链接 | `rdc_app` | 28,680 |
| LTO 编译 | `rdc_device_lto.o` / `rdc_main_lto.o` | 5,400 / 10,096 |
| LTO 设备链接 | `rdc_lto_dlink.o` | 9,280 |
| LTO 主机链接 | `rdc_lto_app` | 28,680 |

13 个产物合计 **158,026 字节（约 154.3 KiB）**，没有 `keep/` 或持久化预处理 `.ii`。体积是本次文件统计，不是编译日志输出，也不保证跨版本完全相同。三个可执行文件均使用 `-cudart=shared`；额外 `readelf -d` 静态核对确认依赖 `libcudart.so.13`。本轮环境的 `LD_LIBRARY_PATH` 包含 `/opt/cuda/lib64`；这些编译课程产物的程序运行与动态加载仍未验证。

本次直接读 PTX 看到 `.version 9.4`、`.entry saxpy_probe` 和 `fma.rn.f32`；cubin 的编译诊断报告 10 个寄存器、0 字节 spill。编译动作不收集 SASS、内联结果、执行时间或 GPU 正确性，不用文件大小代替这些指标。后续 [binary-sass](../binary-sass/README.md) 和 [nvdisasm-sass](../nvdisasm-sass/README.md) 已独立读取本轮 cubin；跨文件 LTO 内联与性能仍未验证。

## 5. 单变量练习与后续读取

1. 只改 dryrun 中的镜像选择：预测去掉 PTX 镜像后哪些计划行改变；不生成中间文件。
2. 对照源码和 `probe.ptx` 找到线程下标、边界跳转、乘加，解释 float 的地址步长为何是 4 字节。
3. 保持同一份 RDC 源码、优化等级和机器目标，对照普通 RDC 与 LTO；先预测调用是否内联，再由独立 SASS 检查验证。LTO 不保证内联，静态差异不等于性能优势。

后续二进制工具仍可读取既有路径 `probe.ptx`、`probe.cubin`、`probe.fatbin`、`probe.o`、`probe`、`rdc_app`、`rdc_lto_app`，无需重编。独立 cubin 已传 `-Xptxas=--preserve-relocs`，供 `nvdisasm` 使用。检查入口以其自身 README 为准，不在本课程脚本里混入检查或运行。

`--keep`（保存内部中间文件）、`-E`（输出预处理文本）、`--cuda`（输出转换后主机 C++）可用于进阶调查，本路线只介绍而不执行；也不额外导出依赖列表、独立 `.ltoir` 或仅 PTX 的可执行程序。编译器内部的临时预处理是正常编译所需，不等于把大文件保存到课程目录。

修改源码/选项后，手动重做相关编译和下游链接；脚本没有缓存、新鲜度检查或自动依赖调度。只清自身输出也意味着其他旧产物不会自动失效，不能把它们当作新源码结果。

参考：[NVCC](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)、[Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html)。回到 [nvcc 专题入口](../../NVCC_COMPILATION.md)。
