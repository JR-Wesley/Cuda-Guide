# 00 · Machine Explorer

对应 [M0](../LEARNING_PATH.md)，并承接原环境主题。先建立本机的 CPU、互连、GPU、SM、缓存和显存逻辑模型，再设计性能实验。这里提供教学与采集清单，尚无独立机器探测脚本或本次新记录；逻辑模型见 [体系结构基础](../ARCHITECTURE.md)。

| 主题 | 计划观察 | 完成时留下的证据 |
| --- | --- | --- |
| CPU 与系统 | CPU 型号、核心/线程、内存、OS/WSL 情况。 | `lscpu` 等原始输出与采集时间。 |
| GPU 与软件 | 设备名、计算能力、驱动、Toolkit、分析工具版本。 | `nvidia-smi -q`、`nvcc --version` 等原始日志。 |
| 设备属性 | SM、warp、线程、寄存器、shared、L2、异步引擎等可查询属性。 | CUDA Runtime 属性查询结果与 API/字段名。 |
| 拓扑 | CPU ↔ GPU、PCIe/NVLink 可见信息。 | `lspci -vv`、`nvidia-smi topo -m`，并注明虚拟化视图限制。 |

## 环境清单如何影响程序

- **设备与工具链**：`nvidia-smi` 中的 CUDA 版本字段不是已安装 `nvcc` 的版本；分别记录驱动、Toolkit、Runtime 和分析器版本。程序实际使用的可见 device 编号及其属性也要保存，不能仅看系统中第一张 GPU。
- **资源上限**：设备的每 SM/每 block 资源、kernel 编译资源、launch 请求是三个层次。设备支持较大 shared memory，不等于任意 kernel 可未经 opt-in 就按上限启动；见 [carveout 与 dynamic shared](../04_sync/03_thread_scopes/README.md)。
- **构建目标**：先确认设备 compute capability 与 Toolkit 支持的 target，再选择子专题的 `CUDA_ARCH`。交叉编译只能验证编译路径；驱动能加载 cubin 不证明其 PTX JIT 路径也可用。架构专用后缀另行检查。
- **系统内存与互连**：pinned、mapped、managed 是不同分配/访问方式。CPU/GPU 并发、peer 原子和异步传输需核对相应属性，不能用“统一地址”或 system scope 推导全支持。
- **工具与权限**：检查 `cuobjdump`、Compute Sanitizer、Nsight Systems、Nsight Compute 的实际可用性。缺工具或计数器权限时保留失败原因；本任务不授权修改驱动、权限或重置 GPU。

Linux/WSL 可按表中命令采集；其他系统采用可用的等价查询并记录差异，不把 `lscpu`/`lspci` 假定为所有平台通用。WSL/虚拟化中的拓扑可能只显示虚拟连接，不能据此计算物理 PCIe 带宽。

## 最小程序应具备的环境检查

进入具体专题，先核对其入口与参数；本实验室没有统一构建命令。程序至少应检查设备选择、内存分配、拷贝、launch 参数和完成点错误，再校验数值。launch 后的即时错误检查不等于 kernel 已成功执行；后续同步才可能暴露执行期错误。计时时将诊断同步与正式测量边界区分开。

把支持状态按“编译 / 原生执行 / JIT / sanitizer / profiler”分别记录；可选路径的 `SKIPPED` 不计作通过。高级特性映射见 [执行](../01_execution/README.md)、[同步](../04_sync/README.md)、[GEMM](../06_gemm/README.md) 和 [异步](../07_async/README.md)。

在机器图上区分**工具观测值、官方规格与推断**。[SDK 当前环境检查说明](../../cuda-sdk-lab/experiments/environment/README.md)提供采集方法。原链接的环境记录已不可用，不能继续用于版本对照或实测证据；当前 README 也不代表已经完成测量。当前没有本目录的运行记录。
