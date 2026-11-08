# 环境检查：先确认工具，不运行应用

## 要解决的问题

工具、GPU、驱动和编译目标是不同条件。先确认当前命令调用到的版本，再决定能否编译、加载或采集；不能沿用以前“ncu 没安装”的判断，也不能只看版本号就认为采集成功。

## 1. 底层查询

当前目录：`cuda-sdk-lab/experiments/environment/`。

```bash
nvcc --version
ncu --version
nvidia-smi --query-gpu=index,name,compute_cap,driver_version --format=csv,noheader
```

`--version` 查询工具本身；`--query-gpu` 指定设备索引、名字、计算能力和驱动字段；CSV 是逗号分隔文本，`noheader` 不打印表头。`nvidia-smi` 的 CUDA Version 表示驱动能力，不是安装的 nvcc 版本。

## 2. 独立环境入口

```bash
bash check.sh
```

本脚本只查询 `nvcc`、`ptxas`、`nvlink`、`cuobjdump`、`nvdisasm`、`compute-sanitizer`、`nsys`、`ncu`、`nvidia-smi` 及设备信息，不编译、不启动 CUDA 工作负载、不改系统配置。可通过同名大写环境变量指定工具（如 `NVCC`、`NVDISASM`、`NSYS`；Sanitizer 使用 `COMPUTE_SANITIZER`）。

在当前目录写 `log/<步骤>.sh` 命令、同名 `.log` 输出及 `log/status.tsv` 退出码。重跑覆盖本动作同名记录；缺失工具记 `SKIPPED`，不是通过。脚本不切换目录。

## 3. 本次检查

本轮清理此前环境日志后重新执行，以下来自当前记录，而非旧环境推测：

| 项目 | 本次结果 | 原始证据 |
| --- | --- | --- |
| GPU / 计算能力 / 驱动 | RTX 4060 Laptop / 8.9 / 596.21 | [gpu_info.log](log/gpu_info.log) |
| nvcc | Toolkit 13.4，13.4.92 | [nvcc_version.log](log/nvcc_version.log) |
| nvdisasm | 13.4.92 | [nvdisasm_version.log](log/nvdisasm_version.log) |
| Nsight Compute | 2026.3.1.0 | [ncu_version.log](log/ncu_version.log) |
| Nsight Systems | 2026.3.2 | [nsys_version.log](log/nsys_version.log) |
| 其余工具查询 | 成功，详见逐项退出码 | [status.tsv](log/status.tsv) |

本次十项查询均退出 0。性能计数器采集的验证在 [ncu](../ncu/README.md)，不是由本动作证明；PTX JIT 的真实兼容性失败在 [module-driver-ptx](../module-driver-ptx/README.md)。离线 CFG 已能生成，但本轮未找到 Graphviz 的 `dot`，没有渲染图片或安装额外软件。

## 4. 练习

为什么 nvcc 13.4 能编译成功，而驱动仍可能无法加载其 PTX？分别说出源码编译、PTX 语言版本与驱动 JIT 的职责。定位时保留原始错误，不直接要求升级或修改系统。
