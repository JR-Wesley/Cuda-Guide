# GPGPU-Sim 最小环境

独立官方源码在 `repo/gpgpu-sim_distribution/`，固定起点为 `c3eeb6d19016960ab66789efdd8a3200cc30222a`。构建和运行采用隔离的 Ubuntu 24.04 / CUDA 12.8 Docker 环境，容器只挂载本目录，不使用宿主 CUDA 13.4，不安装宿主系统包，不需要向容器传入 GPU。

官方完整镜像下载遇到 `unexpected EOF` / `network unreachable`。为完成基础运行，默认使用本地精简镜像 `cuda-sim-minimal:cuda12.8-ubuntu24.04`：基于官方 Ubuntu 镜像，在容器内安装 NVIDIA 官方仓库的 CUDA 12.8 最小工具链及编译依赖。它不是上游发布的完整预构建镜像；可复现构建文件在 [`../simulator-setup/Dockerfile`](../simulator-setup/Dockerfile)，下载及构建日志在 [`../simulator-setup/logs/`](../simulator-setup/logs/)。

## 当前入口：先编译，再观察与运行

这里用一个小向量加法验证模拟器能否替代真实 CUDA runtime（负责启动 CUDA 程序的运行库）。编译只生成程序；运行才执行模拟并与 CPU 结果比较。本次入口迁移只做静态检查，以下新脚本尚未进行 Docker 实测。

前提是 Bash、已有 `repo/gpgpu-sim_distribution/` 源码、Docker daemon 可用且当前用户有权限、本地镜像已构建。默认使用 [image.txt](image.txt) 中的固定镜像 ID。若复建镜像得到新 ID，可在下面每个动作前分别加 `SIM_IMAGE=cuda-sim-minimal:cuda12.8-ubuntu24.04`；选择上游完整镜像也必须事先准备好，入口不会自动 pull。完整部署和历史排查见 [DEPLOYMENT_AND_TEST.md](DEPLOYMENT_AND_TEST.md)。

1. **编译，不运行：**

   ```bash
   cd /home/mikasa/hpc/cuda/gpgpu-sim-lab
   bash commands/build/run.sh
   ```

   容器内从 `repo/gpgpu-sim_distribution/` 加载官方 `setup_environment release`，再执行 `make -j2`（最多两个并行任务）构建模拟库。随后用 `nvcc` 编译 `vector_add.cu` 到固定的 `build/vector_add`。`--cudart shared` 选择动态运行库，`compute_70` PTX 是供模拟器读取的虚拟指令。上游构建继续复用 `repo/gpgpu-sim_distribution/build/` 与 `repo/gpgpu-sim_distribution/lib/`，不复制源码或按动作重新建一套 build。

2. **可选：只导出已有程序的 PTX，观察编译目标：**

   ```bash
   bash commands/static/run.sh
   ```

   `cuobjdump --dump-ptx` 从二进制提取 PTX，文本直接写入 `commands/static/output.log`；可在其中查看 `.version` 和 `.target`。此动作只读挂载本项目，不编译、不模拟。导出成功不能代表程序数值正确。

3. **只运行与校验：**

   ```bash
   bash commands/run/run.sh
   ```

   缺少 `build/vector_add` 立即失败，不调用 make/nvcc。运行前用 `ldd` 检查动态库确实解析到 `/work/repo/gpgpu-sim_distribution/lib/` 下的模拟器 `libcudart.so`，防止误用真实 CUDA runtime。官方 QV100 配置复制到 `commands/run/`，模拟器也从那里运行；`timeout 300` 限制本次模拟最多 300 秒。预期 PASS 行如下，另外还要求进程退出 0、周期和指令统计为正。失败时先检查 run 日志中的库路径、错误及退出码，不要把 build 成功当作 run 成功。

每个动作独占 `commands/<action>/run.sh` 和同目录 `output.log`；日志包括展开的命令、镜像 ID、完整输出和最终 `exit_code`，build/run 的容器实现还打印 `container_exit_code`。重跑覆盖本动作日志，不覆盖其他动作；日志仅在实际执行时生成，控制台先提示日志路径。`container.sh` 是同动作容器内实现，不是 dispatcher。

源码、镜像或工具链变化后，需先显式 build 并确认成功；run/static 只使用已有产物，不做哈希或自动新旧判断。不要并发执行同一动作，也不要构建时同时运行。旧顶层 `run.sh`、`container_run.sh` 已移除；不再创建时间戳目录、快照、哈希清单或自动归档。所有容器动作使用 `--network none` 和 `--pull never`，不传入 GPU。

## 校验内容

最小 case 为 257 个浮点元素相加，启动 3 个 block，每个 128 个线程，包含最后一个 block 越界保护。每个输出都和 CPU 期望值逐一比较，所有 CUDA API 及 kernel 返回状态也会检查。期望输出为：

```text
PASS vector_add n=257 blocks=3 threads=128 checked=257 first=1 last=769
```

使用仓库原样提供的 `configs/tested-cfgs/SM7_QV100/` 配置，模拟 compute capability 7.0 的 QV100。这是 PTX execution-driven 的性能模拟，`-gpgpu_ptx_sim_mode 0`；宿主真实 GPU 型号与这一模拟目标无关。模拟周期不等于实机耗时，本实验不做性能精度验证，也没有测量宿主 DRAM 流量或功耗。

`runs/` 下的旧日志、配置、哈希和源码快照全部保留不动，不是当前动作的输出位置。首次下载来源记录在 `setup/`；共享镜像的旧拉取日志在 `../simulator-setup/logs/`，也不会被新入口改写。

官方依据：[GPGPU-Sim README](https://github.com/gpgpu-sim/gpgpu-sim_distribution/blob/c3eeb6d19016960ab66789efdd8a3200cc30222a/README.md)、[官方容器仓库](https://github.com/accel-sim/Dockerfile)。`setup/upstream-README.md` 和 `setup/upstream-devcontainer.json` 是此次拉取版本的副本。

## 历史完成记录（不是新入口验证）

2026-10-24（UTC+8）已实际构建并运行通过。成功记录为 [`runs/20261023T174247Z-90756/`](runs/20261023T174247Z-90756/)，宿主和容器退出码均为 `0`，上游源码无修改。

| 项目 | 历史证据 |
| --- | --- |
| 镜像 | `sha256:75701f3449b84bdc01ad21a26d50fdf94fbebab0a99eb13b0c2d1dcba9672746`，详见 [image.json](runs/20261023T174247Z-90756/image.json) |
| 工具链 | Ubuntu 24.04.5，GCC 13.3.0，CUDA 12.8.93，见 [container.log](runs/20261023T174247Z-90756/container.log) |
| 构建 | `make -j2` 成功，见 [build.log](runs/20261023T174247Z-90756/build.log) |
| 动态库（历史原始路径） | `libcudart.so.12 => /work/repo/lib/gcc-13.3.0/cuda-12080/release/libcudart.so.12`，见 [ldd.log](runs/20261023T174247Z-90756/ldd.log) |
| 数值校验 | `PASS vector_add n=257 blocks=3 threads=128 checked=257 first=1 last=769` |
| 模拟统计 | `gpu_tot_sim_cycle = 5497`，`gpu_tot_sim_insn = 6794`；见 [simulation.log](runs/20261023T174247Z-90756/simulation.log) |

前两次失败及修复也完整保留，见 [过程记录](DEPLOYMENT_AND_TEST.md)。这些模拟统计只说明本 case 已进入并完成模拟，不用于实机性能结论。
