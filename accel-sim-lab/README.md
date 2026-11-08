# Accel-Sim 基础环境与最小 trace 重放

本目录在 Docker 中构建和运行 Accel-Sim；只挂载本目录，容器以当前用户 UID/GID 运行，不需要 GPU 直通或宿主 CUDA。源码、trace、构建产物和证据均留在本目录。

部署方法、失败尝试与修复依据见 [DEPLOYMENT.md](DEPLOYMENT.md)。

2026-10-24 已完成实际构建与基础运行：Pathfinder 的 4/4 个 kernel 重放通过，退出码 0；总计 28,717 模拟周期、1,059,640 条模拟指令。证据：[构建日志](runs/build-20261023T174300Z-CFDbzt/output.log)、[运行日志](runs/run-20261023T174414Z-Ikj6aG/output.log)、[检查结果](runs/run-20261023T174414Z-Ikj6aG/summary.json)。

## 来源与固定版本

- [Accel-Sim 官方仓库](https://github.com/accel-sim/accel-sim-framework)：`1c9cc831c5d948f89fa9dc8f21f6c87952f5f2ea`。
- [Accel-Sim 使用的 GPGPU-Sim 后端](https://github.com/accel-sim/gpgpu-sim_distribution)：`c24e8858eee552b3bc2919b147f14e2a88571775`。
- [pybind11](https://github.com/pybind/pybind11)：`fdec8c350aa0c7fbb552ec70d8176f8a13eb0561`（配置时依赖；本次只构建 CLI 和 cudart）。
- 最初尝试[官方完整镜像](https://github.com/accel-sim/Dockerfile)，但大层下载遇到 `unexpected EOF` / `network unreachable`。基础运行改用官方 Ubuntu 24.04 基础镜像和 NVIDIA 官方 CUDA 12.8 软件包构建的精简隔离环境；Dockerfile 和镜像构建证据见 [`../simulator-setup/`](../simulator-setup/)。这不是 Accel-Sim 官方预构建镜像；本地固定镜像 ID 见 `image.txt`。
- Trace 来自官方 `repo/accel-sim-framework/short-tests.sh` 和 `.github/scripts/main-docker.sh` 所用的 [V100 Rodinia 2.0-ft 包](https://engineering.purdue.edu/tgrogers/accel-sim/traces/tesla-v100/latest/rodinia_2.0-ft.tgz)，约 21 MB。采集时 CUDA 9.1；这个版本是 trace 来源，不是本次容器编译工具链。

## 当前入口：准备、编译、运行分开

目标是先把源码编译成模拟器，再让已有模拟器读取 trace（记录好的 GPU 指令序列）。构建成功不等于重放通过。前提是 Bash、Docker 和已构建的本地镜像；重放检查还需要宿主 `python3`。本次迁移仅做静态检查，下述新入口尚未进行容器实测；上文数值仍属于原历史记录。

1. **首次准备，已有源码和 trace 时跳过。** 从本目录执行，两项独立动作需要网络；不会启动 Docker：

   ```bash
   cd /home/mikasa/hpc/cuda/accel-sim-lab
   bash commands/prepare-source/run.sh
   bash commands/prepare-traces/run.sh
   ```

   `prepare-source` 只在缺少克隆时获取固定提交；已有克隆只核对提交并显示工作树状态，不 reset 或覆盖本地改动。`prepare-traces` 复用已有压缩包，否则下载到固定 `.part` 文件，检查 gzip 完整性后改名；解压跳过已有文件，包括 trace 包自带的 `run.sh`。它不计算哈希、不自动重试，也不更新 `downloads/SHA256SUMS`。因此 gzip 检查只证明压缩数据完整，**不保证上游 `latest` 与历史包内容相同**；不应将新下载的结果直接归于历史输入。

2. **只编译，不运行：**

   ```bash
   bash commands/build/run.sh
   ```

   使用官方 CMake 路径，`-j2` 表示最多两个并行编译任务。复用 `repo/accel-sim-framework/gpu-simulator/build/release/`，安装产物仍在上游规定的 `repo/accel-sim-framework/gpu-simulator/bin/release/` 等位置；不把源码或 build 复制到动作目录。

3. **只运行已有产物并核对重放结果：**

   ```bash
   bash commands/run/run.sh
   ```

   缺少可执行文件直接报错，不隐式编译或下载。使用 `SM7_QV100` 基础配置及对应 `trace.config`；运行所需配置放在 `commands/run/`，从该目录执行模拟。检查器要求完成全部预期 kernel、正常结束标记及正的周期/指令统计；失败时先查看本动作日志末尾的命令和退出码。

每个动作只有自己的 `commands/<action>/run.sh` 与同目录 `output.log`。日志包含展开后的命令、输出、镜像 ID（容器动作）及最终 `exit_code`；容器内部另打印 `container_exit_code`。同一动作重跑覆盖自己的日志，不影响其他动作。日志在执行时生成，不预填成功结果；运行会打印日志路径，输出写入该文件。不要并发执行同一动作，也不要一边构建一边运行。

`commands/build/container.sh` 和 `commands/run/container.sh` 是同一动作的容器内实现，不是另一个多动作入口。容器使用当前 UID/GID、`--network none`（关闭容器网络）及 `--pull never`（不自动拉镜像），只挂载本项目，不传入 GPU。`run` 不判断源码或镜像是否比二进制新；修改源码、工具链或镜像后，应先显式 build 并确认退出码 0，再运行。

首次在其他机器复现时，先按 [`../simulator-setup/`](../simulator-setup/README.md) 构建镜像，再更新 `image.txt`，或为 build/run 分别设置 `SIM_IMAGE=cuda-sim-minimal:cuda12.8-ubuntu24.04`。旧 `lab.sh` 已移除；不再产生时间戳目录、自动快照、哈希清单或归档。`runs/`、`logs/`、`downloads/SHA256SUMS` 是保留不动的历史证据，不是当前动作的日志。

基础 case 为 Rodinia Pathfinder：`pathfinder 1000 20 5`，使用包内完整的 4 个 kernel trace（共约 1.95 MB），路径见 `case.txt`。该输入是这个官方短测试包中 trace 总大小最小的完整应用输入；没有截断 kernel 或 trace。

## 结果边界

Trace 重放通过表示 trace 已被模拟器完整读取、所有预期 kernel 完成，并产生非零周期和指令统计；这不等于重新运行 CUDA 程序并验证数值结果。本次不测真实 GPU 性能、不新采集 SASS trace，也不测功耗。周期数属于 QV100 模型，不能解释为宿主 RTX 4060 的性能。
