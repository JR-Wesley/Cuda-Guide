# CUDA 模拟器隔离环境

此目录为 `../accel-sim-lab` 与 `../gpgpu-sim-lab` 提供共享 Docker 构建环境。源码、构建产物及运行证据分别放在两个实验目录。

## 本目录的维护边界

这里提供共享 `Dockerfile` 和部署说明，**没有自维护执行脚本或需要迁移的多动作入口**，不为导航补造 dispatcher。`upstream.Dockerfile` 是上游参考副本，`logs/` 是已有部署证据，均保留不动。下方 Docker 命令是手动操作说明，不会因阅读文档或运行两套模拟器的 run 动作而自动执行。

两套实验已采用各自的 `commands/<action>/run.sh` 和同目录 `output.log`；编译与运行分开，共享已有源码和 build。当前入口迁移仅经静态检查，没有启动 Docker、重建镜像或重跑模拟。以下镜像版本与数值均是历史记录，不保证本机现在仍具备相同环境。

## 历史环境

- 本地镜像：`cuda-sim-minimal:cuda12.8-ubuntu24.04`，约 1.12 GB。
- 历史最终镜像 ID：`sha256:75701f3449b84bdc01ad21a26d50fdf94fbebab0a99eb13b0c2d1dcba9672746`。
- 官方 Ubuntu 24.04 基础镜像固定 digest；CUDA 软件包来自 NVIDIA 官方 apt 源。
- `nvcc` 12.8.93，`cuobjdump` 12.8.90，`cuda-profiler-api-12-8` 12.8.90-1。最终全部包版本见 [container-packages-profiler.tsv](logs/container-packages-profiler.tsv)。
- 容器运行使用当前用户 UID/GID，仅挂载目标实验目录，不要求 GPU 透传；构建与模拟阶段关闭容器网络。

这是本地精简镜像，并非 Accel-Sim 发布的完整镜像。依赖选择依据 [官方 Dockerfile](https://github.com/accel-sim/Dockerfile/blob/Ubuntu-24.04-cuda-12.8/Dockerfile) 及两套模拟器实际构建文件；下载时的官方文件保存在 [upstream.Dockerfile](upstream.Dockerfile)。

官方完整镜像 `ghcr.io/accel-sim/accel-sim-framework:ubuntu-24.04-cuda-12.8` 的压缩层合计约 11.42 GiB，本次下载遇到 `unexpected EOF` / `network is unreachable`。为优先完成基础运行，改用此精简环境；原始证据见 [manifest](logs/image-manifest.json)、[下载日志](logs/docker-pull.log) 和 [Docker 诊断](logs/docker-download-diagnostics.log)。没有安装或修改宿主 CUDA、驱动及系统软件包。

## 重建

从工作区根目录执行：

```bash
docker build -t cuda-sim-minimal:cuda12.8-ubuntu24.04 simulator-setup
```

构建需要网络和 Docker daemon 访问权限。基础镜像已固定 digest，但 apt 源中的包可能更新，重建后镜像 ID 不保证相同；本次确切版本、镜像 metadata 和构建输出分别保留在 `logs/` 中。最终成功构建日志为 [minimal-image-profiler-build.log](logs/minimal-image-profiler-build.log)，退出码为 `0`，镜像 metadata 见 [minimal-image-profiler.json](logs/minimal-image-profiler.json)。

运行入口和各自的验证结果见 [Accel-Sim](../accel-sim-lab/README.md) 与 [GPGPU-Sim](../gpgpu-sim-lab/README.md)。

## 部署与排查记录

以下是可复核的操作与决策摘要；原始记录保留在链接的日志中。

| 步骤 | 观察或测试 | 处理与依据 | 证据 |
| --- | --- | --- | --- |
| 宿主检查 | Arch Linux，CUDA 13.4，Docker 29.8.2，RTX 4060 Laptop cc 8.9 / driver 596.21 | 两项目在容器中使用 CUDA 12.8；模拟本身无需宿主 GPU。最初沙箱内 `nvidia-smi` 被限制，随后只读宿主查询成功，不能据最初错误判断没有 GPU | [host-environment.log](logs/host-environment.log) |
| 官方完整镜像下载 | 压缩层约 11.42 GiB，Docker 报 `unexpected EOF` / `network is unreachable` | 原下载会话中断后检查实际进程与镜像，确认未完成才重试；重试仍受网络影响，停止本任务 pull，保留失败证据，改用官方来源的软件包构建精简容器 | [首次日志](logs/docker-pull.log)、[续拉日志](logs/docker-pull-resume.log)、[诊断](logs/docker-download-diagnostics.log) |
| 首次本地构建命令 | `docker build --progress=plain ...` 返回 `unknown flag: --progress`，退出 125；当前未安装 buildx | 改用本机已有传统构建器 `docker build`，无需安装宿主组件 | [失败日志](logs/minimal-image-build.log)、[退出码](logs/minimal-image-build.exitcode) |
| 精简镜像构建 | Ubuntu 编译依赖及 NVIDIA CUDA 最小工具链安装成功，`nvcc --version` / `cuobjdump --version` 成功，退出 0 | 初始镜像约 1.12 GB，可以开始真实编译；此阶段仅证明工具链可用 | [构建日志](logs/minimal-image-build-retry.log)、[初始镜像](logs/minimal-image.json) |
| 真实编译暴露缺失依赖 | 上游 runtime shim 包含 `cudaProfiler.h`，最小 CUDA 安装未带该头文件 | 在 Dockerfile 补 `cuda-profiler-api-12-8`，由官方包提供头文件，不删除源码检查。最终镜像构建退出 0 | [补依赖构建日志](logs/minimal-image-profiler-build.log)、[Accel-Sim 排查记录](../accel-sim-lab/DEPLOYMENT.md) |
| 模拟器验证 | 必须完成真实模拟，并检查退出码与结果 | GPGPU-Sim 使用有 CPU 校验的向量加法；Accel-Sim 回放完整官方 trace。trace 回放通过只证明模拟完成，不等同于计算输出数值校验 | 各实验目录的 README、历史 runs 日志与配置快照 |

### 实际命令

镜像拉取与诊断（首次拉取未完成）：

```bash
docker pull ghcr.io/accel-sim/accel-sim-framework:ubuntu-24.04-cuda-12.8
docker image inspect public.ecr.aws/docker/library/ubuntu:24.04
journalctl -u docker --since '40 minutes ago' --no-pager
```

精简环境构建与验证（最终成功）：

```bash
docker build -t cuda-sim-minimal:cuda12.8-ubuntu24.04 simulator-setup
docker image inspect cuda-sim-minimal:cuda12.8-ubuntu24.04
docker run --rm --network none --user "$(id -u):$(id -g)" \
  --entrypoint dpkg-query cuda-sim-minimal:cuda12.8-ubuntu24.04 -W
```

`Dockerfile` 内已经执行工具链版本检查。原始日志与退出码来自实际命令；没有完整退出码的中断下载不记为成功。两套模拟器各自的部署文档记录 clone、build、run 命令及后续失败/修复。

## 历史基础运行结果

2026-10-24（UTC+8）两套模拟器均已从本次拉取的源码构建并实际运行。两者都使用官方 `SM7_QV100` 模型，模拟周期不能当作 RTX 4060 的实测性能。

| 项目 | 实际 case | 通过证据 | 统计与边界 |
| --- | --- | --- | --- |
| GPGPU-Sim | 自带 `vector_add.cu`，257 元素，3×128 线程 | [模拟日志](../gpgpu-sim-lab/runs/20261023T174247Z-90756/simulation.log) 的 `PASS ... checked=257`；[容器退出码](../gpgpu-sim-lab/runs/20261023T174247Z-90756/container.exitcode) 和 [宿主脚本退出码](../gpgpu-sim-lab/runs/20261023T174247Z-90756/host.exitcode) 均为 0；[动态库证据](../gpgpu-sim-lab/runs/20261023T174247Z-90756/ldd.log) 指向模拟器 libcudart | 5,497 模拟 cycles，6,794 模拟 instructions；全部元素逐一 CPU 校验通过 |
| Accel-Sim | 官方 Pathfinder `1000 20 5` trace，完整 4 个 kernel | [运行日志](../accel-sim-lab/runs/run-20261023T174414Z-Ikj6aG/output.log)、[退出码 0](../accel-sim-lab/runs/run-20261023T174414Z-Ikj6aG/exitcode)、[摘要 PASS / 4 of 4](../accel-sim-lab/runs/run-20261023T174414Z-Ikj6aG/summary.json) | 28,717 模拟 cycles，1,059,640 模拟 instructions；trace 重放完成，未重新计算应用输出，数值正确性未测量 |

当前独立入口（工作区根目录；先准备好镜像、源码和 trace）：

```bash
# 编译动作：只生成/更新共享产物，不运行模拟
bash gpgpu-sim-lab/commands/build/run.sh
bash accel-sim-lab/commands/build/run.sh

# 运行动作：只消费已有产物，缺失时失败，不隐式编译
bash gpgpu-sim-lab/commands/run/run.sh
bash accel-sim-lab/commands/run/run.sh
```

各动作输出覆盖自己的 `commands/<action>/output.log`，不向旧 `runs/` 或本目录 `logs/` 写入。GPGPU-Sim 另有可选 PTX 导出 `bash gpgpu-sim-lab/commands/static/run.sh`。同一动作不要并发执行；更换镜像后要为各动作指定一致的 `SIM_IMAGE` 并显式重新 build。

完整部署与失败排查记录：[GPGPU-Sim](../gpgpu-sim-lab/DEPLOYMENT_AND_TEST.md)、[Accel-Sim](../accel-sim-lab/DEPLOYMENT.md)。最终上游源码工作树均无修改；所有环境适配都在实验包装脚本和 Dockerfile 中。
