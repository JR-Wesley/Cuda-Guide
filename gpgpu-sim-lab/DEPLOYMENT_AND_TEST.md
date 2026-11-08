# GPGPU-Sim 部署与测试记录

历史实测日期：2026-10-24（UTC+8）；历史 `runs/` 目录名与原始工具输出使用 UTC。当前部署方法与入口已更新，排查和结果仍保留原证据；新脚本只做静态检查，不将历史实测或编译完成等同于新入口运行通过。

## 部署方法

现有工作区已经保留官方源码。来源为 `https://github.com/gpgpu-sim/gpgpu-sim_distribution.git`，提交为 `c3eeb6d19016960ab66789efdd8a3200cc30222a`。首次下载命令、网络失败及成功重试见 [setup/clone-record.txt](setup/clone-record.txt)，仓库身份见 [setup/git-remote.txt](setup/git-remote.txt) 和 [setup/git-commit.txt](setup/git-commit.txt)。在新的空目录获取同一源码可执行：

```bash
mkdir -p repo
git clone https://github.com/gpgpu-sim/gpgpu-sim_distribution.git repo/gpgpu-sim_distribution
git -C repo/gpgpu-sim_distribution checkout --detach c3eeb6d19016960ab66789efdd8a3200cc30222a
```

本次先尝试上游推荐的完整镜像：

```bash
docker pull ghcr.io/accel-sim/accel-sim-framework:ubuntu-24.04-cuda-12.8
```

大镜像下载遇到连接失败后，改用同一 Ubuntu/CUDA 主次版本的精简容器。Dockerfile 固定 Ubuntu 基础镜像 digest，并从 NVIDIA 官方 apt 仓库安装 CUDA 12.8；该本地镜像不等同于上游完整镜像。复建命令：

```bash
cd /home/mikasa/hpc/cuda
docker build -t cuda-sim-minimal:cuda12.8-ubuntu24.04 -f simulator-setup/Dockerfile simulator-setup
cd gpgpu-sim-lab
SIM_IMAGE=cuda-sim-minimal:cuda12.8-ubuntu24.04 bash commands/build/run.sh
SIM_IMAGE=cuda-sim-minimal:cuda12.8-ubuntu24.04 bash commands/static/run.sh  # 可选 PTX 导出
SIM_IMAGE=cuda-sim-minimal:cuda12.8-ubuntu24.04 bash commands/run/run.sh
```

历史已验证镜像的 ID 在 [image.txt](image.txt)。各动作默认使用该 ID；复建得到新 ID 时可用上面的 `SIM_IMAGE` 分别覆盖。每个动作的 `commands/<action>/output.log` 记录所用镜像 ID、命令、输出和最终退出码，同动作重跑覆盖自己的日志，不创建新的 `image.json` 或历史目录。

容器以宿主用户 UID/GID 运行，设置 `--network none --pull never`，仅挂载本实验目录，不传入 GPU，不修改宿主工具链。显式设置容器主机名及 `/etc/hosts` 映射，供上游脚本执行 `hostname -d/-f`。所有安装动作只发生于镜像构建阶段。

## 验证方法与选择依据

实际命令分为三个独立入口：

- [build](commands/build/run.sh) 及其 [容器实现](commands/build/container.sh)：记录工具版本和源码提交，从仓库根目录 `source ./setup_environment release`，执行 `make -j2`，再编译测试程序到共享的 `build/vector_add`。
- [static](commands/static/run.sh)：对已有 `build/vector_add` 执行 `cuobjdump --dump-ptx`，输出进入本动作日志，不编译或运行。
- [run](commands/run/run.sh) 及其 [容器实现](commands/run/container.sh)：检查已有程序和模拟动态库，使用官方配置运行并核对数值与模拟统计，不隐式重建。

源码和上游 build/lib 保持原位置，不按动作复制。当前不自动生成哈希、源码快照、时间戳目录或归档；旧顶层入口已经移除。更换镜像或源码后应显式 build 成功再 run，具体步骤见 [README](README.md)。

测试程序使用 CUDA 12.8 编译 `compute_70` PTX，并通过 `--cudart shared` 链接 CUDA runtime。`ldd` 必须显示 `libcudart.so` 来自 `/work/repo/gpgpu-sim_distribution/lib/`；否则脚本立即失败，防止误把真实 CUDA runtime 的行为当作模拟。

采用官方 `configs/tested-cfgs/SM7_QV100/` 配置，PTX timing mode 为 `-gpgpu_ptx_sim_mode 0`，目标 capability 为 70。选择该配置是为了与 `compute_70` PTX 一致，并保持上游配置原样。宿主 GPU 不参与模拟，未以宿主 GPU 型号替换模拟目标。

`vector_add.cu` 使用 257 个 float、3 个 block、每 block 128 个线程。输入为 `a[i]=i`、`b[i]=2*i+1`，逐元素 CPU 校验应得到 `c[i]=3*i+1`；这些小整数可被 float 精确表示。257 让最后一个 block 同时覆盖有效线程与越界保护。检查每个 CUDA API、kernel 错误与同步，校验全部 257 个输出。运行超时设为 300 秒，脚本启用 pipeline 状态传播；通过条件同时包含进程成功、精确 PASS 行、模拟器周期/指令统计。

## 排查与修复记录

| 尝试 | 观察证据 | 修复与依据 |
| --- | --- | --- |
| 初次克隆 | 沙箱代理连接失败，退出 128 | 相同官方 URL 在网络授权范围内重试成功，保留 [clone-record.txt](setup/clone-record.txt) |
| 官方完整镜像 | 下载错误 `unexpected EOF` / `network unreachable` | 使用官方 Ubuntu 基础镜像与 NVIDIA 官方包搭建精简容器，记录于 [共享 Dockerfile](../simulator-setup/Dockerfile) 和 [共享日志](../simulator-setup/logs/) |
| [20261023T174108Z-87509](runs/20261023T174108Z-87509/container.log) | 上游 `hostname -d` 返回 `Temporary failure in name resolution`，容器和宿主脚本均退出 1 | 为无网络容器添加本机 hosts 映射；保留网络隔离。还将 source 的工作目录设为 repo，修正上游使用 `git --git-dir` 时因外部工作目录产生的错误修改计数 |
| [20261023T174129Z-88534](runs/20261023T174129Z-88534/build.log) | `cuda_runtime_api.cc:133:10: fatal error: cudaProfiler.h: No such file or directory`，退出 2 | 在容器镜像中补装 NVIDIA `cuda-profiler-api-12-8`，保持上游源码不变；更新固定镜像 ID 后增量续构建 |
| [20261023T174247Z-90756](runs/20261023T174247Z-90756/container.log) | 构建、编译、动态库检查及模拟全部完成，宿主与容器退出码均为 0 | 全部 257 个输出通过 CPU 校验，模拟器正常退出 |

编译中上游代码的警告保留在完整构建日志中，没有通过删除校验或修改上游源代码绕过失败。

## 实际成功结果

成功运行采用 `sha256:75701f3449b84bdc01ad21a26d50fdf94fbebab0a99eb13b0c2d1dcba9672746` 镜像、Ubuntu 24.04.5、GCC 13.3.0 与 nvcc 12.8.93。PTX 导出为 `.version 8.7`、`.target sm_70`，见 [vector_add.ptx.txt](runs/20261023T174247Z-90756/vector_add.ptx.txt)。

```text
PASS vector_add n=257 blocks=3 threads=128 checked=257 first=1 last=769
gpu_tot_sim_cycle = 5497
gpu_tot_sim_insn = 6794
```

结果来自 [simulation.log](runs/20261023T174247Z-90756/simulation.log)。[host.exitcode](runs/20261023T174247Z-90756/host.exitcode) 和 [container.exitcode](runs/20261023T174247Z-90756/container.exitcode) 均为 0；[ldd.log](runs/20261023T174247Z-90756/ldd.log) 确认 CUDA runtime 解析到本次构建的模拟库。[source.diff](runs/20261023T174247Z-90756/source.diff) 为空，当前 `git -C repo status --short` 也为空；无需修改官方模拟器源码。成功目录下执行 `sha256sum -c input.sha256`，程序、测试源码和配置三项均为 OK。

成功构建复用了前轮在同一源码/编译器下完成的对象文件；前轮完整编译输出仍保存在失败目录。这是正常增量构建，未将前轮构建失败记作成功。

## 证据范围

历史独立运行目录保存 `host.log`、`container.log`、两个退出码、`build.log`、`nvcc.log`、`ldd.log`、`simulation.log`、`image.json`、源码 tar、源码差异、脚本/测试副本、配置及输入校验和。这些记录保留原样，不由新入口更新。失败目录在对应失败点停止，因此不应期待失败目录包含尚未执行阶段的日志。

本 case 只验证环境、PTX 加载、模拟执行和简单数值正确性。模拟周期是模拟器统计，不是宿主 GPU 实测延迟。性能精度、DRAM 实际流量、功耗与其他 CUDA API 的覆盖均未测量。
