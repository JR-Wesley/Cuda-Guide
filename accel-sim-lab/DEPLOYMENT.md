# Accel-Sim 部署与测试记录

历史实测日期：2026-10-24（UTC+8）。历史 `runs/` 目录名称使用 UTC，所以记录可能以 `20261023T...Z` 开头。下面“部署方法”已更新为独立动作入口；其余排查与结果仍引用当时证据，不代表新脚本已实测。

## 部署方法

环境构建由工作区的 [`simulator-setup/`](../simulator-setup/) 统一管理。使用官方 Ubuntu 24.04 基础镜像与 NVIDIA 官方仓库的 CUDA 12.8 软件包，安装仅发生在 Docker 镜像内。Accel-Sim、其 GPGPU-Sim 后端及 pybind11 的固定提交见 [README.md](README.md)。

在本机，依赖镜像已经存在时直接运行：

```bash
cd /home/mikasa/hpc/cuda/accel-sim-lab
bash commands/prepare-source/run.sh  # 已有固定源码时跳过
bash commands/prepare-traces/run.sh  # 已有 trace 时跳过
bash commands/build/run.sh
bash commands/run/run.sh
```

`prepare-source` 获取三份固定 commit 源码；`prepare-traces` 获取并解压官方短测试 trace 包。首次准备需要网络；两个动作互不调用。当前不做内容哈希校验，gzip 完整性检查不能证明下载的 `latest` 包与历史包相同。`image.txt` 存放实际使用的镜像 ID；复制到其他机器后，应先按共享环境文档构建镜像，再记录该机器的 ID。不能将本地镜像 ID 当成 Docker Hub/GHCR 上可拉取的镜像地址。

`commands/build/run.sh` 和 `commands/run/run.sh` 分别直接启动本动作容器：`--network none --pull never --user UID:GID`，仅挂载本目录到 `/work`，没有 `--gpus`。构建调用官方 CMake 配置，目标是 CLI `accel-sim.out` 和 `cudart`，并发数为 2。容器工具链为 CUDA 12.8；trace 本身由 CUDA 9.1 应用在 V100 上采集，两者用途不同。

运行使用 `SM7_QV100/gpgpusim.config` 加对应 `trace.config`，完整重放 Pathfinder `1000 20 5` 的 4 个 kernel。源码与上游固定 build/bin 目录由动作共享；run 不编译，缺产物即失败。每个动作的命令、输出和最终退出码写入自己的 `commands/<action>/output.log`，同动作重跑覆盖。运行检查摘要直接写进 run 的日志。当前不生成源码快照、时间戳目录、哈希或自动归档；配置副本只是 `commands/run/` 中实际运行所需的输入。

下面历史链接中的 `source/`、`image.json`、哈希文件、`summary.json` 等是旧流程产物，保留不动。新入口的操作顺序、日志及限制见 [README](README.md#当前入口准备编译运行分开)。

## 排查记录与决策摘要

下面记录可复核的现象、修复及选择理由。

| 阶段 | 观察与处理 | 证据 |
| --- | --- | --- |
| 源码获取 | 初次 clone 无法通过沙箱代理连接 GitHub，退出 128；取得网络执行权限后同一仓库 clone 成功，退出 0。未改变系统网络配置。 | [初次日志](logs/clone.log)、[重试日志](logs/clone-network.log)、[获取命令](logs/acquisition-commands.txt) |
| Trace 获取 | HTTPS 下载遇到 OpenSSL `unexpected eof while reading`，两次已记录尝试退出 56；使用 `curl -C -` 续传。后续下载日志到达 100%，但该进程结束码未被旧会话保存，不补写推测值。 | [首次下载](logs/trace-download.log)、[续传失败](logs/trace-download-resume.log)、[最终续传](logs/trace-download-resume2.log) |
| Trace 验证 | 恢复会话后 `gzip -t`、解包均成功；固定 SHA256 为 `aeb7de478785856e4ac834d12be0e71ab0df297f43fc02650e2ca90dea66d8b1`。随后正式 `prepare` 退出 0。 | [prepare 日志](logs/prepare-lRCcs7Pm.log)、[退出码](logs/prepare-lRCcs7Pm.log.exitcode)、[SHA256](downloads/SHA256SUMS) |
| 隔离环境 | 官方完整镜像大层下载失败；改用官方基础镜像和官方 CUDA 软件包构建精简镜像。选择的依据是本次只需要模拟器构建与离线 trace 重放，不需要 cuDNN 或完整 benchmark 数据。 | [共享环境记录](../simulator-setup/) |
| 首次构建 | CMake 配置完成，但 `cuda_runtime_api.cc:141` 缺少 `cudaProfiler.h`，构建退出 2。该头文件属于额外的 CUDA profiler API 包，需要补充镜像依赖，保留原始错误。 | [构建日志](runs/build-20261023T174108Z-GXX4OS/output.log)、[退出码](runs/build-20261023T174108Z-GXX4OS/exitcode) |
| 依赖修复与构建 | 镜像补装 `cuda-profiler-api-12-8=12.8.90-1`；继续官方 CMake 构建，CLI 和 cudart 安装成功，退出 0。模拟器源码没有修改。 | [成功构建日志](runs/build-20261023T174300Z-CFDbzt/output.log)、[退出码](runs/build-20261023T174300Z-CFDbzt/exitcode) |
| 基础运行 | Pathfinder 4/4 kernel 正常完成；总计 28,717 模拟周期、1,059,640 条模拟指令，模拟器与检查脚本共同退出 0。 | [原始运行日志](runs/run-20261023T174414Z-Ikj6aG/output.log)、[结构化结果](runs/run-20261023T174414Z-Ikj6aG/summary.json)、[退出码](runs/run-20261023T174414Z-Ikj6aG/exitcode) |

选择最小 case 时，比较了解包后各个完整应用输入的 `.traceg` 总大小：Pathfinder 为 1,953,393 字节，4 个 kernel，是该短测试包中最小的完整输入。这个选择保留应用的 kernel 序列和复制命令，没有删除 trace 指令来缩短运行。

## 已验证结果

本次使用镜像 ID `sha256:75701f3449b84bdc01ad21a26d50fdf94fbebab0a99eb13b0c2d1dcba9672746`；容器内 Ubuntu 24.04.5、GCC 13.3.0、NVCC 12.8.93、CMake 3.28.3，详见成功构建日志。模拟输出中 `kernel_launch_uid` 依次为 1、2、3、4，kernel 名称均为 `_Z14dynproc_kerneliPiS_S_iiii`。最后两个标记为 `simulation thread exiting` 和 `exit detected`。

完整运行命令保存在 [command.sh](runs/run-20261023T174414Z-Ikj6aG/command.sh)。输入、配置、源码版本和二进制检查记录都位于该运行目录。周期数属于 QV100 模拟模型；日志显示的约 2 秒是本次 CPU 模拟执行耗时，不能当作应用在真实 GPU 上的 kernel 时间。

## 判定边界

`commands/run/check_run.py` 根据输入列表要求全部预期 kernel 完成（当前历史 case 为 4 个），程序打印模拟线程结束及退出标记，周期数和指令数为正，并排除已知中途终止文本；外层还必须退出 0。单纯编译成功不算运行成功。

Trace 重放不重新计算 CUDA 应用输出，因此数值正确性未测量。本任务验证基础模拟链路，不测真实 GPU 性能、模拟精度或功耗，也没有为当前宿主 GPU 采集新 trace。
