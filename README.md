# CUDA 学习与实验工作区

这里按主题存放本地编写的 CUDA 学习材料、实验脚本与运行记录；克隆的顶层第三方仓库统一放在 [`third-party/`](third-party/) 下。**这不是一个统一构建的项目**：各目录有各自的依赖、构建方式和验证入口，请先阅读对应 README。

## 本地目录

| 目录 | 内容 |
| --- | --- |
| [`ncu/`](ncu/README.md) | Nsight Compute 系统学习、第一课实操与 CUDA/CUTLASS/CuTe 示例；按动作独立执行、就地记录日志。 |
| [`gpu-architecture-lab/`](gpu-architecture-lab/README.md) | 统一的 CUDA 编程与 GPU 体系结构实验室：编程模型、执行/存储层次、指令、同步与异步；含 9 个可执行专题、历史证据和微架构学习路径。 |
| [`cuda-sdk-lab/`](cuda-sdk-lab/README.md) | CUDA Toolkit 编译、反汇编、性能工具及矩阵转置等练习。 |
| [`cuda-bpg-lab/`](cuda-bpg-lab/README.md) | BPG（CUDA Best Practices Guide，最佳实践指南）学习实验：七阶段从正确性、计时到优化；按功能独立运行脚本，历史记录留在本项目内。 |
| [`cuda-python-dsl-lab/`](cuda-python-dsl-lab/README.md) | Triton、Gluon、CuTe DSL 和 cuTile Python 的环境与算子实验。 |
| [`cuda-samples-analysis/`](cuda-samples-analysis/README.md) | 对第三方 `cuda-samples` 的本地源码解读与导航。 |
| [`cuda-op/`](cuda-op/) | 算子学习用的源码目录；其中的 `SGEMM_CUDA`、`flash-attention`、`flash-attention-minimal` 分别是嵌套的独立 Git 仓库。 |
| [`simulator-setup/`](simulator-setup/README.md) | Accel-Sim 与 GPGPU-Sim 共用的容器构建环境。 |
| [`accel-sim-lab/`](accel-sim-lab/README.md) | Accel-Sim 的部署、trace 重放和实验记录；`repo/accel-sim-framework/` 是上游源码 submodule。 |
| [`gpgpu-sim-lab/`](gpgpu-sim-lab/README.md) | GPGPU-Sim 的构建与最小运行实验；`repo/gpgpu-sim_distribution/` 是上游源码 submodule。 |
| [`.agents/`](.agents/) | 本地 Agent 技能配置。 |

## 所有自维护项目统一的执行约定

- 每个动作有自己的目录、短脚本和日志：优先 `commands/<action>/run.sh`，已有 `experiments/<action>/` 和 Python `run.py` 也可使用。
- 日志放在该动作目录的 `output.log`、配置名 `.log` 或固定 `log/`；重跑只更新对应动作日志，不集中写到项目顶层。
- 当前源码保留一份，编译产物共享 `build/` 或 `build/<variant>/`；先显式编译，再运行或采集，不隐式重建。Python DSL 的首次 JIT 编译按框架模型单独说明。
- 不使用万能 `lab.sh`，不自动生成哈希、快照、随机归档或 `history/`。已有历史记录保留原样，不当成新入口的验证结果。
- 分析器报告放在对应动作目录；第三方和嵌套仓库的上游构建接口不强制重写。

例如第一课从 `ncu/01_ncu_basics/` 执行：

```bash
bash commands/env/run.sh
CUDA_ARCH=sm_89 bash commands/compile/run.sh
bash commands/run/run.sh 16
bash commands/basic/run.sh 67108864
```

`sm_89` 仅适用于匹配的目标 GPU，其他设备应先核查。完整规则见 [`AGENTS.md`](AGENTS.md)。

## 第三方克隆

以下目录各自保留独立 Git 历史，以 submodule 纳入工作区仓库；本地未提交修改仍留在各自仓库中，不会进入工作区提交：

| 目录 | 用途 |
| --- | --- |
| [`third-party/CUDA-Python-Op-Lib/`](third-party/CUDA-Python-Op-Lib/README.md) | PyTorch 自定义 CUDA 算子开发模板。 |
| [`third-party/CUDApunk/`](third-party/CUDApunk/README.md) | CUDA、CUTLASS、CuTe 等练习与笔记。 |
| [`third-party/LeetCUDA/`](third-party/LeetCUDA/README.md) | CUDA 算子与性能优化练习。 |
| [`third-party/cuda-samples/`](third-party/cuda-samples/README.md) | NVIDIA CUDA 官方 C++ 与 Python 示例。 |
| [`third-party/cuda-training-series/`](third-party/cuda-training-series/README.md) | OLCF CUDA 培训课程材料。 |
| [`third-party/pmpp/`](third-party/pmpp/README.md) | 《Programming Massively Parallel Processors》配套示例。 |

`cuda-op/` 中的嵌套克隆及模拟器实验目录 `repo/<仓库名>/` 下的源码也作为 submodule 登记。新克隆工作区后执行 `git submodule update --init --recursive` 获取第三方源码；已存在的子仓库本地修改不会由工作区提交保存。`build/`、`downloads/`、`logs/`、`runs/`、`results/`、`traces/` 及分析器报告等路径由根目录 `.gitignore` 排除，新克隆不会获得其中的历史证据。查看差异或运行测试时，请进入目标目录并确认其 Git 根目录。工作区操作约定见 [`AGENTS.md`](AGENTS.md)。
