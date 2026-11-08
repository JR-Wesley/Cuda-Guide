# CUDA Python DSL 学习工作区

本目录用于在同一块 GPU 上学习 Triton、Gluon、CuTe DSL 和 cuTile Python，并保留可复现的环境及实验记录。历史部署日期为 **2026-11-04（Asia/Hong_Kong）**。

四种 DSL 使用三套 Python 3.12 虚拟环境：Gluon 属于 Triton，不单独安装。环境实际版本、GPU 校验结果和限制见 [当前环境记录](docs/ENVIRONMENT.md)；完整创建流程见 [SETUP.md](docs/SETUP.md)。

## 快速开始

在本目录执行，每次选择一个动作：

```bash
cd /home/mikasa/hpc/cuda/cuda-python-dsl-lab
python3 commands/inspect/run.py
python3 commands/check/run.py all
python3 commands/run/run.py all
```

`inspect` 记录宿主环境，`check` 检查依赖及导入，`run` 实际 JIT 编译、启动 GPU 内核并校验结果。只有 `run` 成功才能说明本目录的基础示例可运行。命令失败返回非零，并保留原始日志。

需要重建或补齐环境时：

```bash
python3 commands/setup/run.py all
```

`setup` 优先使用已有带 SHA256 的锁文件，不自动升级。也可以把 `all` 换成 `triton`、`cute` 或 `cutile`；`run triton` 同时运行 Triton 和 Gluon。

## 目录

```text
cuda-python-dsl-lab/
├── README.md
├── commands/<action>/
│   ├── run.py               # 独立动作实现，不调用统一 dispatcher
│   └── log/                 # 固定日志；重跑同动作覆盖
├── requirements/
│   ├── triton.in / .lock     # Triton + Gluon + PyTorch
│   ├── cute.in / .lock       # CuTe DSL CUDA 13 + PyTorch
│   └── cutile.in / .lock     # cuTile + CuPy CUDA 13
├── scripts/                  # 简单记录辅助、环境/导入检查与结果汇总
├── examples/                 # 四种 DSL 的 Vector Add
├── docs/                     # 配置流程、当前环境、学习计划
├── .python/                  # 项目私有 CPython 3.12.15
├── .venvs/{triton,cute,cutile}/
├── .cache/                   # uv 下载缓存、安装临时文件
├── build/                    # 框架管理的共享 JIT 产物
└── runs/                     # 只保留旧证据；当前动作不写入
```

## 手动学习

```bash
source .venvs/triton/bin/activate
python examples/triton_vector_add.py
python examples/gluon_vector_add.py
deactivate

source .venvs/cute/bin/activate
python examples/cute_vector_add.py
deactivate

source .venvs/cutile/bin/activate
python examples/cutile_vector_add.py
deactivate
```

直接运行适合交互调试；需要本次日志时使用对应 `commands/<action>/run.py`。各动作的 `log/` 保存原始命令、退出码和输出，重跑只覆盖本动作目录；旧 `runs/` 不重写，也不创建时间/随机目录或 history。源码仍在 `examples/`、`experiments/`、`benchmarks/`，不按动作复制。

**JIT 编译边界（允许的框架例外）**：现有 DSL 示例在首次调用时按输入编译。`run`/`learn` 不安装环境，也不调用其他构建动作；框架可能在首次 prepare/launch 触发 JIT，共享缓存放 `build/`，不另建缓存判定框架。这符合当前入口约定，不是迁移未完成项，也不要求改为 AOT（提前编译）。`learn` 将 prepare、首次 launch+同步耗时记录在 `startup`；可选 `--bench` 在正确性检查与预热之后测量重复启动，不把首次 JIT 开销当作稳态 kernel 时间。仅激活 venv 不会设置这些缓存变量。

Vector Add 覆盖 `N=1,255,256,4099,65536`，使用 FP32，每组运行两次，以 CPU 结果校验。包含小于一个 block、整 block、非整除尾块和较大输入。它们是环境冒烟测试，未做性能计时，不用于推断带宽或 DSL 性能排名。

## 按学习计划逐项实验（功能优先）

现已按 [学习计划](docs/LEARNING.md) 建立独立源码、共享校验、测试脚本和证据索引：**[逐项入口与实际完成状态](docs/experiments/README.md)**。

环境已安装时直接复用，无需重装；本次检查 `.venvs/` 为空，运行 DSL 前需要用户另行执行 setup。以下每次只选择一个命令：

```bash
python3 commands/learn/run.py cpu
python3 commands/learn/run.py triton
python3 commands/learn/run.py gluon
python3 commands/learn/run.py cute
python3 commands/learn/run.py cutile
# 只复查某一计划项
python3 commands/learn/run.py triton --ids T1.4
python3 commands/learn/run.py gluon --ids G3.7
```

`bash tests/test_small.sh`（等价于 `python3 commands/learn/run.py all`）串行执行小规模功能集。默认无事件 benchmark、无可选参数扫描；`--bench` 和 `--compare-configs` 必须显式指定，本轮未执行。

- `experiments/`：各算子及四个 DSL suite；Gluon/CuTe 索引推导另有独立脚本。
- `benchmarks/`：同一 CPU 输入与 SHA256、FP64 参考、误差、两次 NaN 投毒/校验和可选事件计时。
- `tests/`：CPU 回归测试与小功能测试入口。
- `docs/experiments/`：按原计划编号的说明、状态与证据。
- `commands/<action>/log/`：本次命令、退出码、输入/参考/输出、原始日志和可用编译产物；同动作覆盖，无源码快照/制品哈希。
- `runs/`：仅保留历史原始证据，不作为当前入口已经验证的证明。
- `python3 commands/static/run.py commands/learn/log`：离线读取已有结果与 PTX/cubin；产物缺失则报错，不启动 GPU 或编译。输出固定在 `commands/static/log/`。
- `python3 commands/compare/run.py commands/learn/log`：读取已有的多 DSL 结果，对照输入标识与功能状态；不重新运行 kernel。输出固定在 `commands/compare/log/`。

**历史验证**：2026-11-05 四个 suite **112/112 小功能 cases 通过**；另有布局索引/所有权校验和离线 PTX/SASS 检查。完整 Profile、性能排名、CuTe 多 Stage、CUDA C++/库性能基线和 H20 专题仍未完成，不能将功能 PASS 当作十二周全部验收。本次入口迁移不执行 GPU 或安装环境；历史 PASS 不等于新入口复验。

锁文件下载制品校验（`uv --require-hashes`）保证依赖完整性，保留。`benchmarks/core.py` 的哈希也保留：输入族和 shape 的摘要确定 PCG64 随机种子；数组 dtype、shape、内容的摘要作为 `input_id`，供跨 DSL 检查输入是否一致。结果正确性由 `validate()` 检查形状、类型、连续性、有限值及数值容差，不靠哈希代替。cuTile 和 Gluon 布局输出不再生成源码/产物指纹或源码副本；这些与输入身份、依赖完整性校验是不同用途。
