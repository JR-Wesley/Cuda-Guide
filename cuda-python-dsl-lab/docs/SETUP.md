# 环境创建与复现

## 1. 本次方案

使用系统已有 `uv` 创建项目私有的 CPython **3.12.15**，再建立三套普通 venv。选择 3.12 是为了统一教学环境；不替换系统 Python 3.14，不修改系统 CUDA、驱动或 shell 启动文件。

| 环境 | 直接依赖 | 用途 |
| --- | --- | --- |
| `.venvs/triton` | `triton torch numpy pytest` | Triton 和 `triton.experimental.gluon` |
| `.venvs/cute` | `nvidia-cutlass-dsl[cu13] torch numpy pytest` | CuTe DSL，CUDA 13 安装路径 |
| `.venvs/cutile` | `cuda-tile cupy-cuda13x numpy pytest` | cuTile，使用系统 TileIR 编译器 |

Triton/CuTe 使用 PyTorch 分配张量，cuTile 使用 CuPy。PyTorch 的 CUDA 运行库来自该 venv 的依赖，不等于 `/opt/cuda` 的版本。PyTorch 也会传递依赖 Triton，所以 CuTe 环境中出现 Triton 是正常现象；这不取消三套环境的隔离。

本机已经有 `/opt/cuda/bin/tileiras`、`ptxas` 和 CUDA 13.4，因此 cuTile 未使用 `[tileiras]` extra。迁移到没有完整 Toolkit 的机器时，可以在新环境使用官方 `cuda-tile[tileiras]` 方案，并检查 TileIR、NVCC、NVVM 的主次版本一致；不能只修改依赖后继续宣称沿用本次校验结果。

## 2. 安装前检查

```bash
cd /home/mikasa/hpc/cuda/cuda-python-dsl-lab
python3 commands/inspect/run.py
```

记录 OS/内核、GPU、驱动、计算能力、显存、`nvcc`、`ptxas`、`tileiras`、系统 Python、uv 和可用磁盘空间。`nvidia-smi` 显示的驱动 CUDA 能力、系统 Toolkit、Python wheel 的 CUDA runtime 是三层信息，需要分别记录。

本次实际是 WSL2 中的 Arch Linux，GPU 由宿主机提供。沙箱内的 GPU 访问和下载曾被拦截，转到允许访问 GPU/网络的执行环境后成功读取设备；不能把沙箱错误解释为 GPU 不存在，也不能靠安装 Python 包修复宿主 GPU 权限。

要求：Linux x86_64、可访问 GPU、`python3`、`bash`、`uv`、网络可访问 PyPI 及 Astral/GitHub Python 下载站点。系统 Toolkit 路径默认 `/opt/cuda`；其他机器可通过 `CUDA_HOME` 和 `CUDA_PATH` 指定。锁文件针对本次 Linux x86_64 / CPython 3.12 解析，不承诺跨 OS、跨架构直接适用。

## 3. 实际安装入口

```bash
python3 commands/setup/run.py all
```

内部按如下步骤执行，准确命令和退出码以 `commands/setup/log/manifest.json` 为准：

1. 设置项目内 `UV_PYTHON_INSTALL_DIR=.python`、`UV_CACHE_DIR=.cache/uv` 和临时目录 `.cache/tmp`。
2. `uv python install 3.12.15 --no-bin` 下载独立 Python，不改系统 Python，不写用户级可执行文件目录。
3. 为每个目标执行 `uv venv --seed --python 3.12.15 .venvs/<name>`。已经存在的环境保留。
4. 若锁文件不存在，从 `requirements/<name>.in` 用 `uv pip compile --generate-hashes` 解析依赖。已有锁文件直接复用。
5. `uv pip sync --require-hashes` 根据锁文件安装精确版本，只操作目标 venv。该动作会移除目标 venv 中不属于锁文件的额外包，学习依赖应写入 `.in` 并显式更新锁文件。
6. 安装完成后，另执行 `python3 commands/check/run.py all`，它才运行 `uv pip check`、`uv pip freeze` 和导入检查；导入成功仅代表安装层验证。setup 不隐式调用 check 或 run。

下载源明确为 `https://pypi.org/simple`；未使用 nightly、预览索引或源码主分支。`uv` 共享下载缓存，并尽可能通过硬链接减少相同包的重复占用。PyTorch 及 NVIDIA 运行库较大，首次安装需要下载数 GB；临时文件放在工作区，避免占满小容量 `/tmp`。

## 4. 分阶段验证

```bash
# 无需内核计算，检查包依赖和导入
python3 commands/check/run.py all

# GPU 上下文、实际 JIT、内核启动、同步、CPU 参考校验
python3 commands/run/run.py all

# 只验证一个环境
python3 commands/run/run.py cute
```

检查中不以 `SKIPPED` 作为通过；一个环境失败后仍尝试其余环境，最终整体返回非零。GPU 初始化失败时该环境的内核不会被运行，不把未运行误记为通过。

日志固定在 `commands/<action>/log/`：

- `manifest.json`：工作目录、各条命令、标签及退出码。
- `<label>.log`：原始合并 stdout/stderr，包含失败输出。
- `status.txt`：本次动作整体结果。

重跑同动作覆盖其 log，不影响其他动作及历史 `runs/`；不创建快照、源码/制品哈希、时间目录或归档，不复制锁文件。`requirements/*.lock` 的包完整性校验仍保留，不自动升级。当前入口迁移未执行安装、GPU 或 profiler。

JIT 产物使用项目共享 `build/`，由各框架自身管理。run/learn 首次调用可能触发框架编译，这是允许的 JIT 例外，不是迁移缺项，不要求 AOT。首次调用、预热和稳态计时的区分见根 README 的 JIT 编译边界。

## 5. 手动重建一套环境

推荐用上述入口。理解其底层流程时，可按以下命令在本目录操作：

```bash
export UV_PYTHON_INSTALL_DIR="$PWD/.python"
export UV_PYTHON_BIN_DIR="$PWD/.python/bin"
export UV_CACHE_DIR="$PWD/.cache/uv"
export TMPDIR="$PWD/.cache/tmp"
mkdir -p "$TMPDIR"
uv python install 3.12.15 --no-bin
uv venv --seed --python 3.12.15 .venvs/triton
uv pip sync --python .venvs/triton/bin/python \
  --index-url https://pypi.org/simple --require-hashes requirements/triton.lock
uv pip check --python .venvs/triton/bin/python
```

不要将 venv 目录直接复制到另一台机器：其解释器链接及启动器可能含绝对路径。迁移时复制源码和锁文件，重新运行 `setup`，然后 `inspect`、`run`。

## 6. 有意升级依赖

`.in` 表达需要的包，`.lock` 记录本次完整依赖及哈希。仅修改 `.in` 不会隐式覆盖已有 `.lock`。升级应作为一次新的配置变更，先备份原锁文件和已有实验记录，再解析：

```bash
UV_CACHE_DIR="$PWD/.cache/uv" uv pip compile \
  --python .venvs/triton/bin/python --index-url https://pypi.org/simple \
  --upgrade --generate-hashes requirements/triton.in \
  -o requirements/triton.next.lock
```

审查 `triton.next.lock` 与现有版本差异后再替换正式锁文件，执行 `setup triton` 和 `run triton`，并新增环境记录。升级 Triton 时同时考虑 PyTorch 对 Triton 的版本约束；不要直接对已验证环境执行无记录的 `pip install -U`。

## 7. 常见故障

| 现象 | 检查和处理 |
| --- | --- |
| `GPU access blocked by the operating system` | 检查执行沙箱、WSL GPU 透传及权限，在有 GPU 权限的终端重试 `inspect` |
| 下载报 `Operation not permitted` / proxy tunnel 错误 | 检查执行环境的联网限制和代理，保留错误，不修改系统包来绕过 |
| 能 import，但内核失败 | 查看对应 `run` 的完整 traceback；检查驱动、PTX、所选编译器及设备架构 |
| `unsupported PTX version` | 新工具链生成的 PTX 可能超出驱动 JIT 支持；原生 cubin 成功不代表 PTX JIT 成功 |
| cuTile 找不到编译器 | 检查 `tileiras` 和其依赖工具是否存在、版本是否匹配；必要时在新环境采用 `[tileiras]` 路线 |
| 官方示例失败但本目录通过 | 核对官方示例 commit、安装包版本、GPU 架构与可选依赖；基础 Vector Add 不能覆盖所有 API |
| 换成 H20 后失败 | 重新采集 SM90 环境并验证；尤其核对所用 TileIR 编译器是否支持 Hopper |

## 8. 官方依据

以下页面于 2026-11-04 查询，后续可能变化；本地锁文件和原始日志代表本次实际安装。

- [Triton 安装](https://triton-lang.org/main/getting-started/installation.html)：稳定 wheel 安装与 Python 支持范围。
- [Triton / Gluon 源码与教程](https://github.com/triton-lang/triton/tree/main/python/tutorials/gluon)：Gluon 随 Triton 提供。
- [CuTe DSL Quick Start](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html)：CUDA 12/13 安装选择，以及示例版本匹配要求。
- [cuTile Quickstart](https://docs.nvidia.com/cuda/cutile-python/quickstart.html)：驱动、Python、GPU 要求及系统/venv 编译器安装方式。
- [cuTile 官方仓库](https://github.com/NVIDIA/cutile-python)：版本相关的架构支持说明。仓库旧说明和最新文档可能不同，迁移时按实际 compiler 版本验证。
