# 当前环境配置与验证记录

初始部署记录日期：**2026-11-04，Asia/Hong_Kong**；最新工具检查：**2026-11-06**。部署位置：`/home/mikasa/hpc/cuda/cuda-python-dsl-lab`。本记录只描述这台机器，不代表 H20 或其他服务器的环境；除下面的更新节外，原部署表格和校验记录保留当时的状态。

当前入口已迁到 `commands/<action>/run.py`，日志位于各动作 `log/`。本页安装版本与 PASS 均为历史记录；本次迁移检查发现当前 `.venvs/` 为空，未安装依赖或重新验证这些环境。

## 2026-11-06 更新：ncu 已可从 PATH 调用

用户重新安装后，在当前 WSL 环境运行现有 [learning_tools.py](../scripts/learning_tools.py) 的工具版本检查，结果如下：

| 项目 | 本次实际结果 |
| --- | --- |
| PATH 中的命令 | `/usr/sbin/ncu` |
| 解析后的路径 | `/usr/lib/nsight-compute/ncu` |
| Nsight Compute CLI | **2026.3.1.0**，build **38829034** |
| `/usr/sbin/ncu --version` | 退出码 **0**，stderr 为空 |
| 检查时间 | 2026-11-06 14:06 UTC（Asia/Hong_Kong 22:06） |

原始证据：[manifest](../runs/20261106T140618Z-ncu-check-rx5bgfry/manifest.json)、[工具结果 JSON](../runs/20261106T140618Z-ncu-check-rx5bgfry/tools.json)、[原始日志](../runs/20261106T140618Z-ncu-check-rx5bgfry/tools.log)。该独立目录保留检查脚本快照、SHA256、命令和退出码，没有覆盖此前 `unavailable` 的记录。

现在可以直接执行：

```bash
command -v ncu
ncu --version
```

先前的 `unavailable` 是当时检查未在 PATH/指定回退位置找到命令，不代表从未安装或使用过。此前还在 Windows 安装目录的 `target/linux-desktop-glibc_2_11_3-x64/ncu` 找到过可在 WSL 执行的 Linux 版 2025.3.1.0；重新安装后本项目优先使用当前 PATH 中的 2026.3.1.0，无需配置那个旧路径。历史文字曾引用 `third-party/CUDA-punk/basics/memory-access/ncu.log`，本次迁移检查时该路径不存在；不重建第三方记录，也不将它作为本学习项目的 Profile 验收。

**本次仅检查工具版本，没有运行 GPU kernel、benchmark 或新的 Profile，没有修改驱动、计数器权限、shell 配置或 Python 环境。** 当前 GPU 计数器访问权限、新版 ncu 的实际采集兼容性和本项目完整 kernel Profile 仍未验证；`--version` 成功不等于采集成功，E0.4 保持部分完成。

## 宿主环境

| 项目 | 本次检测值 |
| --- | --- |
| Linux 发行版 | Arch Linux rolling，x86_64 |
| 内核 | `6.18.40.1-microsoft-standard-WSL2` |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| Compute Capability | 8.9，即 SM89 / Ada |
| 显存 | 8188 MiB |
| NVIDIA 驱动 / KMD | 596.21 |
| NVIDIA-SMI 工具版本 | 615.71.09；不要将其当作驱动版本 |
| nvidia-smi CUDA UMD | 13.2 |
| 系统 CUDA Toolkit | 13.4，`V13.4.92` |
| 编译器路径 | `/opt/cuda/bin/nvcc`、`ptxas`、`tileiras` |
| 系统 Python | 3.14.7，`/usr/sbin/python3` |
| 项目 Python | CPython 3.12.15，Astral python-build-standalone |
| uv | 0.12.23，`/usr/sbin/uv` |
| 宿主 C 编译器 | GCC 16.2.1 20260810 |

宿主采集原始记录：[inspect 日志](../runs/20261104T135228Z-inspect-pk73en6p/00-host.log)。GCC 另由本次 `cc --version` 检查确认；Triton 首次启动需要编译宿主 launcher，因此 C 编译器也属于复现条件。

## Python 环境

三个环境都是 CPython 3.12.15，解释器链接到本目录 `.python/cpython-3.12.15-linux-x86_64-gnu/bin/python3.12`。没有使用系统 site-packages。

| 环境 | 核心包 | 张量框架 | 其他直接依赖 |
| --- | --- | --- | --- |
| `.venvs/triton` | Triton 3.9.0；包含 Gluon | PyTorch 2.15.0+cu132 | NumPy 2.5.4、pytest 9.2.0 |
| `.venvs/cute` | nvidia-cutlass-dsl 4.8.1，`[cu13]` | PyTorch 2.15.0+cu132 | NumPy 2.5.4、pytest 9.2.0 |
| `.venvs/cutile` | cuda-tile 1.7.0 | cupy-cuda13x 14.2.0 | NumPy 2.5.4、pytest 9.2.0 |

完整传递依赖分别见 [triton.lock](../requirements/triton.lock)、[cute.lock](../requirements/cute.lock)、[cutile.lock](../requirements/cutile.lock)。锁文件包含版本与下载制品 SHA256。venv 的引导工具 pip 本次为 26.3，由 `uv venv --seed` 安装，不在 DSL 依赖锁文件内；uv 本身也使用宿主现有安装。

CuTe 的依赖解析包含 `nvidia-cutlass-dsl-libs-{base,core,cu12,cu13}==4.8.1`。包列表中同时出现 cu12/cu13 是本版本依赖解析结果；实际导入的编译器报告 **CUDA 13.4**，已用 GPU 示例验证。不要仅凭包名判断当前选择的后端。

## CUDA 各层版本

| 层次 | 实际观察 |
| --- | --- |
| 宿主驱动 CUDA API / UMD | 13.2 |
| 系统 nvcc / ptxas / tileiras | 13.4.92 |
| Triton wheel 附带 `backends/nvidia/bin/ptxas` | 12.9.86，通过该文件 `--version` 检查 |
| CuTe 实际加载的编译器 | `DSLCudaVersion(major=13, minor=4)` |
| PyTorch 编译/runtime 标识 | `torch.version.cuda = 13.2`，wheel 标识 `+cu132` |
| CuPy 查询的 driver API / runtime | 两者均返回 `13020`，即 13.2 |

这些版本没有被人为强制统一。系统 Toolkit 13.4 不意味着每个 Python 包都调用系统 `ptxas`，也不意味着每个进程的 runtime 都是 13.4。当前结果证明这些基础内核在本机可运行；没有单独执行系统 NVCC 生成 PTX 后交给驱动 JIT 的兼容性测试，因此不能推广为“CUDA 13.4 的任意 PTX/API 均受支持”。

## 实际校验

初次 GPU 验证中，Triton、Gluon、CuTe DSL、cuTile 的 Vector Add 均通过。随后将所有可配置 JIT 缓存位置统一到项目 `.cache`，再次验证最终入口；最终记录见下表。

| 检查 | 验证内容 | 记录 |
| --- | --- | --- |
| 安装及初次依赖检查 | 三套 venv、锁定安装、`uv pip check`、freeze、import | [setup manifest](../runs/20261104T135137Z-setup-021rq2tf/manifest.json) |
| 最终依赖及导入检查 | 三套环境分别检查，无 GPU 运行结论 | [check manifest](../runs/20261104T135725Z-check-m2hfnpms/manifest.json) |
| 最终 GPU 校验 | GPU 上下文和四种 DSL 的实际内核校验 | [run manifest](../runs/20261104T135724Z-run-hb2rwl3u/manifest.json) |

Vector Add 校验参数：FP32，block/tile 大小 256，`N=1,255,256,4099,65536`，每组重复两次，每次先将输出置为 NaN，执行后同步并与 CPU 参考比较；`rtol=1e-6, atol=1e-6`。总计四种 DSL、20 组规模配置、40 次正确性执行。随机种子均为 `20261104`；PyTorch 与 NumPy 的随机序列不同，因此这些是各自正确性验证，不是严格同输入性能对比。

原始日志中保留以下非致命警告：

- Triton 首次构建 launcher 时，Python 头文件与系统头文件的 `_POSIX_C_SOURCE` 定义不同；构建继续并完成 GPU 校验。
- Triton 3.9 提示 FP fusion 默认策略将在 3.10 变化。本次未覆盖默认策略；当前算子只有加法，该警告不构成 GEMM/FMA 数值行为的验证。

初始开发验证目录 [首次 run](../runs/20261104T135519Z-run-3_zskrd0/manifest.json) 同样保留。该轮执行期间补充过环境采集代码，复现最终入口应采用上表的最终记录及源码快照。

## 配置边界与未验证项

- 只新建本目录中的 Python、venv、依赖和脚本；未升级系统驱动、CUDA Toolkit 或系统 Python，未修改 shell 启动文件。
- `.cache` 保存 uv/JIT 缓存，`.cache/tmp` 用作临时目录。最终入口设置 `TRITON_CACHE_DIR`、`CUTE_DSL_CACHE_DIR`、`CUDA_TILE_CACHE_DIR`、`CUPY_CACHE_DIR`、`CUDA_CACHE_PATH` 和 `XDG_CACHE_HOME`。
- 安装结束后项目实际磁盘占用约 6.1 GiB；uv 缓存与 venv 共享硬链接，不能把分别 `du` 得到的逻辑占用简单相加。
- 未测量：性能、带宽、缓存命中率、DRAM 流量、寄存器占用；本次只验收环境和正确性。
- 未验证：Transpose、Reduction、Softmax、GEMM、sanitizer、Nsight、全部官方教程、H20 / SM90、Hopper/Blackwell 专属功能。
- 本工作区根目录不是 Git 仓库；本次未初始化仓库或提交。`.gitignore` 已排除 venv、缓存和运行产物，便于以后管理源码；原始运行证据仍保留在本地 `runs/`。

创建步骤、升级方式和官方链接见 [SETUP.md](SETUP.md)。
