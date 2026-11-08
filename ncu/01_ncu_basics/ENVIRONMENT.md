# CUDA 与 Nsight Compute 环境搭建：WSL2 检查、权限设置与验证

本文件是[第一课实操](lab01.md)的环境准备入口，重点解决：**CUDA 程序能运行，但 NCU 报 `ERR_NVGPUCTRPERM`**。NCU（Nsight Compute）是 GPU 函数的性能分析器；GPU 函数称为 **kernel（内核）**，例如让许多线程分别完成一项数组乘法。**性能计数器**是硬件记录事件的统计机制，例如累计传输了多少字节。这个错误表示工具无权访问相关计数器，不表示乘法答案算错。

适用环境：Windows 宿主机 + WSL2 + NVIDIA GPU。**宿主机**指运行 WSL 的 Windows 系统；**WSL（Windows Subsystem for Linux）**是在 Windows 上运行 Linux 环境的机制，**WSL2** 使用 Linux 内核，但 GPU 仍由 Windows 侧驱动管理。原生 Linux 指直接运行 Linux 的机器，它与 WSL2 的驱动配置方式不同，不要混用。

> **当前状态：** 本次 stride 模板源码仅通过静态文本检查，相关脚本 `bash -n` 通过；未编译运行，旧日志不能验证当前模板源码与调用。
>
> **改前版本的历史验证（Vector Multiply）：** 当时默认入口已编译、运行通过： `CUDA_ARCH=sm_89 bash commands/compile.sh` 使用 nvcc 13.4、C++17、`-O3 -lineinfo`，见[编译日志](log/compile.log)；`bash commands/run.sh` 为 N=16、PASS/exit 0，见[运行日志](log/run.log)。当时环境记录为单张 RTX 4060 Laptop、cc 8.9；`nvidia-smi` 驱动查询为 596.21，SMI 页面用户态版本为 615.78.08，二者分开记录。
> 当时未测试全部参数；profile 仅通过 `bash -n`，未执行 NCU、NSYS、workers 或 SASS 检查，无性能结论，当前计数器权限尚未通过采集验证。下文环境表与 [旧 Vector Add 验证](RESULTS.md#本轮新入口验证) 仍是历史记录，不以新日志改写。

## 1. 先理解三个组成部分

| 组成 | 作用 | WSL2 下在哪里准备 |
|---|---|---|
| NVIDIA 驱动 | 管理 GPU，提供程序访问 GPU 的能力 | Windows 宿主机安装支持 WSL CUDA 的 NVIDIA 驱动 |
| CUDA Toolkit | 提供 `nvcc` 编译器、开发头文件和库 | 在 WSL 内安装适合发行版的 Toolkit |
| Nsight Compute（NCU） | 分析 GPU kernel 的硬件行为 | 在 WSL 内安装 Linux x86_64 版本；计数器权限由 Windows 侧设置 |

例如，在 WSL 终端编译程序靠 WSL 内的 `nvcc`，但开放这里的 GPU 计数器权限要到 Windows 设置。这两个位置不同，重装编译器并不能代替权限授权。

**能使用 GPU 计算，不代表有权读取 GPU 性能计数器。** 后者是额外的分析权限，因此计算程序可以通过，而 NCU 仍然失败。

如果从零准备环境，按“Windows 驱动 → WSL2 → WSL 内 Toolkit 与 NCU → 计数器权限 → 验证”的顺序进行。安装包和命令依 Windows、Linux 发行版及版本而异，按 [CUDA on WSL 官方指南](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)选择，不照搬其他发行版的包管理命令。

**不要在 WSL 内安装用于原生 Linux 的 NVIDIA 显示驱动来修复此问题。** 选择 Toolkit 安装方式时，避免使用会额外安装 Linux 显示驱动的元包。已有工具时先检查，不必为了本课程重装整个环境。

## 2. 确认是 WSL2，并检查工具

后文会切换两种命令环境：**PowerShell** 是 Windows 命令解释器，**Bash** 是本文使用的 Linux 命令解释器。代码围栏的语言名提示在哪一侧执行；不要把 Bash 的路径和语法直接粘贴进 PowerShell。

### 2.1 在 Windows PowerShell 中确认 WSL 版本

```powershell
wsl --list --verbose
```

观察当前发行版的 `VERSION` 是否为 `2`；这是 WSL 版本，不是 Ubuntu 等发行版的版本号。

### 2.2 在 WSL Bash 中检查内核

```bash
uname -r
```

历史检查的真实输出：

```text
6.18.40.1-microsoft-standard-WSL2
```

其中 `microsoft-standard-WSL2` 表明当时检查的是 WSL2，不是原生 Linux 主机。自己执行时以自己的输出为准；确认是 WSL2 后，权限配置才走 Windows 分支。

### 2.3 在 WSL Bash 中检查 GPU、编译器与分析器

```bash
command -v nvidia-smi
command -v nvcc
command -v ncu

nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
nvcc --version
ncu --version
```

- `command -v`：确认命令是否可找到，并显示实际路径。
- `nvidia-smi`：查询 GPU 和驱动状态。
- `compute_cap`：compute capability（计算能力），表示 GPU 的架构能力版本，不是性能分数；例如 8.9 对应编译目标 `sm_89`，不表示比 8.0 快固定比例。
- `--query-gpu`：指定要查询的字段；`--format=csv`：按表格字段输出。
- `nvcc --version`：查询实际编译器版本。
- `ncu --version`：查询实际 NCU 版本。

历史检查曾确认（本轮未复查）：

| 项目 | 实际值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| Compute capability | 8.9，对应本课 `-arch=sm_89` |
| Windows NVIDIA 驱动版本 | 596.21 |
| CUDA Toolkit / nvcc | 13.4 / V13.4.92 |
| Nsight Compute | 2026.3.1.0 |

这些是历史环境值，不是当前状态保证，也不是课程要求的唯一版本。若命令找不到，先检查对应工具是否安装、可执行文件目录是否在 `PATH` 中；不要立刻把所有问题归结为驱动权限。

注意：普通 `nvidia-smi` 顶部显示的 `CUDA Version` 表示驱动支持的 CUDA 版本信息，不能据此证明已经安装对应版本的 Toolkit；编译器版本要看 `nvcc --version`。

## 3. 为什么 `/etc/modprobe.d/` 的配置在这里不能解决问题

NVIDIA 文档中的这行配置：

```conf
options nvidia NVreg_RestrictProfilingToAdminUsers=0
```

含义是：在加载 **Linux 的 NVIDIA 内核模块**时，将“仅管理员可分析”的限制设为关闭。它属于原生 Linux 驱动的配置方式，可能还需要更新 initramfs（开机早期加载驱动等所用的初始文件系统）或重启才能生效；这里仅解释这行配置，不要求执行这些操作。

历史只读检查发现 `/etc/modprobe.d/nvidia.conf` 中存在这行配置，但当时 WSL 中的 NCU 仍然报权限错误。原因是 **WSL2 的 GPU 性能计数器权限需要在 Windows 宿主机设置**，不是再添加一份 Linux 模块参数即可生效。

因此，在当前环境下：

- 不需要重复创建同样的 `.conf` 文件。
- 不要为了此问题在 WSL 中执行 `modprobe -r` 卸载驱动。
- 不要把 WSL 中的 `sudo` 当成 Windows 驱动权限已开放的保证。
- 本文不删除已有配置，也不自动修改任何系统设置。

## 4. 在 Windows 上开放 GPU 性能计数器

这是管理员级别的系统设置，不是修改 CUDA 源码或各动作的 `run.sh`。

### 方法 A：NVIDIA 控制面板

在 **Windows 桌面**操作：

1. 使用具有管理员权限的账户，以管理员权限打开 NVIDIA 控制面板。
2. 在顶部 **桌面（Desktop）** 菜单中，勾选 **启用开发者设置（Enable Developer Settings）**。
3. 左侧进入 **开发者（Developer） → 管理 GPU 性能计数器（Manage GPU Performance Counters）**。
4. 选择 **允许所有用户访问 GPU 性能计数器（Allow access to the GPU performance counters to all users）**。
5. 点击 **应用（Apply）**，完成可能出现的管理员确认。

具体中文标签可能随界面版本略有不同，可按英文标签定位。

### 方法 B：NVIDIA App

如果使用 NVIDIA App，且版本提供相应选项，可进入：

```text
System → Advanced → Developer → Manage GPU Performance Counters
系统 → 高级 → 开发者 → 管理 GPU 性能计数器
```

选择允许所有用户访问并应用。NVIDIA 官方权限说明指出，该入口从 NVIDIA App 11.0.6 起提供。两种方式选择一种即可，不要求同时设置。

### 修改前要知道的影响

“允许所有用户”是系统级权限放开，并非只授权本课程程序。性能计数器能暴露底层 GPU 活动，默认限制与侧信道安全风险有关：其他用户可能从硬件活动统计中间接推断某个程序的行为，而不仅是看到自己程序的统计。

- 个人学习电脑：在理解影响后按需要开启。
- 共享、学校或公司设备：先征得管理员同意，不绕过管理策略。
- 不再需要分析时：可在同一页面恢复仅允许管理员访问。

## 5. 先验证计算，再验证 NCU

以下是在 **WSL Bash** 中复现实验的步骤。共享头见 [common.cuh](../kernels/common.cuh)；改前版本默认入口曾编译并普通运行通过，历史证据为[编译日志](log/compile.log)和[运行日志](log/run.log)，不验证当前 stride 模板源码；本次仅静态检查，未编译运行或采集 NCU。

### 5.1 编译并核对最小输入

```bash
cd /home/mikasa/hpc/cuda/ncu/01_ncu_basics
mkdir -p build
nvcc -std=c++17 -O3 -lineinfo -arch=sm_89 kernels/main.cu -o build/lesson01
./build/lesson01 16
```

第一行 `cd` 切换到本课目录，给出的是本工作区实际绝对路径；其他电脑应使用自己的 `ncu/01_ncu_basics/` 路径。`mkdir -p` 创建构建目录并允许它已存在；`./build/lesson01 16` 运行产物，参数 16 是数组长度 N。`sm_89` 只是示例，历史设备的 compute capability 为 8.9；未来按实际设备与 Toolkit 支持调整，`-arch` 指定这个编译目标。
`-std=c++17` 选择语言版本，`-O3` 启用优化，`-lineinfo` 保留源码行号映射，`-o` 指定输出文件。

程序把两个输入数组按项相乘：第 i 项输入是 `(i%1024)*0.125` 和 2，输出应是 `(i%1024)*0.25`。**线程**是执行其中一项的任务，256 个线程组成一块（block），多个块组成网格（grid）；N=16 时一个块中只有前 16 个线程读写数组。预期：打印 `PASS`，并得到 `c[0]=0` 到 `c[15]=3.75`，相邻输出相差 0.25。若这里失败，先处理编译或 CUDA 运行错误，不继续分析错误程序的性能。

### 5.2 用最小的基础采集验证权限

**Launch** 是一次 kernel 启动提交；本程序先启动 10 次预热，再启动 20 次计时，预热是计时前先运行以改变初始化、缓存和频率状态。**Metric** 是单个观测量，**Section** 是围绕一个问题的指标组，**Set** 是多个组的集合；`basic` 是基础集合名。硬件未必能一次同时收齐所有指标，工具可能 **Replay（重放）**目标工作，每轮采集称一个 **pass**。因此“只选一次启动”不等于“GPU 只执行一轮”。

```bash
ncu --set basic \
    --kernel-name-base function -k vector_mul \
    -s 10 -c 1 \
    --replay-mode kernel --cache-control all --clock-control base \
    -o log/profile.ncu-rep ./build/lesson01 1048576
```

- `--set basic`：采集基础指标集合。
- `-k vector_mul`：只匹配目标函数。
- `-s 10 -c 1`：跳过前 10 次匹配启动，只采集随后一次。
- `--replay-mode kernel`：需要多个采集 pass 时重放目标 kernel。
- `--cache-control all`：请求在重放 pass 前清理缓存。
- `--clock-control base`：请求基础时钟控制；它与计数器访问权限是不同事项，是否支持以实际输出为准。
- `-o`：设置报告输出路径，这里显式写出 `.ncu-rep` 后缀，不使用强制覆盖。已有同名报告时，先手工改名保留再运行。

这里 N=1048576、stride=1（相邻线程连续取输入），每项 float 占 4 字节，三组 GPU 数组合计 `3*N*4=12 MiB`，CPU 数组也约 12 MiB；`1 MiB=2^20` 字节。这是各侧分配容量，不是计数器测得的显存传输量。空间不足时减小 N，不用大分配代替最小正确性核对。

**预期成功标准，而非本次已取得的结果：**

1. 不再出现 `ERR_NVGPUCTRPERM`。
2. 有实际采集进度和有效指标输出。
3. 成功生成并可读取 `log/profile.ncu-rep`。

程序打印 `PASS` 只证明结果核对成功，不能单独证明 NCU 成功。查询命令退出码为 0 也不一定代表全部查询成功；本课曾出现打印权限错误但查询返回 0 的情况。

### 5.3 查看报告

仅在报告成功生成后执行：

```bash
ncu --import log/profile.ncu-rep --page details
```

`--import` 读取已有报告，`--page details` 显示详细结果，不会重新运行 kernel。
如已有同名报告，先手工改名保留再重新采集，不覆盖旧报告。检查本次采集输出，避免把之前留下的报告误认为本次成功；已有的 `build/vector_add_basic.ncu-repz` 保留原地，不移动、不删除、不以重采结果替代。

理解上述命令后，可使用各动作的独立脚本执行等价步骤（下列相对路径以本课目录为起点）：

```bash
CUDA_ARCH=sm_89 bash commands/compile.sh
bash commands/run.sh 16
bash commands/profile.sh 1048576
```

脚本按自身位置定位本课，可从任意目录通过实际脚本路径调用。只有 `commands/compile.sh` 要求显式 `CUDA_ARCH`，固定 C++17、`-O3 -lineinfo`，将 `kernels/main.cu` 编译为共享的 `build/lesson01`；编译前移除旧目标，失败即停止。run/profile 缺产物提示先编译，不隐式构建。普通运行的可选参数 N 默认 `16`，profile 默认 `1048576`。环境查询保留上文的手工命令，不再提供 env 动作脚本。

三个动作分别把展开的命令、输出与最终 `exit_code` 写入本课 `log/compile.log`、`log/run.log`、`log/profile.log` 并显示在终端；重跑覆盖各自新日志，保留独立实验需先另存日志与报告。profile 导出 `log/profile.ncu-rep`，已有报告则拒绝覆盖；NCU 失败即停止，报告内容由实验者检查。旧 `commands` 非脚本证据已原样移到 `log/legacy_commands/<原相对路径>`，原始日志命令不改；`runs/`、旧 build 产物保持原地，不新增自动归档。脚本不设置系统权限。旧版编译与普通运行曾通过；当前 stride 模板源码仅静态检查，未编译运行或采集；验证范围见 [lab01](lab01.md#当前验证状态与旧证据边界)，旧 compile-only 仍只代表历史源码。

## 6. 应用设置后仍然报错怎么办

先重新打开 Windows 的设置页面，确认更改已保存，然后重新启动一次 NCU 命令。

若仍为同一个权限错误，可尝试重启 WSL。**先保存编辑内容并停止 WSL 内的任务**，然后在 **Windows PowerShell** 中执行：

```powershell
wsl --shutdown
```

这会关闭所有运行中的 WSL 发行版及其进程，可能中断 Docker 工作负载和当前远程开发连接；不是只关闭当前终端。它也不是重装或删除发行版。

重新进入 WSL 后，再执行第 5 节的采集命令。如果仍失败：

1. 确认改的是实际运行 WSL 的 Windows 宿主机，而非另一台电脑。
2. 确认管理员授权与 GPU 性能计数器设置没有被系统策略恢复。
3. 重新检查驱动、Windows/WSL 与 NCU 版本是否满足官方支持范围。
4. 保留完整错误。如果已经变为时钟控制、指标不支持等其他错误，应按新的错误定位，不能继续当成计数器权限问题。

不要通过重装 Linux GPU 驱动、关闭安全机制或反复运行 `--set full` 盲目尝试。

## 7. 环境验收清单

已勾选项仅代表改前版本的历史验证，不代表当前 stride 模板源码已编译运行；本次仅静态文本与 `bash -n` 检查通过。

- [ ] 确认发行版运行于 WSL2。
- [x] 旧版环境记录识别到单张 RTX 4060 Laptop、cc 8.9。
- [x] 旧版使用 nvcc 13.4，显式选择 `CUDA_ARCH=sm_89`。
- [x] 旧版 Vector Multiply 编译成功，见[编译日志](log/compile.log)。
- [x] 旧版默认 N=16 全量核对 `PASS`、exit 0，见[运行日志](log/run.log)。
- [ ] 当前 NCU 工具与计数器权限完成实际采集验证。
- [ ] Windows 侧允许当前学习场景所需的性能计数器访问。
- [ ] NCU 实际采集成功，且报告可导入查看。

完成后回到[第一课实操](lab01.md)，继续区分启动、指标、指标组与集合。那里还将解释 CUDA Event（插在 GPU 工作队列中的时间标记）与 NCU 报告时长为何不一定相同；环境验收本身不要求先掌握计时分析。

## 官方参考

- [ERR_NVGPUCTRPERM：权限说明与 Windows 设置入口](https://developer.nvidia.com/ERR_NVGPUCTRPERM)
- [Nsight Compute：Linux 与 WSL 系统要求](https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html#linux-and-wsl)
- [CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
