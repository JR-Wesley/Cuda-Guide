# 实验 10｜报告里的 vector_mul 究竟是哪一次调用？

## 为什么计算相同，还要学选择范围

先约定本实验的角色：CUDA 是编写和运行 GPU 程序的平台，CPU 部分负责准备和提交，**kernel** 是在 GPU 上由许多线程执行的函数；一次把函数及参数提交给 GPU 叫一次 **launch**。**预热**是先执行若干次、让首次准备和缓存状态趋于稳定，但不把这些调用当作正式样本。例如先做 10 次乘法再分析后 20 次，函数名仍然完全相同。

设一个程序启动同名 kernel 30 次，前 10 次只用于预热。你按函数名采到一个报告，却可能正好选中了第一次预热。指标再详细，分析对象错了也没有意义。本课固定逐元素乘法，只改变“采谁、采哪段、采几次”，不以算法加速为目标。

配合 [lec10](lec10.md) 第 1–3 节学习进程、启动与范围，第 7 节学习子进程。**进程**是一次程序运行的独立实例，PID 是其编号；**launch** 是 CPU 提交一次 GPU 函数；**NVTX** 是给分析器看的业务名字标记；`cudaProfilerStart/Stop` 是采集启停 API（程序调用接口）。标记和启停都不是 GPU 算法，也不替代同步或计时。

先把被观察的计算讲清楚：这里的“向量”只是一列按下标排列的数，向量乘指对应位置相乘，不是把所有乘积加起来的点积。kernel 接收两个 float 输入数组 a、b，一个同长度输出数组 c，以及长度 n 和输入步长 stride；本课 stride 固定为 1，因此结果是 `c[i]=a[i]*b[i]`。例如 a 的前四项为 `[0,1,2,3]`，b 为 `[1,2,3,4]`，输出就应是 `[0,2,6,12]`。线程 2 只读取 a[2]、b[2] 并写 c[2]，不需要等待线程 1，也不与其他线程交换中间结果。

GPU 线程以 **block（线程块）**分组，本次启动全部块组成 **grid**；例如 256 线程一块，n=257 就要两块才能覆盖。**逻辑 GPU0** 是当前程序看到的第 0 个设备，不一定是机器插槽编号 0；float 是这里每项 4 字节的单精度浮点数，stride 是相邻有效输入在数组下标上的步长。

host（在 CPU 上执行的部分）生成这两份输入，在逻辑 GPU0 分配三个数组并上传 a、b，然后用向上取整的 grid 启动 GPU 线程。每次 launch 都从相同输入重算并覆盖 c，而不是在上次结果上继续累积；完成全部启动后，host 下载最后一次输出，逐项重新相乘核对。因此最后的 PASS 证明下载结果正确，却不能单凭它证明每次启动都被采集或分别验证过。

选择这个小程序，正因为它没有复杂算法变化来干扰“分析对象是否选对”这个问题。30 次启动可以拥有完全相同的函数名、输入和 grid/block，却分别属于预热或正式区间；输出也无法告诉我们报告来自哪一次。接下来要补上的不是乘法优化，而是启动身份的证据：先认识计算，再沿 CPU 提交顺序区分区间，最后一次只改变一个筛选条件。

本课阅读入口是 [main.cu](kernels/main.cu)，计算只复用 [vector_multiply.cuh](../kernels/vector_multiply.cuh) 的 `vector_mul`，错误检查复用 [common.cuh](../kernels/common.cuh)。不要误读成调用 `run_vector`：本课自己组织预热和范围，输入规则也不同。以下命令均是后续操作说明，在 Linux/Bash 的 `cuda/ncu/10_advanced_profiling/` 执行。

## 1. 用 16 个元素认识“有效线程”

**目的与代码：** 读 main 顶部常量和初始化，保留默认 n=16、block=256、stride=1；纸上先填前四项，不执行命令。

| i | a[i]=i%17 | b[i]=i%7+1 | 预期 c[i] |
| --- | ---: | ---: | ---: |
| 0 | 0 | 1 | 0 |
| 1 | 1 | 2 | 2 |
| 2 | 2 | 3 | 6 |
| 3 | 3 | 4 | 12 |

`%` 是取余。以下除法是非负整数除法，会舍去小数；所以 `ceil(n/block)`（向上取整）可写成 grid=`(n+block-1)/block`，例如 `(257+255)/256=2`。当前 n=16 时只有 1 个 block；256 个线程不等于 256 个输出。共享 kernel 只让 `i<n` 的线程读写，输入地址是 `i*stride`，输出地址是 i。

**预期与解释：** 全部 16 项都被 CPU 精确比较，因为这里的小整数及其乘积能被 float 精确表示。程序不逐项打印数组，不能把此表当成终端实测输出。

**异常检查与练习：** 本课 host 数组只分配 n 项，不要把 stride 改成 2：最后几次读取可能越界。预测 n=257 时 grid 为多少、最后一个 block 有几个有效线程。

## 2. 沿 CPU 提交顺序画范围

**目的与代码：** 继续读 main 的两个循环与同步点，先不运行。时间顺序是：

```text
分配、上传
  → 10 次预热 launch → 同步等待 GPU
  → cudaProfilerStart
    → nvtxRangePushA("measure")
      → 20 次目标 launch → 同步等待 GPU
    → nvtxRangePop
  → cudaProfilerStop
  → 下载、CPU 核对、释放
```

图中 `cudaProfilerStart/Stop` 只通知分析器开启/关闭采集，不会替你等待计算；`nvtxRangePushA` 开始命名范围，`nvtxRangePop` 结束最近的范围。push/pop 像给代码配括号，内层括号先关闭；range 是这种有开始、结束的区间，不是一个 GPU 计算指令。

GPU 启动通常是异步的：CPU 提交后可能立即继续，不代表计算完成。前面的 `cudaDeviceSynchronize` 使预热真正结束，后面的同步保证目标 GPU 工作结束再关闭范围。NVTX push/pop 像压栈/出栈，为其中提交的 launch 赋予名字；一般 NVTX 归属依据提交时的 CPU 范围，不能简单按 GPU 时间戳是否落入范围判断。

**预期观察：** 全程序 30 次同名启动，命名范围里 20 次。上传、下载、CPU 核对不在目标范围。若删掉同步，标记本身不会补上 GPU 等待；这里仅作纸上推理，不要求改代码。

**理解检查：** 为什么 CPU 的一次 pop 不是 GPU 的完成通知？这解释了 [lec10](lec10.md) 第 3 节强调的哪种异步误读？

## 3. 编译和普通运行，先排除程序错误

**目的与命令：** 未来先独立查询环境，不把旧 sm_89 记录当作当前设备能力：

```bash
nvidia-smi
nvcc --version
ncu --version
ncu --list-sections
```

`nvidia-smi` 查看可见 GPU 和驱动，驱动负责操作系统与设备交互；`nvcc` 是 CUDA 编译器，`--version` 查版本；**NCU（Nsight Compute）**是分析 GPU kernel 的工具，`--list-sections` 列出可用 section（按问题组织的指标组）。CUDA Toolkit 是包含编译器、头文件等的开发工具集合。分别确认可见 GPU/驱动、CUDA 编译器版本、NCU 版本和可用指标组；compute capability 是 GPU 指令能力编号，还应按实际设备确认。`sm_89` 仅为下面命令的目标架构示例，执行前替换为设备和 Toolkit 支持的值。

```bash
mkdir -p build log
nvcc -std=c++17 -O3 -lineinfo -arch=sm_89 kernels/main.cu -o build/lesson10 -ldl
```

`mkdir -p` 创建缺失目录；`-std=c++17` 选择语言标准，`-O3` 优化，`-lineinfo` 保留源码映射，`-arch` 指 GPU 架构，`-o` 指产物。链接是把程序引用的外部函数实现接入可执行程序；Linux 下 `-ldl` 链接 NVTX3 可能使用的动态加载函数。源码需要完整的 `<nvtx3/nvToolsExt.h>` 及内部依赖；已有安装不在搜索路径时用 `-I` 指向其真实 include 目录，不改成旧 NVTX、不加 `-lnvToolsExt`，也不自动下载 fallback。

随后单独运行：

```bash
./build/lesson10
```

**看什么：** 预期退出 0，打印 `PASS` 和 `n=16 block=256 warmup=10 measure=20 checked=16 errors=0`。这是验收条件，不是本次结果。失败先看第一条 CUDA 错误或 mismatch，再查是否重编。源码或共享头变化后必须显式重编，编译失败不继续用旧产物冒充成功。

**CUDA Event** 是在 GPU 的提交队列（stream，流）中记录的完成点；计时事件还能记录设备时间戳。例如 start→乘法→stop，等待 stop 后才能读两个时间戳的间隔。它不等于 CPU 上那对 NVTX 标记，也不自动覆盖其他流的工作。**Duration** 是报告对象的持续时间，须确认对象和测量条件。

本程序没有 CUDA Event 独立计时输出；不可从一次普通运行的 PASS 推导性能，也不可把 NCU Duration 当成普通基线。若要独立计时，需要另行设计，不把含预热的共享 `time_kernel` 套入这个范围破坏实验。

## 4. 实验 A：只用 Profiler API 排除预热

前面的普通运行回答了“乘法结果对不对”，还没有回答“分析器看的是哪次乘法”。预热和目标启动的函数名、形状完全相同，单靠函数过滤无法排除预热；程序中的 Start/Stop 恰好提供一个额外的开关。如果让分析器开始时关闭采集，那么前十次虽然执行，却不应成为目标报告。这里首先检验这个区间假设，而不是从 Duration 猜报告身份。

**目的：** 保持程序不变，只改变分析器选择条件。先用 `ncu --help` 核对安装版本选项，在本课目录执行：

```bash
mkdir -p log
ncu --profile-from-start off --set basic -k 'regex:vector_mul' -c 1 \
  -o log/api ./build/lesson10
```

`--profile-from-start off` 使采集起初关闭，到程序的 Start API 才开启；`--set basic` 选择基础指标组，`-k` 用正则匹配函数名，`-c 1` 只采一个符合条件的启动，`-o` 指报告前缀。已有同名报告时手工改名，不加覆盖选项。

**看什么与解释：** 核对报告中的 PID、逻辑 GPU0、函数名、grid=1 和 block=256，再结合筛选配置确认属于目标区间。不能把报告内部 launch 编号直接当作源码循环下标。相同形状的 20 次启动仅凭名字/形状无法区分，需要调用信息，必要时用 NSYS（Nsight Systems，系统时间线工具）核对 API、范围与 GPU 活动。

**异常检查：** 无 kernel 时先查启停是否匹配、过滤名字及工具报错，不放开全部过滤后把任意报告算作成功。未采到指标不填零；采集成功也不自动证明对象选对。

## 5. 实验 B：只换为 NVTX 名字，再练习 skip

API 开关回答“现在允许采吗”，NVTX 名字则回答“这次提交属于哪段业务”。本程序把目标循环放在名为 measure 的范围内，预热在范围外，所以两种方法理应选择同一组候选启动。换用 NVTX 是检验另一种定位依据，不是换了乘法 kernel，也不是预期得到算法加速。若两份报告的启动身份不一致，应先解释筛选差异，而非比较指标大小。

**目的与命令：** 程序不改，另建报告，用业务范围筛选：

```bash
ncu --nvtx --nvtx-include 'measure/' --set basic -k 'regex:vector_mul' -c 1 \
  -o log/nvtx ./build/lesson10
```

默认这里 NVTX push/pop 跟随发起提交的 CPU 线程；线程是进程内的一条执行路线，进程内另一条 CPU 线程提交的工作不应被默认为也在此范围。`--nvtx` 启用 NVTX 过滤，`--nvtx-include` 选择范围。`measure/` 末尾斜杠表示 push/pop 范围，不是文件夹，也不是可以随便省略的字符；start/end 形式使用不同匹配语法。本课只实现 push/pop。

**预期观察：** A 和 B 都应排除预热，但机制不同：A 按采集启停，B 按提交时的命名归属。二者可以合用，本轮先分别理解。**replay（重放）**是 NCU 为分轮收集指标而重复执行工作，一轮叫 pass；筛选条件决定选谁，重放决定如何重复采，不把两种过滤器合用误称为新增 replay 模式。

再独立把 B 命令增加 `-s 1`，报告改名为 `nvtx_second`，其他不变。`-s` 跳过符合当前过滤条件的启动，预期选第二个目标启动，而非整个程序的第二次启动。**不要沿用前课的 `-s 10`：** 此处预热已在范围外，额外跳 10 次会跳过目标中的前 10 次。

**异常检查与理解检查：** 选错先核对范围语法和 skip 的计数对象，再检查是否误用旧报告。解释“同名”“同参数”“同一次业务调用”为什么是三种不同身份。

## 6. 最后才引入 Python 子进程

一个**父进程**可创建另一个独立运行的**子进程**，它们有不同 PID 和各自的地址空间；例如 Python 只负责启动已经编译好的 C++ 程序，GPU 工作可以完全属于 C++ 子进程。一个进程在某设备上的 CUDA 资源和状态容器叫 **context（上下文）**；它不是父子关系，也不是设备号。

到这里，启动身份都只在一个 CUDA 进程内区分。现在增加一个只负责启动程序的 Python 父进程：真正分配数组、计算向量乘和核对结果的仍是 C++ 子进程，输入输出及每线程计算完全不变。父进程的 PID 和 GPU 工作所属 PID 因而不同；只盯住最初启动的 Python，就可能得到“程序运行了却没有 kernel”的现象。将 worker 数增至 2 时，两份子进程各有自己的 a、b、c，不是把同一个向量拆成两半，也没有相互传递结果。这使我们能在不引入新算法的情况下检验进程选择。

**目的与代码：** 读已有 [workers.py](kernels/workers.py)。默认 `WORKERS=1`，Python 标准库只启动已经编译好的 lesson10，打印子进程 PID/退出码并等待；Python 本身不计算、不编译，也不写动作日志。

```bash
python3 kernels/workers.py
```

先看子进程是否 `PASS`、退出码是否 0；找不到程序就回到步骤 3，不在 worker 内隐式构建。之后才采子进程：

```bash
ncu --target-processes all --profile-from-start off \
  --nvtx --nvtx-include 'measure/' --set basic -k 'regex:vector_mul' -c 1 \
  -o log/workers python3 kernels/workers.py
```

`--target-processes all` 跟踪 Python 启动的 CUDA 子进程。核对报告 PID 与脚本打印的 PID；父进程没有 kernel。

**单变量对照：** 只将 `WORKERS=1` 改为 2，再普通运行与采集，报告另名。两个进程各自使用 GPU0、独立分配数组，可能竞争资源。`-c 1` 在多进程中的计数作用域按本机 NCU 版本与实际报告核对，不凭它猜进程树只有一份结果。进程同时存活不证明 GPU kernel 并发，分析器也可能改变原调度；需要时间线验证。MPS 是 NVIDIA 的多进程 GPU 服务机制，可能影响进程间执行组织，本练习不需要它，不修改系统 MPS 配置。

## 7. 巩固练习与完成标准

依次单独实施，每次恢复其余基线；修改 C++ 常量后重做步骤 3。

`1<<24` 是整数左移，值为 2²⁴=16,777,216；每个 float 4 B，三个数组就是 `3×4×2²⁴ B=192 MiB`（1 MiB=2²⁰ B）。后面的 L2 指 GPU 的二级缓存，容量因设备而异。

| 唯一变量 | 目的及预期观察 | 不符合预期先查 |
| --- | --- | --- |
| n：16→257 | 检查第二个 block 的单个有效输出，必须 checked=257 | grid、边界与旧产物 |
| 固定程序，`-c 1→2` | 报告选择两个目标启动，不是算法加速 | 过滤范围、版本计数规则 |
| 正确性通过后 n→`1<<24` | 三个 GPU 数组约 192 MiB，建立独立大输入配置 | 显存/host 内存；不保证超出 L2 |
| 固定大 n，block：256→128 | 看启动形状、资源和 Duration 的变化 | 是否同时改变了其他变量 |

完成标志是能证明“哪个 PID、哪个设备、哪个函数、哪段范围、哪个匹配启动”，并解释 `measure/` 与不加 `-s 10` 的原因，而不是仅交付一份报告。重放可能改变缓存与调度，任何 NCU 耗时均不能充当无分析器基线。CUDA Graph 是把任务和依赖预先组织成图，node 是其中一项任务（如一次乘法），图可以整体重复提交；这与 NCU 为收指标而重放不同。CUDA Graph、真实框架训练与独立性能统计未在此入口实现，讲义讨论不等于本实验覆盖。

## 8. 操作入口、旧记录与未验证项

理解底层命令后，在本课目录分别执行：

```bash
CUDA_ARCH=sm_89 bash commands/compile.sh
bash commands/run.sh
bash commands/profile.sh
```

三个脚本均无位置参数、按自身位置定位；仅 compile 必填 `CUDA_ARCH`（按设备替换），固定 C++17、`-O3 -lineinfo` 并加 `-ldl`，生成 `build/lesson10`，编译前移除旧目标，失败即停止。run/profile 缺产物报错，不隐式构建，改常量/共享头后须显式重编。profile 针对实际源码同时使用 `--profile-from-start off`、`--nvtx --nvtx-include 'measure/'`，按函数基名筛选 vector_mul，`-c 1`，**不加 `-s 10`**；固定 kernel replay/cache all/clock base。该脚本验证 API 与 NVTX 条件的交集；A/B 各自独立对照和 Python worker 仍执行前文的明确命令，不放入分派器，不把脚本交集采集当作 A/B 两份独立证据。

新日志为 `log/compile.log`、`log/run.log`、`log/profile.log`，含展开命令、输出与 `exit_code`；重跑覆盖各自新日志，对照需先另存。报告 `log/profile.ncu-rep` 已存在则拒绝覆盖，NCU 失败即停止，报告内容由实验者检查。旧证据原样迁至 `log/legacy_commands/<原相对路径>`，原始命令不改；旧 compile-only **不代表改后源码或新脚本**。

**当前状态：** 本次 stride 模板源码仅通过静态文本检查，相关脚本 `bash -n` 通过；未编译运行，旧日志不能验证当前模板源码与调用。

**改前版本的历史验证：** 当时已执行 `CUDA_ARCH=sm_89 bash commands/compile.sh`（nvcc 13.4、C++17、`-O3 -lineinfo -arch=sm_89`，另加 `-ldl`），见[编译日志](log/compile.log)，exit 0；随后执行 `bash commands/run.sh`，见[运行日志](log/run.log)，PASS/exit 0。当时只验证所选入口配置，未测试全部参数；profile 当时仅通过 `bash -n`，未执行 NCU、NSYS、workers 或 SASS 检查，无性能结论。[旧 compile-only 日志](log/legacy_commands/compile/source_check.log)仅作历史证据。