# U5.1–U5.4：cuTile Array、Tile、归约与 MMA

## 范围与当前状态

实现入口：[`experiments/cutile_suite.py`](../../experiments/cutile_suite.py)，统一运行入口为
`python3 commands/learn/run.py cutile`，使用已有 `.venvs/cutile`。默认仅做小规模功能验证，不开启计时。
输入生成、CPU 数学参考、容差、两次 poison/check 均由
[`benchmarks/CONTRACT.md`](../../benchmarks/CONTRACT.md) 约定的公共 runner 负责。
适配器只上传共享 NumPy 输入、预分配独立连续输出、提交 kernel；没有私有随机输入、
私有参考结果或放宽容差。每次 launch 复用参数和设备数组，在当前 CuPy stream 提交，
不在 launch 内申请输出、导出编译文件或执行主机同步。

**已有 GPU 功能记录：默认 15/15 用例 PASS，每项完成 2 次 poison/launch/CPU-reference 校验。**
证据为 [`runs/20261105T060543Z-learn-osusvmpm/cutile/results.json`](../../runs/20261105T060543Z-learn-osusvmpm/cutile/results.json)，
模式为 `functional-only`；[功能日志](../../runs/20261105T060543Z-learn-osusvmpm/03-cutile-functional.log)
记录 15 passed、0 failed、退出码 0。15 个用例均保留实际 TileIR 导出，文件及 SHA256 已核对。

[运行 manifest](../../runs/20261105T060543Z-learn-osusvmpm/manifest.json) 保存命令、环境变量和退出码，
其中 `functional_only=true`、`compare_configs=false`、`selected_ids=null`；
[源码哈希清单](../../runs/20261105T060543Z-learn-osusvmpm/source-sha256.json) 对应同目录 `source/` 快照。
本页更新只核对既有记录，不重新执行用例。

通过范围为 U5.1 向量、U5.2 转置/行和、U5.3 **本地最小 GEMM**。
匹配版本的官方 tutorial matmul 示例尚未运行；U5.4 可选第二组 tile 的 2 个用例也未运行。
没有 PTX/SASS、资源或性能测量，不能将默认用例通过等同于全部 U5 学习任务完成。

## API 依据与版本

直接读取已有虚拟环境的安装源码和 `inspect.signature`，没有根据其他 DSL 推测 API：

| 项目 | 本次读取的版本/位置 |
| --- | --- |
| Python | **3.12.15**，`.venvs/cutile/bin/python` |
| cuda-tile | **1.7.0** |
| cupy-cuda13x | **14.2.0** |
| NumPy | **2.5.4** |
| GPU / compute capability | NVIDIA GeForce RTX 4060 Laptop GPU / **8.9** |
| 驱动版本 | `nvidia-smi --query-gpu` 记录 **596.21** |
| CUDA driver API / CuPy runtime | **13020 / 13020**（13.2） |
| 系统 nvcc / ptxas / tileiras | `/opt/cuda/bin/`，均为 **13.4 / V13.4.92** |
| 基础模式 | [`examples/cutile_vector_add.py`](../../examples/cutile_vector_add.py) |
| kernel/load/store/transpose/sum/mma/astype | `.venvs/cutile/lib/python3.12/site-packages/cuda/tile/_execution.py` |
| launch、CallingConvention | 同目录 `_cext.pyi`，以及安装模块的实际签名 |
| MMA 形状与类型检查、TileIR lowering | 同目录 `_ir/ops.py` 的 `mma_impl` / `TileMma` |
| 导出签名与接口 | 同目录 `compilation/_signature.py`、`compilation/_export.py` |

当前工作负载的 `metadata.provenance` 和导出报告仅记录 Python 与已安装包版本，不读取上述 API 文件或生成源码指纹。历史报告曾记录这些文件的路径和 SHA256，原始记录保留，不代表当前动作仍生成指纹。设备、驱动和系统编译工具见
[主机/工具链日志](../../runs/20261105T060543Z-learn-osusvmpm/00-E0.1-host.log)，
Python 环境、CUDA driver API/runtime 和 GPU context 检查见
[cuTile 环境日志](../../runs/20261105T060543Z-learn-osusvmpm/02-cutile-gpu.log)。
系统 Toolkit 13.4 与运行时/driver API 13.2 是不同记录，不能混为同一版本。
此处兼容性结论仅限该环境和已运行的默认用例，不推广到其他 GPU，也不据此断言任意 PTX 可 JIT。

已确认的关键 API：

```python
ct.launch(stream, grid, kernel, kernel_args)
ct.load(array, index, shape, padding_mode=ct.PaddingMode.ZERO)
ct.store(array, index, tile)
ct.transpose(tile)                  # 二维 Tile 的两个轴互换
ct.sum(tile, axis=1)                # 默认 keepdims=False
ct.mma(x, y, acc)                   # 保持 acc 的 dtype；f16 输入支持 f32 累计
ct.astype(acc, ct.float16)
```

`ct.load`/`ct.store` 的 `check_bounds` 默认是 `True`，本实现保持此默认值。
未使用对齐/整除假设、`check_bounds=False`、occupancy、TMA 或 warp 数提示。

## U5.1：Array 与 Tile、Copy/Add

源码：[`vector/cutile_ops.py`](../../experiments/vector/cutile_ops.py)。

- Array 是设备全局内存中的数组；Tile 是 kernel 中的逻辑值，由编译器决定线程、寄存器和数据搬运映射。
- `BLOCK=256` 是 256 个逻辑元素，不是「256 个 CUDA 线程」。
- `ct.bid(0)` 是 tile block 索引；`ct.load(a, (bid,), (256,))` 从
  `bid * 256` 开始加载。索引是 **tile 空间**索引，不是元素起点。
- 尾部 load 用零填充，store 忽略越界元素。grid 为 `ceil(N / 256)`，没有完全在数组外的 tile。
- Copy 写入独立数组；Add 分别加载共享的 A/B，再执行 FP32 加法。
- `ct.launch` 接收 stream、三维 grid、kernel 和参数元组，而非显式线程块大小。

## U5.2：真实转置和行归约

### Transpose

源码：[`transpose/cutile_ops.py`](../../experiments/transpose/cutile_ops.py)。
输入为 C-contiguous `(M,N)`，输出单独分配为 C-contiguous `(N,M)`，不是 `A.T` 视图，
也不靠修改 stride 伪装搬运。加载 `(16,32)` tile，执行 `ct.transpose` 得到 `(32,16)`，
以互换后的 tile 索引 `(col,row)` 写回。
矩形 tile 和 `37x53` 输入同时检查轴、tile 索引、M/N 尾块语义。
局部转置由编译器映射；源码本身不能证明采用了共享内存或某种访存合并方案。

### Row sum

源码：[`reduction/cutile_ops.py`](../../experiments/reduction/cutile_ops.py)。
一个 tile block 处理一行；列数向上补至 2 的幂：127→128、129→256、4097→8192。
加载 `(1,BLOCK)` 的 FP32 tile，越界元素为加法单位元 0，`ct.sum(..., axis=1)`
产生 `(1,)` tile，再写到一维 `(rows,)` 输出。
不是全矩阵总和，没有原子操作，也不需要依赖输出初值。8192 宽的逻辑 tile 不等于 8192 个线程。
仅支持这些小用例的单 tile 行归约，不声称是任意超长行的最优实现。

## U5.3：FP16 MMA，FP32 累计，FP16 输出

源码：[`gemm/cutile_ops.py`](../../experiments/gemm/cutile_ops.py)。
输入 A 为 `(M,K)`、B 为 `(K,N)`，均为 C-contiguous FP16；输出为独立的 `(M,N)` FP16。
每个 tile block 计算一个 `(BM,BN)` 输出块：

1. `ct.full((BM,BN), 0.0, dtype=ct.float32)` 建立 FP32 累计值。
2. K 循环执行 `ceil(K/BK)` 次，分别加载 `(BM,BK)` 和 `(BK,BN)`。
3. `ct.mma(x,y,acc)` 直接表达矩阵乘加，**没有用标量 GEMM 冒充 MMA**。
4. 最后仅做一次 `ct.astype(acc, ct.float16)`，再 bounds-checked store。

所有 load 均 zero-pad，保证 M/N/K 尾部有效；K 尾块中的补零乘积不会污染有效输出。
默认 BK=16，K=32 用两次完整迭代；K=29 用一次完整迭代和一次有尾块的迭代。
`37x53x29` 同时覆盖 M/N/K 尾部；FP32 累计与 FP16 输出容差由公共参考决定。
两个默认 GEMM 用例均已在记录环境中通过 2 次正确性校验。这里验收的是本地最小实现，
不是官方匹配版本 tutorial 示例；后者仍未运行。

一个 K tile 内，A 的元素参与 BN 列的计算，B 的元素参与 BM 行的计算。
这是算法层面的复用，不是已经测得的 L1/L2 命中或 DRAM 流量。
没有显式异步流水线、双缓冲、split-K、persistent 调度或 warp specialization；
实际流水线阶段及硬件 MMA 指令由编译器和目标决定，未观测。

## U5.4：两个 tile 配置和真实编译导出

| 配置 `(BM,BN,BK)` | 计划 / 启用方式 | 验证状态 |
| --- | --- | --- |
| `(32,32,16)` | U5.3，默认 | 2/2 GPU 用例 PASS，各 2 次校验，实际 TileIR 已保存 |
| `(32,64,16)` | U5.4，`--compare-configs`（suite 内为 `--scan-tiles`） | 可选 2 项未运行，没有该配置的 GPU 对照结果 |

两个配置均满足逻辑 tile 的 2 的幂、矩阵内维和累加器形状约束，开发时已通过安装版 1.7.0 的
CPU frontend TileIR 导出；这不替代第二配置的 GPU 验收。
第二配置只改变 BN，保持 BM/BK、输入和精度不变。BN 变大可能改变复用、padding 和资源压力，
但配置对比、目标指令和资源对比均未完成，不能断言第二配置更快。

### Artifact 回调

公共 runner 完成两次正确性校验后调用 `Workload.artifacts()`，此回调使用真实公开 API：

```python
signature = KernelSignature.from_kernel_args(
    kernel, args, CallingConvention.cutile_python_v1())
export_kernel(kernel, [signature], buffer,
              output_format="tileir_bytecode", bytecode_version="13.3")
```

实际记录的 15 个用例各自的 `artifacts/` 下均有非空 `kernel.tileirbc` 和
`compiler-artifacts.json`；15 份报告均为 `status="exported"`，bytecode version 为 `13.3`。
已逐项核对 30 个文件的 SHA256 与 `results.json` 的 `artifact_sha256` 一致，
TileIR 文件的 SHA256 也与各自导出报告一致，而非仅根据文件名判断导出成功。

含尾块 GEMM 的具体证据：
[导出报告](../../runs/20261105T060543Z-learn-osusvmpm/cutile/014-U5.3.cutile.gemm.37x53x29.t32x32x16/artifacts/compiler-artifacts.json)、
[原始 TileIR](../../runs/20261105T060543Z-learn-osusvmpm/cutile/014-U5.3.cutile.gemm.37x53x29.t32x32x16/artifacts/kernel.tileirbc)。
上述历史报告包含 SHA256；当前报告保留签名、API 名称、包版本和导出范围，不再生成 TileIR 或源码指纹。仍导出真实 bytecode 并检查它非空。回调在导出失败时会记录
`status="not measured"` 和异常，不会生成假的 PTX 或吞掉 launch/正确性错误；本次 15 项未走该失败路径。
固定 bytecode 13.3 且不传 `gpu_code` 是安装接口支持的架构无关前端导出，不需要在导出时查询 GPU
或探测 `tileiras`。它是同一 kernel/参数的**单独导出**，并非从 JIT cache 截获的最终机器码；
不能据此声称与实际 JIT cubin 完全相同。签名从实际数组推导，可能含地址/stride 假设，
仅用于该次实验分析，不作为能对任意数组直接加载运行的通用二进制。

这些导出文件不包含 PTX、cubin 或 SASS；寄存器数、共享内存用量、硬件映射、性能和缓存命中率
仍为 **未测量**，不能从 GPU 正确性通过或 `ct.mma` 源码推导这些指标。
后续已有的离线归档说明见 [06-static.md：cuTile 的机器码证据缺口](06-static.md#cutile-explicit-missing-machine-code-evidence)：
其中保留了 7 个代表性用例的 TileIR 和报告，也明确没有 cuTile 机器码、资源或性能结论。

## 精确 ID

ID 以学习计划编号为前缀：向量为 `U5.1`，transpose/sum 为 `U5.2`，
基线 GEMM 为 `U5.3`，可选第二个 GEMM tile 为 `U5.4`。
默认 registry 按 `U5.1 → U5.2 → U5.3` 排列；启用扫描时在末尾追加 `U5.4`，
保持计划数字顺序。原无计划编号前缀的 ID 不再使用；`--ids` 支持以下完整 ID，
也支持 `U5.3` 等计划前缀（选择该前缀下的全部用例）。

默认运行以下 15 项，均已在上述记录中 PASS，每项 2 次校验：

```text
U5.1.cutile.copy.n1.b256
U5.1.cutile.copy.n255.b256
U5.1.cutile.copy.n256.b256
U5.1.cutile.copy.n4099.b256
U5.1.cutile.add.n1.b256
U5.1.cutile.add.n255.b256
U5.1.cutile.add.n256.b256
U5.1.cutile.add.n4099.b256
U5.2.cutile.transpose.32x32.t16x32
U5.2.cutile.transpose.37x53.t16x32
U5.2.cutile.sum.1x127.b128
U5.2.cutile.sum.3x129.b256
U5.2.cutile.sum.2x4097.b8192
U5.3.cutile.gemm.32x32x32.t32x32x16
U5.3.cutile.gemm.37x53x29.t32x32x16
```

仅配置比较时加入（独立 learn 动作 `--compare-configs`，suite 内为 `--scan-tiles`）；以下 2 项尚未运行：

```text
U5.4.cutile.gemm.32x32x32.t32x64x16
U5.4.cutile.gemm.37x53x29.t32x64x16
```

默认 U5.1–U5.3 的已实现 ID 有上述 GPU 验收记录；U5.4 仅已实现，不能混列为通过。
没有零维/空输入、负 stride、任意 dtype、Softmax、自动调优或任意 GEMM tile 的支持声明。

## 命令（从 cuda-python-dsl-lab 根目录）

独立 learn 动作使用已有环境，覆盖本动作 commands/learn/log/，保存命令、退出码及日志，不自动快照，
不要覆盖上述历史记录。以下默认命令对应已有的 15 项功能记录；本次文档更新没有重跑：

```bash
# 默认 15 个小功能用例，各做 2 次正确性校验；不做性能计时
python3 commands/learn/run.py cutile

# 支持计划前缀：只选择 U5.3 的两个本地最小 GEMM 用例
python3 commands/learn/run.py cutile --ids U5.3

# 也支持完整 ID：只选择含尾块的基线 GEMM
python3 commands/learn/run.py cutile --ids U5.3.cutile.gemm.37x53x29.t32x32x16
```

可选配置比较命令如下，**U5.4 尚未运行**；它不是默认验证，也不会自动开启计时：

```bash
# 注册第二组 tile，再选择基线及第二配置，共 4 个小 GEMM 用例
python3 commands/learn/run.py cutile --compare-configs --ids U5.3,U5.4
```

仅传 `--ids U5.4` 不会自动注册第二配置，必须同时传 `--compare-configs`，
该开关由独立 learn 动作转为 suite 的 `--scan-tiles`。
同 op/shape 的两个配置由公共 runner 取得相同的 CPU 数组及 SHA256。
只有显式 `--bench` 才启用性能计时；上述运行记录和所有命令示例均为功能模式。

只查看 registry、不运行 GPU 用例时可用：

```bash
.venvs/cutile/bin/python -B -m experiments.cutile_suite --list-ids
.venvs/cutile/bin/python -B -m experiments.cutile_suite --scan-tiles --list-ids
```

## 已有 GPU 验证与结论边界

以下统计来自上述 `results.json` 和原始功能日志，不是重新执行测试的结果：

| 计划 / 算子 | GPU 功能结果 | 每项正确性重复 | 实际 TileIR 导出 |
| --- | --- | --- | --- |
| U5.1 copy / add | 8/8 PASS | 2 | 8 份 |
| U5.2 transpose | 2/2 PASS | 2 | 2 份 |
| U5.2 row sum | 3/3 PASS | 2 | 3 份 |
| U5.3 本地最小 GEMM，`(32,32,16)` | 2/2 PASS | 2 | 2 份 |
| U5.4 第二 tile，`(32,64,16)` | 未运行 | 未测量 | 本次 GPU 记录无该配置产物 |

- 默认合计 **15/15 PASS、30 次正确性校验**。公共 runner 每次先用 NaN poison 输出，
  再 launch、同步、复制回 CPU 并校验；也检查输出为连续存储。各用例目录保留共享输入、
  CPU reference、`actual-0.npy` 和 `actual-1.npy`。
- 真实数组的 artifact 回调已经运行；15 个 TileIR 文件和 15 份导出报告已核对内容与哈希。
  这不是只验证 Python 导入或离线 frontend 的结果。
- `mode="functional-only"`，所有 case 的 timing 均为 `not measured: functional-only run`。
  `startup` 中的 prepare/first-launch 同步耗时包含上传、分配或 JIT，**不是 kernel 性能测量**；
  cache 沿用既有项目缓存，未清空，也不声称测得 cold-cache 性能。
- 已验证的是记录环境上的默认形状、尾块、精度和公共 runner 集成，不是任意设备/Toolkit 的兼容性。
- 尚未完成：官方匹配版本 tutorial matmul 示例运行、U5.4 第二配置 GPU 运行及配置对照、
  PTX/SASS/机器码分析、寄存器与共享内存资源、硬件计数器、性能结论。
- 本次仅修订本文并只读核对历史证据，没有新增测试、GPU 运行或源码修改。
