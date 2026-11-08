# 02 — Triton E / T1 / T2：最小功能实现

## 状态与边界

共享 [adapter contract](../../benchmarks/CONTRACT.md) 的 Triton 后端已完成默认小规模 GPU 功能验证：**40/40 cases PASS，每个 case 输出投毒后校验两次，共 80 次校验**。原始证据见 [results.json](../../runs/20261105T060430Z-learn-ne9a7tkh/triton/results.json)；模式为 `functional-only`。实际 compiler artifacts 已保存，[静态报告](06-static.md) 已提供所选 GEMM、Transpose、Fusion 的 PTX/SASS 与资源证据。**未运行配置扫描、性能 benchmark 或 Profile；不能据此宣称提速或完成全部学习验收。**

运行使用已有 `.venvs/triton`，结果记录 Python `3.12.15`、PyTorch `2.15.0`、Triton `3.9.0`、RTX 4060 Laptop GPU、compute capability 8.9、CUDA runtime 13.2。原始命令与退出码见 [manifest](../../runs/20261105T060430Z-learn-ne9a7tkh/manifest.json)，设备/驱动信息见 [host 日志](../../runs/20261105T060430Z-learn-ne9a7tkh/00-E0.1-host.log) 与 [GPU 检查日志](../../runs/20261105T060430Z-learn-ne9a7tkh/02-triton-gpu.log)。环境背景另见 [ENVIRONMENT.md](../ENVIRONMENT.md)。

源码：

- [suite / adapter](../../experiments/triton_suite.py)：输入约束、预分配、实验目录、artifact closure、CLI 转发。
- [vector](../../experiments/vector/triton_ops.py)：Copy/Add/AXPY/Scale、融合和两 kernel 链。
- [reduction](../../experiments/reduction/triton_ops.py)：Row Sum、稳定 Softmax。
- [GEMM](../../experiments/gemm/triton_ops.py)：FP16 输入、FP32 累计、FP16 输出与 M/N/K 尾块。
- [transpose](../../experiments/transpose/triton_ops.py)：朴素与逻辑分块转置。
- [CPU adapter tests](../../tests/test_triton_adapter.py)：合约替身、fake launch；**不是 kernel 数值测试**。

## 实验 ID 与默认目录

`build_experiments()` 不创建设备张量。目录按 E0.1–E0.4、T1.1–T1.5、T2.1–T2.4 排列；共 40 个很小的 case。同一学习 ID 可有多个 shape/config，`--ids T1.1` 选择这一组，而非只选择一个 case。每个 case 都通过 `Experiment(id, op, shape, config, prepare)` 交给 runner；输入生成、SHA256、CPU reference、容差、输出两次投毒、同步及计时完全由 runner 管理。

| ID | 默认 case | 本后端实现范围 / 未完成验收 |
| --- | --- | --- |
| E0.1 | Add，N=1 | smoke 校验通过；本次 learn 另保留 host/GPU 检查日志 |
| E0.2 | Add，N=255 | host/kernel 边界见下文；不代替其余三种 DSL 的逐行讲解 |
| E0.3 | Add，N=4099 | 公共输入、两次投毒校验、结果与 artifact 保存已实际运行；未启用计时 |
| E0.4 | Copy，N=256 | 真实 PTX/cubin 已保存；所选 GEMM/Transpose/Fusion 另有离线反汇编报告；未运行 ncu/Profile |
| T1.1 | Copy/Add/AXPY，各 N=1,255,256,4099 | BLOCK=256，4 warps，覆盖整块和尾块 |
| T1.2 | Add，N=4099 | 默认仅 BLOCK=256/4 warps；扫描是显式可选项 |
| T1.3 | Sum/Softmax，各 (1,127),(3,129),(2,4097) | 单行单 program，4 warps；非二次幂和长行 padding |
| T1.4 | GEMM，(M,N,K)=(32,32,32),(37,53,29) | Tile=(16,32,16)，4 warps，2 stages；覆盖 M/N/K 尾部 |
| T1.5 | Softmax (1,127)/(1,129)；尾块 GEMM 两种 Tile | 功能校验通过；GEMM 仅改 BLOCK_M=16/32，已有静态指令/资源证据，未测耗时 |
| T2.1 | Transpose，(32,32)/(37,53)，各 naive/tiled | 同输入、真实连续输出，不是 stride view |
| T2.2 | 尾块 GEMM，Tile=(16,32,16)，4 warps，2 stages | 默认基线；warp/stage 候选仅 opt-in；未确认优化收益 |
| T2.3 | Fusion，N=4099，unfused/fused | 完全相同的两次 FP32 舍入公式；分别 2/1 次 kernel 启动 |
| T2.4 | Transpose (37,53) 两变体 + Fusion N=4099 两变体 | 两组 GPU 校验与 artifacts 已保存，并有静态分析；计时/Profile 验收仍不完整 |

上述 40 个默认 case 均已通过两次 GPU 校验，但 E 类工作负载、环境检查、静态分析与 Profile 是不同证据。功能通过不等于扫描、性能分析或全部学习任务已验收；T1.2/T2.2 扫描、T1.5 性能比较及 T2.3/T2.4 完整计时分析仍未完成。

## 数学语义与 launch 映射

### Host / JIT 边界

`prepare(cpu_arrays, config)` 先检查 NumPy dtype、shape、连续性及非空要求，再复制到 CUDA；只在准备阶段分配输入、输出及 unfused scratch。返回的 `Workload` 使用 `framework='torch'`，输出连续，`launch()` 不分配设备张量、不生成输入、不复制到 CPU、不调用 host 同步。第一次真实 launch 会触发 Triton JIT/runtime 分配，这不同于 wrapper 在稳态计时内分配张量。

所有 launch 使用当前 PyTorch CUDA stream，无自建 stream。首次编译/调用与稳态测量的分离、同 stream 的计时和正确性同步由 runner 负责。闭包持有输入、输出、scratch 与返回的 compiled kernel；引用更新不产生额外设备工作。

### Vector 与 fusion

- Copy：`out[i] = a[i]`。
- Add：`out[i] = round_fp32(a[i] + b[i])`。
- Scale：`out[i] = round_fp32(a[i] * 0.5)`，提供为辅助内核，不另扩默认测试集。
- AXPY / Fusion：`out[i] = round_fp32(round_fp32(a[i] * 0.5) + b[i])`。
- Unfused：Scale 写入预分配 scratch，然后 Add 读取 scratch；每次调用都会重写 scratch。
- Fused：在一个 kernel 中完成相同公式；显式 `enable_fp_fusion=False`，不允许将乘加收缩为单次舍入 FMA。这里“融合”指减少 kernel 启动与中间全局内存，不改变数学语义。

Grid 为 `ceil(N/BLOCK)`；每 program 操作 BLOCK 个逻辑元素，`offset < N` 控制 load/store。默认 4 warps，即 NVIDIA 上每 program 128 个线程；BLOCK=256 **不是** 256 个线程，BLOCK=1024 也不是 1024 个线程。元素到线程/寄存器的分配由 Triton 决定。

AXPY 与 Fusion 具有相同公式；[公共生成器 `benchmarks/core.py`](../../benchmarks/core.py) 按 **输入 family + shape** 确定 seed，而非按 op 或 DSL。相同 shape 的 Add/AXPY/Fusion 使用相同的两个 FP32 输入，因此输入 hash 相同；Copy/Scale 只包含第一个输入，其完整输入集合的 hash 不应与双输入算子混同。[本次结果](../../runs/20261105T060430Z-learn-ne9a7tkh/triton/results.json) 中 N=4099 的 AXPY 与各 unfused/fused Fusion case 均记录 `e11fd21b005c5da98a011ab4dc5da5ccea60810814e88817a01e5625fd7fdcbf`。Fusion 的 unfused/fused 配对保持输入、公式和舍入语义一致，仅改变 kernel 链。

### Row Sum / Softmax

Grid 为行数，每 program 由 4 warps 协作处理一行，逻辑 BLOCK 为 `next_power_of_2(cols)`；4097 列会扩展到 8192 个逻辑位置。Sum 的 padding 为 0，FP32 累加。Softmax 对有限输入计算：

```text
m = max(x)
z = exp(x - m)
y = z / sum(z)
```

Softmax padding 在 max 前为 `-inf`，减去有限最大值后仍为 `-inf`，其指数为 0。所有有效项仍在最大值和分母中；只存储有效列。`tl.exp` 为近似指数，reduction 顺序也不承诺等同 CPU；沿用公共容差，不修改它来隐藏失败。

只支持非空、有限 Softmax 输入；NaN/Inf 在 CPU prepare 阶段拒绝。当前单 program 行算法限制列数 <=8192，不提供 multi-pass/persistent reduction。warp 内/跨 warp 的 shuffle/shared-memory 通信由编译器决定；不能仅凭 `tl.sum` 源码断言具体通信指令或资源用量。

### GEMM

`A[M,K] @ B[K,N] -> C[M,N]`，A/B 为连续 FP16，`tl.dot` 输入 FP16，accumulator 初始化为 FP32；所有 K tiles 累计完成后，仅在写回前转 FP16。不是 FP32/TF32 GEMM，也没有用标量循环代替矩阵点积。

Grid 为 `(ceil(M/BLOCK_M), ceil(N/BLOCK_N))`。一个 program 负责一个输出 tile，K 维按 BLOCK_K=16 顺序遍历；A/B load 同时检查行、列及 K 上界，越界补 0，输出按 M/N mask 写回。A tile 在输出 N 方向复用，B tile 在输出 M 方向复用。默认 Tile=(16,32,16)，对照 Tile=(32,32,16) 只改变 M tile。

`num_stages` 是 Triton 编译器流水线选项，不是手写 async pipeline 的证据。默认 4 warps/2 stages 的两种 tile 已通过 GPU 校验；[静态报告](06-static.md) 在所选尾块 GEMM 中观察到 `mma.sync`/`HMMA.16816.F32` 与 `ldmatrix`/`LDSM`，两种 tile 的 `cuobjdump` 资源日志均报告 REG=40。这些是静态证据，不是 occupancy 或性能测量；资源日志的 `SHARED:0` 也不等于 launch 的总 shared memory 为零。没有 TMA、warp specialization、显式双缓冲、split-K、autotune 或 epilogue。额外的 stage 1/3、warp 8 扫描候选尚未验证，不保证所有设备支持或受益于这些配置。

### Transpose

输入为 `[M,N]` 连续 FP32，输出新分配为 `[N,M]` 连续 FP32，`out[j,i] = in[i,j]`，无算术或精度转换。

- Naive：1D grid，按输入线性偏移访问，输出地址为 `col*M + row`。逻辑上连续输入、跨步输出。
- Tiled：2D grid，默认 32x32 逻辑 tile；加载 `[tile_m,tile_n]`，使用 `tl.trans`，按 `[tile_n,tile_m]` 输出 tile 的地址写回，行列分别 mask。

Triton 自行决定布局转换和 shared-memory 使用。此处实现的是逻辑分块，不宣称手工 padded shared-memory tile、bank-conflict 消除或访存事务减少。[静态报告](06-static.md) 对 (37,53) 的两个版本均观察到 shared load/store 与 barrier；“naive” 不代表编译后不使用 shared memory。静态指令差异不能证明 tiled 更快或访存事务更少。

## Compiler artifacts

`Workload.artifacts()` 仅应在第一次 launch 后调用；提前调用会报错，不返回伪造代码。每个实际 kernel 的 launch 返回值保留在闭包中，从其 `asm` 导出：

- `<label>.ptx`：实际 PTX 文本；缺失时报错。
- `<label>.cubin`：实际 cubin bytes；缺失时报错。
- `<label>.ttir/.ttgir/.llir`：仅当该版本确实导出相应字符串时附加。
- `<label>.metadata.json`：实际 compiled kernel 的 target、warp/stage/shared、可用时的 register/spill 字段；缺失字段为 `null`，不填假零。

Unfused 同时导出 `scale.*` 和 `add.*`，避免只保存最后一个 kernel。公共 runner 已为各 case 建立独立输出位置，并在 [results.json](../../runs/20261105T060430Z-learn-ne9a7tkh/triton/results.json) 中记录 `artifact_sha256`，不同 case 的同名文件不互相覆盖。实际保存的文件示例：

- [尾块 GEMM artifacts](../../runs/20261105T060430Z-learn-ne9a7tkh/triton/024-T1.4/artifacts)：`gemm.ptx/cubin`、IR 与 metadata。
- [Unfused artifacts](../../runs/20261105T060430Z-learn-ne9a7tkh/triton/038-T2.4/artifacts)：同时保留 `scale.*` 与 `add.*`。
- [Fused artifacts](../../runs/20261105T060430Z-learn-ne9a7tkh/triton/039-T2.4/artifacts)：`vector.*`。

后续离线反汇编与分析见 [06-static.md](06-static.md) 和其链接的 [修正版静态汇总](../../runs/20261105T061056Z-static-3xunxkwj/analysis-v2/SUMMARY.md)。所选 Fusion 的 SASS 中，Scale 为 `FMUL`，Add 为 `FADD`，Fused 同时保留 `FMUL` 与 `FADD`，未观察到 `FFMA`；这是已保存二进制的静态观察，不是动态 launch trace。PTX 不等同 SASS，compiler/runtime register 元数据也不等同实测 occupancy、缓存命中率或 DRAM 流量。

## 串行运行入口

从项目根目录使用已有环境，独立 learn 动作只运行选定功能测试，日志覆盖 commands/learn/log/；环境探测另用 inspect 动作，不自动快照：

```bash
python3 commands/learn/run.py triton
python3 commands/learn/run.py triton --ids T1.4,T2.3
```

第一条对应上述完整默认集的实际运行，见 [manifest](../../runs/20261105T060430Z-learn-ne9a7tkh/manifest.json)；第二条是支持的按 ID 选择方式，不是额外运行记录。默认仅做小规模正确性检查，**不启用 `--bench`**。不要覆盖历史记录，也不要并行运行 GPU suites。

可选 `--scan` 只扩展小规模配置目录到 48 cases，不自动调优，不扩大到性能规模：

- T1.2：BLOCK=128/256/512/1024 固定 4 warps，然后固定 BLOCK=256 比较 4/8 warps；不做笛卡尔积。
- T1.5：增加单行 4097 列 Softmax，保留行数/warp 配置。
- T2.2：固定 GEMM tile；先加入 8 warps/2 stages，再固定 4 warps 加入 1/3 stages，与 4 warps/2 stages 基线比较。

`build_experiments(scan=True)` 本身仅生成目录。suite CLI 的 `--scan` 会将扩展目录交给 runner 执行，不是 dry-run 开关；learn 动作对应 `python3 commands/learn/run.py triton --compare-configs`。**这些扫描均未执行**，本次 [manifest](../../runs/20261105T060430Z-learn-ne9a7tkh/manifest.json) 记录 `compare_configs=false`。直接调用 suite 时，它消耗 `--scan`，原样转发 `--output/--ids/--bench/--warmup/--repeat/--batches` 等其余参数；共享 runner 提供 `--help`。

之后若显式选择 `--bench`，一次 unfused closure 的计时包含 Scale+Add 两次启动及中间读写，不能标为单 kernel 时间；fused closure 只有一次启动。目前未运行性能 benchmark，结果中的 startup 字段也不是稳态 kernel 时间。当前 adapter 不独立计时链中两个 component；T2.3 的完整验收仍需独立 component 测量或 profiler 证据。即使 AXPY 与 Fusion 输入 hash 相同，也不能将不同 case 的总时间相减当作 component 测量。

## T2.4 证据对与未完成验收

| 配对 | 固定变量 | 改变变量 | 本次证据状态 |
| --- | --- | --- | --- |
| Naive / tiled transpose | op、(37,53)、FP32、4 warps、公共输入 | 访问方式、逻辑 tile/grid | [Naive](../../runs/20261105T060430Z-learn-ne9a7tkh/triton/036-T2.4) / [tiled](../../runs/20261105T060430Z-learn-ne9a7tkh/triton/037-T2.4) 均 GPU 两次 PASS，已保存输出和 artifacts；[静态报告](06-static.md) 记录两者的 shared 指令/barrier；耗时/Profile 未测量 |
| Unfused / fused chain | op、N=4099、FP32 两次舍入、BLOCK=256、4 warps、公共输入 | 2 次启动+scratch vs 1 次启动 | [Unfused](../../runs/20261105T060430Z-learn-ne9a7tkh/triton/038-T2.4) / [fused](../../runs/20261105T060430Z-learn-ne9a7tkh/triton/039-T2.4) 均 GPU 两次 PASS，已保存输出和 artifacts；[静态报告](06-static.md) 记录两份/一份 kernel 及 `FMUL`/`FADD`；耗时/Profile 未测量 |

正确性状态与误差见 [results.json](../../runs/20261105T060430Z-learn-ne9a7tkh/triton/results.json)。已有功能与静态证据，但缺少计时数据、独立 component 测量和 Profile，完整验收仍未完成。当前不能回答哪组更快，也不能根据 tile、stage 或启动数推导实测占用率/带宽。

## CPU adapter 验证与 GPU 证据的区别

开发阶段另有仅使用 CPU 的 adapter 测试，使用既有环境：

```bash
.venvs/triton/bin/python -B -m unittest discover -s tests -p test_triton_adapter.py -v
```

开发阶段记录：6 个 CPU 测试通过；6 个 Python 文件 AST 检查通过；真实 Triton kernel 模块的 CPU 导入检查通过。CPU adapter 测试仍使用与已发布合约一致的 dataclass 替身，不等同真实 runner/GPU 集成测试。实际集成与 GPU 数值验证现由上述 [40/40 功能运行](../../runs/20261105T060430Z-learn-ne9a7tkh/triton/results.json) 提供，真实编译产物及反汇编证据见 [静态报告](06-static.md)。

测试验证目录 ID 顺序、小规模覆盖、opt-in 参数生成、CLI 转发、输入拒绝条件、Workload 输出 shape/dtype/连续性、artifact 收集行为、两步链 scratch 连接和 launch 内不再分配张量。测试将设备传输和 kernel 替换为 CPU/fake callable，artifact 内容是明确的测试替身；这些内容不会作为实验文件导出。测试不证明 Triton lowering、GPU 数值结果、实际 artifacts 或性能正确。
