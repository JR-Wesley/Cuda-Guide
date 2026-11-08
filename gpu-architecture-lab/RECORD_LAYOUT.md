# 专题构建与记录布局

9 个可执行专题均使用独立的 `commands/<action>/run.sh`：打开一个动作目录即可读到该动作的实际命令与日志。没有万能分派脚本，也没有转调旧入口的包装器。源码仍集中在专题内，所有动作共享 `build/<variant>/`，不复制 CUDA 实现。

## 1. 目录和日志

以 [Warp Functions](04_sync/04_warp_functions/README.md) 为例：

```text
04_sync/04_warp_functions/
├── README.md / RESULTS.md / DEBUGGING.md
├── main.cu / check_static.py / debug_contracts.py
├── commands/
│   ├── README.md
│   ├── compile/run.sh
│   ├── compile/sm_89-release.log
│   ├── compile/sm_89-debug.log
│   ├── run/run.sh
│   ├── run/sm_89-release.log
│   ├── static/run.sh
│   ├── static/sm_89-release.log
│   └── ...                     # sanitizer/debug/gdb/jit 各自独立
├── build/
│   ├── .lock                   # 防止同专题并发写入
│   ├── sm_89-release/
│   │   ├── .main.compiled      # 仅表示显式编译完整成功，不是指纹
│   │   ├── bin/warp_functions
│   │   └── main.ptx
│   └── sm_89-debug/bin/warp_functions
└── results/                    # 已有历史证据，当前动作不写入
```

日志固定为 `commands/<action>/<variant>.log`，与该动作脚本同目录。重跑仅覆盖此动作、此编译配置的日志，失败也记录退出码；不合并到其他动作，不创建时间戳归档、history 或自动快照。日志记录实际命令及参数、stdout/stderr、逐步和整个动作的退出码；设备/工具信息由相应步骤输出。工具返回跳过不等于验证成功。

静态解析器需要的步骤输出暂存在同动作的 `.steps-<variant>/`，正常退出或可捕获中断后清理；其内容已输出到动作日志。强制终止可能留下临时文件或不完整日志，不能视为成功。`flock` 拒绝同专题并发调用，避免共享产物被同时改写；动作执行期间不要修改源码。

## 2. 编译与复用是两个显式步骤

1. 先运行 `commands/compile/run.sh`，编译本配置的程序及后续静态检查所需的 PTX/cubin。**每次调用 compile 都重新编译**，不做哈希、依赖扫描、工具状态探测或自动缓存判定。
2. 再运行 `run/static/sanitizer/jit/gdb` 等独立动作。它们检查成功编译标记和必要文件，只复用产物；缺失时返回非零并提示先编译，不调用编译入口。
3. 修改源码、头文件、编译选项或工具链后，使用者必须主动重新 compile，再重跑受影响的验证。成功标记不是版本校验；旧日志不会自动失效或消失，阅读时必须核对对应编译和运行条件。
4. 编译开始即撤销本组成功标记；编译失败删除该输出的旧文件和部分文件，整个编译动作成功后才恢复标记，防止失败后继续运行旧程序。不同真实配置仍放在不同 variant；不同源程序使用独立二进制。

`types`（访存重载）和 `debug`（setmaxnreg）是**显式的编译契约实验**：正负探针每次调用都重新编译/组装并核对原诊断，不运行 GPU，也不隐式重建主程序。这与运行动作偷偷触发编译不同。

## 3. 从专题目录逐项执行

先查询设备、compute capability、驱动和 Toolkit。下面的 `sm_89` 只是示例，换成实际支持的目标；warp 默认是 `sm_80`。

```bash
cd gpu-architecture-lab/04_sync/04_warp_functions
CUDA_ARCH=sm_89 bash commands/compile/run.sh
CUDA_ARCH=sm_89 bash commands/run/run.sh
CUDA_ARCH=sm_89 bash commands/static/run.sh
CUDA_ARCH=sm_89 bash commands/sanitizer/run.sh
CUDA_ARCH=sm_89 bash commands/jit/run.sh

# gdb 使用真正的调试配置；debug 动作本身仍使用 release
CUDA_ARCH=sm_89 BUILD_MODE=debug bash commands/compile/run.sh
CUDA_ARCH=sm_89 bash commands/gdb/run.sh
```

`compile` 中的 `-O3 -lineinfo` 是优化构建并附带行号；`BUILD_MODE=debug` 使用 `-O0 -g -G`，不能用它的耗时代替优化基线。`static` 读取已编译的 PTX，反汇编现有二进制并运行解析器。`jit` 仍需真实驱动执行，原生 cubin 通过不保证 PTX JIT 通过。

其他条件配置：

| 专题 | 显式编译入口参数 | 后续独立动作 |
| --- | --- | --- |
| setmaxnreg | `CUDA_ARCH=sm_89 BUILD_MODE=baseline bash commands/compile/run.sh` | `baseline`、`baseline-sanitizer` |
| setmaxnreg | `CUDA_ARCH=sm_90a BUILD_MODE=dynamic bash commands/compile/run.sh` | `run`、`static`、`sanitizer` |
| Async Barriers | `TARGET=tx90 bash commands/compile/run.sh`（固定 sm_90） | `tx90` |
| Thread Scopes | `TARGET=cluster bash commands/compile/run.sh`（固定 sm_90） | `cluster` |
| Thread Scopes | 默认 `TARGET=standard` | 包括 `system`；运行时仍检查 CPU/GPU 握手能力 |

## 4. 历史证据保持原样

既有 `results/*.log`、历史报告和其中的原命令不改写、不移动、不删除。`results/README.md` 继续解释历史来源，当前动作目录说明新用法。`HISTORICAL IMPORT` 不是本轮验证；不能将旧数值换链到新动作日志。

分析器报告优先与动作脚本、日志就地存放，只有其他动作确需共同读取时才放入共享 build。异步执行的报告仅由 profile 内的分析步骤读取，新报告因此使用 `commands/profile/<variant>-timeline.sqlite` 和 `<variant>-timeline.nsys-rep`。重跑覆盖同配置的新报告与日志，其他配置不受影响；采集失败不会继续解析遗留报告。既有 `build/sm_89-release/timeline.*` 仍是历史报告；若存在旧布局的 `current_timeline.*` 也保持原样，不自动移动、覆盖或归档。

公共层仅负责低层日志、锁、执行和产物检查，见 [tools](tools/README.md)。实验解释与状态要求见 [实验协议](EXPERIMENT_PROTOCOL.md)。
