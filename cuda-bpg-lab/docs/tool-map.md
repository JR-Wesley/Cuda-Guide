# 操作入口怎样连接源码、构建和证据

先理解“编译产生文件、运行读取文件”。在项目根目录执行独立的 `python3 commands/<action>/run.py`；每个脚本拥有自己的动作实现，不调用万能入口。

```text
labs/<阶段>/<实验>.cu + common/
          │
          └─ commands/build/run.py → CMake → build/sm_XX/bin/<实验>
                                                    │
commands/run/run.py ─────────────────────────────────┘
          └─ commands/run/log/{commands.jsonl,status.txt,samples.csv,...}
commands/static/run.py → 同一二进制 → commands/static/log/
commands/verify/run.py → 同一二进制 → commands/verify/log/
```

## 先单目标编译，再做一个动作

1. 先手动确认设备/Toolkit：`nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv`、`nvcc --version`；或使用 `python3 commands/env/run.py` 记录探测。某些工具不可用会出现在日志中，不代表其他探测失败。
2. `CUDA_ARCH=sm_89 python3 commands/build/run.py vector_add`：只配置、编译，不测试。等价的基础 CMake 步骤见 [Stage 1](../labs/01_measurement/README.md)。`sm_89` 只是示例，脚本不再自动推断架构。
3. `CUDA_ARCH=sm_89 python3 commands/run/run.py vector_add --variant scalar --n 8 --warmup 1 --repeats 2 --iterations 3`：读取共享产物，只运行一个程序。预期输出通过校验，先核对输出再解释时间。
4. `CUDA_ARCH=sm_89 python3 commands/verify/run.py vector_add`：执行主机/help 与 GPU CTest、边界、CSV、非法参数检查；不会编译。

缺少共享产物则提示先构建并返回非零。源码/编译参数变化后应显式重编译，不做自动状态发现、哈希或缓存框架。`build` 不指定目标会显式构建全部；学习时优先传单个目标。主机路径单独执行 `python3 commands/host-build/run.py` 后再执行 `python3 commands/host-test/run.py`，复用 `build/host/host_tests`，不需要 GPU。

## 其他动作的边界

- `static` 读取 PTX/SASS/资源，并保留原有指令检查；不是动态性能指标。
- `sanitizer` 保留四种工具及必要边界输入；只证明本次工具覆盖。
- `nsys` / `ncu` 分别采集时间线/指标；不重编译，插桩 CSV 不替代普通基线。缺工具返回 77，并记录 Not Tested。
- `scripts/session.py` 只记录命令；`scripts/build_support.py` 只提供显式构建和产物存在检查；`scripts/verification.py` 保留教学检查。不必先理解这些辅助代码才能学习 kernel。

每个动作的 `log/` 只存本次结果。重新调用同动作会清空其旧 log，包括 CSV、分析器报告和状态；不同动作不受影响。做对照时先阅读或手动保留结果，没有自动归档。程序自身的 CSV 禁止覆盖检查仍保留：同一次 verify 内会验证它；入口通过清理本动作输出允许下一次重跑。

`commands.jsonl` 包含 argv、工作目录、实际/预期退出码和日志名；`invocation.json` 保留输入参数与必要环境，`status.txt` 为整体状态。源码和二进制不复制到日志中。旧 `results/` 的时间目录、快照及原始命令不改写；旧 `lab.sh` 或 `scripts/*.py` 是当时入口，不能视为当前入口已验证。计时与 CSV 定义见 [测量方法](methodology.md)。
