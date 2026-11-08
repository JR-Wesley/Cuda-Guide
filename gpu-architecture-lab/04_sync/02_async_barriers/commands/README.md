# 独立动作入口

先读 [专题说明](../README.md)。以下命令从专题目录执行（不是 commands 目录）；也可用脚本绝对路径，脚本自行定位共享源码。每个链接直接打开该动作的实际命令，不转调万能入口。

```bash
bash commands/compile/run.sh
bash commands/run/run.sh
```

执行前按设备/Toolkit 设置同一 `CUDA_ARCH`。先确认功能，再按学习目标选择 static、sanitizer 或其他动作，不要求一次跑完。

`compile` 默认 `TARGET=standard`，生成 main/debug、main.ptx 以及独立 api_parity 程序。tx90 前运行 `TARGET=tx90 bash commands/compile/run.sh`，固定编译 sm_90。

| 动作脚本 | 目的 | 同目录日志 |
| --- | --- | --- |
| [compile](compile/run.sh) | 显式编译程序及所需 PTX/cubin；每次调用重新编译 | `compile/<variant>.log` |
| [debug](debug/run.sh) | 独立调试/反例验证；具体检查见脚本和专题说明 | `debug/<variant>.log` |
| [diagnose](diagnose/run.sh) | 原生/API parity 的 1/9 轮及 completion 隔离对照 | `diagnose/<variant>.log` |
| [isolate](isolate/run.sh) | 显式排除指定 kernel 的检查；不代替完整 sanitizer | `isolate/<variant>.log` |
| [run](run/run.sh) | 运行正常功能/性能实验及原有边界校验 | `run/<variant>.log` |
| [sanitizer](sanitizer/run.sh) | 检查正常路径，保留工具输出与非零退出码 | `sanitizer/<variant>.log` |
| [static](static/run.sh) | 读取已有 PTX/cubin，反汇编并执行原有静态检查 | `static/<variant>.log` |
| [tx90](tx90/run.sh) | 运行 SM90 transaction 示例并保留能力门控 | `tx90/<variant>.log` |

动作入口为 `commands/<action>/run.sh`；先显式执行 `commands/compile/run.sh`，其余执行/检查动作只复用 `build/<variant>/` 的已有产物，缺失即提示先编译，不隐式构建。修改源码、头文件、编译选项或工具链后必须主动重新编译；没有哈希缓存或自动失效判断。

日志位于脚本同目录 `commands/<action>/<variant>.log`，同动作同配置重跑覆盖（包括失败），其他动作日志不变。命令、实际参数、原始输出及退出码均记录；检查器的临时步骤文件在退出后清理。不生成自动快照、时间戳归档或 history。已有 `results/` 原始日志和历史报告保持原样，当前动作不写入。

详细边界见 [记录布局](../../../RECORD_LAYOUT.md)。
