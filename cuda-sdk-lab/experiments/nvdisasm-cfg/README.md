# CFG DOT：沿控制流读边界判断

## 背景与目标

CFG（control-flow graph，控制流图）用节点和箭头描述指令可能怎样前进。基本块是顺序执行的一段指令；遇到跳转或退出时，图会展示相应的连接关系。本节只生成基本块粒度的 DOT 文本，不生成图片，也不测量分支发生次数。

前置：先读 [SASS/源码行](../nvdisasm-sass/README.md)，再按 [编译专题](../compile/README.md) 准备 `probe.cubin`。编译侧需保留重定位信息（例如向 `ptxas` 传递 `--preserve-relocs`），供工具的数据流分析使用。这里不修改输入、不补编译，不用 `--no-dataflow` 隐藏分析失败。

## 1. 底层命令与两条输出流

在 `cuda-sdk-lab/experiments/nvdisasm-cfg/` 中：

```bash
mkdir -p log
nvdisasm --output-control-flow-graph-with-basic-blocks ../../build/sm_89/compile/artifacts/probe.cubin > log/cfg.dot 2> log/cfg.log
```

`--output-control-flow-graph-with-basic-blocks`（`-bbcfg`）为每个基本块生成节点；不同于 `--output-control-flow-graph`（`-cfg`）的 hyperblock 粒度，本节只选前者，避免同一目标重复输出。

`>` 保存 stdout（正常 DOT 输出），`2>` 保存 stderr（诊断信息）。**不能用 `2>&1` 合并到 DOT 文件**，否则一行警告也可能破坏图语法。DOT 是描述节点和边的文本语言；Graphviz 的 `dot` 才是把它画成图的另一个程序。

## 2. 独立记录脚本

```bash
CUDA_ARCH=sm_89 bash run.sh
```

默认输入为 `../../build/<CUDA_ARCH>/compile/artifacts/probe.cubin`，唯一可选参数为已有 cubin 文件。`CUDA_ARCH` 必填，`NVDISASM` 可指定工具路径。脚本不切目录、不隐式编译、不启动 GPU。

版本通过共享 `record` 保存到 `log/tool_version.*`。CFG 命令单独记录为 `log/cfg.sh`（包括两条重定向），stdout 在 `log/cfg.dot`，stderr 在 `log/cfg.log`，真实退出码在 `log/status.tsv`。脚本返回该工具退出码，不通过 `tee`、渲染器或管道掩盖失败。重跑覆盖同名文件；手动重放 `.sh` 不更新状态表。

## 3. 阅读与失败判断

先确认状态表中 `cfg` 为零，再打开 DOT 看 `digraph`、节点标签和 `->` 边。对照 SASS 中的条件退出/跳转，寻找 `i < n` 附近的路径。优化可能把源码条件变成谓词指令或提前退出；不要要求每个 `if` 都有两个漂亮的分支框。

DOT 非空不等于分析成功：失败时可能留下部分 stdout，必须同时读退出码和 `cfg.log`。发生数据流分析错误时，先确认输入确为当前显式构建且保留了重定位信息，再排查工具兼容性，不冒用旧图。

## 4. 练习

在 DOT 中挑一条边，找到它连接的两个块，再到 SASS 中定位对应的跳转或顺序前进位置。解释这条边为什么只表示“可能到达”，而不代表 GPU 在本次运行中实际走过它。

## 本轮验证与渲染状态

共享 cubin 生成后，已用 `CUDA_ARCH=sm_89 bash run.sh`、`nvdisasm` 13.4.92 生成 [cfg.dot](log/cfg.dot)，[状态](log/status.tsv) 中 `cfg` 退出码为 0。[stderr](log/cfg.log) 为 0 字节（实际为空，不是漏记），DOT 为 987 字节。图内有 `saxpy_probe`、`BB0` 两个基本块和一条前进边：首块末尾为 `@P0 EXIT`，未退出的路径进入包含访存和 `FFMA` 的 `BB0`；图中不必另有一个显式“退出节点”。只核对了工具输出和文本结构，未通过 Graphviz 解析或渲染验证。

本轮 `command -v dot` 未找到渲染器，**渲染未执行**。不安装 Graphviz，不提供自动渲染入口，不创建 SVG/PNG 或冒充成功的报告；读取 DOT 文本不依赖 `dot`。若以后具备渲染器，再另行安排独立渲染动作，不能把渲染失败当成反汇编失败。
