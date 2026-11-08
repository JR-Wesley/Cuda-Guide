# 只读取故障程序的 SASS

## 背景

源码写了某个操作，不代表最终机器指令一定保留。SASS 是 GPU 机器码的可读反汇编文本；先确认产物里有什么，再用检查器观察实际执行，两者证据不同。本动作不运行 GPU、不编译。

## 1. 准备与底层命令

先在 [sanitizer-build](../sanitizer-build/README.md) 显式构建。工作目录：`cuda-sdk-lab/experiments/sanitizer-static/`。

```bash
cuobjdump --dump-sass ../../build/sm_89/sanitizer/cases
```

`cuobjdump` 读取 CUDA 二进制；`--dump-sass` 输出设备机器指令而非 PTX。输入是同一个共享 debug 程序，不另建静态检查专用产物。

## 2. 记录入口

```bash
CUDA_ARCH=sm_89 bash run.sh
```

可用 `CUOBJDUMP` 指定工具。脚本不切目录、缺产物提示先编译。输出直接记录为 [log/sass.log](log/sass.log)，命令为 [log/sass.sh](log/sass.sh)，真实退出码在 [log/status.tsv](log/status.tsv)。重跑覆盖同名日志，无自动归档。

## 3. 本次观察与练习

2026-11-08，`sm_89` 的共享 debug 产物导出退出 0，输出标明 `has debug info` 和 `arch = sm_89`。这证明反汇编成功，不证明四种错误已检出；动态结果见 [故障专题](../sanitizer/README.md)。

在 `sass.log` 找到 `bad_global_write`、`shared_write_race` 的函数段，识别全局写和 shared 写，再对照 `cases.cu` 的地址关系。练习：为什么即使看到 shared 写指令，仍需运行 racecheck 才能报告线程间的冲突？不要由静态指令推断运行时间或同步正确性。
