# 静态资源：一个线程和线程块需要多少资源？

## 背景与目标

GPU 同时容纳的线程数会受寄存器、共享内存等资源限制。先读编译产物声明的资源，再讨论运行时占用率（实际驻留线程束的比例）；两者不是同一个指标。本节只读取小型 `probe.cubin` 的静态资源，不测性能。

前置：线程执行 kernel，线程块内可以共享片上内存；按 [编译专题](../compile/README.md) 显式生成 `probe.cubin`。

## 1. 底层命令

在 `cuda-sdk-lab/experiments/binary-resources/` 中：

```bash
cuobjdump --dump-resource-usage ../../build/sm_89/compile/artifacts/probe.cubin
```

`--dump-resource-usage` 打印设备 ELF 的函数资源记录。这里没有 kernel 启动，也没有硬件计数器采集。

## 2. 记录脚本

```bash
CUDA_ARCH=sm_89 bash run.sh
```

无参数读取 `../../build/<CUDA_ARCH>/compile/artifacts/probe.cubin`，唯一可选参数替换输入文件。`CUDA_ARCH` 必填，`CUOBJDUMP` 可指定工具路径。相对路径按本动作目录解释，脚本不切目录、不编译。

查看 `log/resources.log` 和 `log/tool_version.log`；命令在同名 `.sh`，真实退出码在 `log/status.tsv`。重跑覆盖同名文件，失败即停。输入缺失先按编译专题准备，不把缺输入记成成功或资源为零。

## 3. 解释字段

- `REG`：函数每线程分配的寄存器数，不是某条指令处仍然活跃的值数。
- `SHARED`：静态共享内存字节数；不能单凭这里推断启动时是否另请求动态共享内存。
- `STACK`、`LOCAL`：栈与本地存储相关的静态字节数，不等于测得的内存流量；不能把所有本地内存都当成寄存器溢出。
- 常量存储字段描述常量区域，不能与每线程寄存器数直接相加。

以本机实际出现的字段为准。资源数字本身不证明 occupancy、带宽瓶颈或速度；还缺启动配置、设备限制及运行测量。

## 4. 练习

记录 `saxpy_probe` 的 `REG`，再读 [寄存器活跃区间](../nvdisasm-live-ranges/README.md)。为什么某条指令处的活跃值数不必等于 `REG`？不要为让数字相等而修改源码或编译选项。

## 本轮验证

输入生成后，已用 `CUDA_ARCH=sm_89 bash run.sh`、`cuobjdump` 13.4.92 离线检查，见 [资源输出](log/resources.log) 与 [退出码 0](log/status.tsv)。`saxpy_probe` 本次为 `REG:10 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:376`；这些零来自实际工具输出，不是缺失指标的占位值。运行时 occupancy、流量和耗时未测量，因为本动作没有 GPU 运行。
