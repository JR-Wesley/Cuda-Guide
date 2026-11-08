# 4. 只生成 fatbin

## 背景与目标

一个程序可以携带多种设备代码。fatbin 是设备镜像容器：这里明确请求 `sm_89` 的机器码与 `compute_89` 的 PTX。加载器之后选择哪一种，是运行阶段的问题。

## 底层命令与解释

工作目录：`cuda-sdk-lab/experiments/compile-fatbin/`。

```bash
mkdir -p ../../build/sm_89/compile/artifacts
nvcc -std=c++17 -O3 -lineinfo \
  -gencode arch=compute_89,code=sm_89 \
  -gencode arch=compute_89,code=compute_89 \
  --fatbin ../compile/compiler_probe.cu \
  -o ../../build/sm_89/compile/artifacts/probe.fatbin
```

`--fatbin` 停在设备镜像打包阶段；两条 `-gencode` 分别指定机器码和虚拟指令输出。`-std=c++17 -O3 -lineinfo` 保持语言、优化和行信息一致。本命令从源码重新编译，并不是把前两个动作的文件拼接起来。

## 独立脚本

```bash
CUDA_ARCH=sm_89 bash run.sh
```

日志为 `log/fatbin.sh`、`log/fatbin.log` 和 `log/status.tsv`；命令限时 `timeout 60s`，超时为 124。只覆盖自身日志与 `probe.fatbin`，失败删除部分输出。源码不复制，不启动 GPU。

## 当前日志的实际观察

Toolkit 13.4.92：`fatbin` 退出 **0**，诊断 [fatbin.log](log/fatbin.log) 为空；[probe.fatbin](../../build/sm_89/compile/artifacts/probe.fatbin) 为 **5,816 字节**。命令证明请求了两种镜像，成功状态证明生成了容器；本动作没有执行镜像列表检查。

后续独立静态检查可用 `cuobjdump --list-elf`、`cuobjdump --list-ptx` 读取此文件。这是下一动作的观察方法，不是假称本日志已经列出了镜像。

## 边界与练习

先预测只删除 `code=compute_89` 那一行会丢掉哪种兼容路径，再在手动命令中验证。不要仅凭文件大小猜镜像数：容器元数据、压缩与行信息都会影响大小。

PTX 的存在不保证当前驱动支持该 PTX 版本。下一步：[主机目标文件](../compile-object/README.md)。
