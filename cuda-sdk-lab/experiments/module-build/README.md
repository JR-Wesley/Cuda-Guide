# 只构建 Runtime 程序

## 为什么单独构建

编译把源码变为文件，不会执行 GPU 计算。这个动作从共享的 `../module-loading/runtime.cu` 生成含 CPU 主程序和原生 GPU 镜像的 `runtime`，**不再打包构建 Driver、cubin、PTX**。完整的 64 元素示例与线程对应关系见 [加载专题](../module-loading/README.md)。

## 1. 先读底层命令

工作目录：`cuda-sdk-lab/experiments/module-build/`。本次设备对应 `sm_89`，换设备须先确认架构：

```bash
mkdir -p ../../build/sm_89/module-loading
nvcc -std=c++17 -O2 -cudart=shared \
  -gencode arch=compute_89,code=sm_89 \
  ../module-loading/runtime.cu -o ../../build/sm_89/module-loading/runtime
```

`-std=c++17` 选择语言标准；`-O2` 优化代码；`-cudart=shared` 动态链接 CUDA Runtime，避免把大静态库装入每个小程序。`-gencode` 明确只内嵌 `sm_89` 原生机器码，不包含 PTX 回退；`compute_89` 是编译的虚拟目标。`-o` 指定输出。

## 2. 用独立入口记录构建

```bash
CUDA_ARCH=sm_89 bash run.sh
```

可用 `NVCC` 指定编译器。脚本不切目录、只编译；编译前删除自身旧输出，失败后移除不完整输出并返回真实退出码，不让旧文件冒充新构建。源码变化后须手动重新执行。运行请另进 [module-loading](../module-loading/README.md)。

## 3. 本次观察、记录与练习

2026-11-08，nvcc 13.4.92、`sm_89` 构建退出 0，见 [状态](log/status.tsv)、[实际命令](log/build.sh)、[原始输出](log/build.log)。构建无输出是正常情况，不等于 GPU 已运行。

共享记录接口只写当前动作的 `log/`；重跑覆盖本动作同名文件，不归档、不构建其他镜像。练习：不运行程序，解释为什么即使 `image.ptx` 不存在也能构建 Runtime；再到独立运行动作验证你的预测。
