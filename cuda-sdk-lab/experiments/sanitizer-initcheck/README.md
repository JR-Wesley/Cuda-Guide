# 只检查未初始化读取：initcheck

## 1. 从一个整数开始

`cudaMalloc` 只分配内存，不承诺把内容清零。共享 [cases.cu](../sanitizer/cases.cu) 的 `uninitialized_global_read` 用 1 个线程把未赋值的 `in[0]` 复制到 `out[0]`。地址虽在合法分配区内，读取值仍没有程序定义的初始来源；这是初始化错误，不是越界。

## 2. 底层命令与入口

先在 [sanitizer-build](../sanitizer-build/README.md) 显式构建。工作目录：`cuda-sdk-lab/experiments/sanitizer-initcheck/`。

```bash
LD_LIBRARY_PATH=/opt/cuda/lib64 compute-sanitizer --tool initcheck \
  --error-exitcode 86 ../../build/sm_89/sanitizer/cases initcheck
```

`--tool initcheck` 跟踪全局内存初始化状态；末尾参数选择共享程序里的故障分支。`--error-exitcode 86` 指定检查器检出错误的退出值，`LD_LIBRARY_PATH` 用于找到共享 Runtime。独立记录入口：

```bash
CUDA_ARCH=sm_89 LD_LIBRARY_PATH=/opt/cuda/lib64 bash run.sh
```

可指定 `COMPUTE_SANITIZER`。不切目录、不编译、不运行其他检查器；缺产物返回失败并提示先构建。

## 3. 真实结果

2026-11-08，RTX 4060 Laptop/CC 8.9、驱动 596.21、nvcc 13.4.92、Compute Sanitizer 2026.3.0.0。原始退出 **86**，报告 `Uninitialized __global__ memory read of size 4 bytes`，位置 `cases.cu:19`，汇总 **1 error**。见 [日志](log/initcheck.log)、[命令](log/initcheck.sh)、[状态](log/status.tsv)。

脚本另行记录退出码为 86、特定诊断存在、非零错误汇总三个断言，实测均为 0。脚本退出 0 只说明负例按预期检出，不能把原始错误改称正确执行。`log/` 重跑覆盖同名文件，不归档，不能把其他 API 失败当成本例诊断。

## 4. 练习

预测在启动前给输入整数赋值会怎样，解释为什么“这块内存碰巧全是零”也不等于已经初始化。若动手加初始化，须显式重编译再检查；负例脚本应因为诊断消失而失败。完成后恢复故障并重新采集，不继续引用已被覆盖的日志。
