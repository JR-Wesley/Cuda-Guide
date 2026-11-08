# Hello

A minimal program using cutlass.
Make sure to include the right path to cutlass in cmake file.

## 本地编译入口

原构建入口已迁移为 [commands/compile/run.sh](commands/compile/run.sh)，只编译、不运行。虽然输入只有 C++ 源码，现有 CMake 仍启用 CUDA 并含固定架构参数；仅在环境适用时执行 `CUDA_ARCH=sm_89 bash commands/compile/run.sh`。日志是 `commands/compile/output.log`，产物位于 `build/sm_89/Release/`（Debug 独立目录）。参数、既有 CMake 问题和仅静态验证的边界见 [NCU 导航](../../README.md)。

