# 工具、功能与当前证据索引

先读 [Toolkit 导览](TOOLKIT_GUIDE.md) 建立程序生命周期，再按问题选动作。**每个动作 README 就是该功能的教学文档和当前日志讲解**；本页只索引证据，不另造一份聚合日志或重复报告。

下面路径相对本项目。除环境 `check.sh` 外，各动作在自己的目录执行 `CUDA_ARCH=sm_89 bash run.sh`；报告读取 `ncu-report`、`nsys-stats` 只需 `bash run.sh`。架构须与实际构建条件对应，不把本机 `sm_89` 当作通用默认。

## 1. 从源码到一个能校验的程序

| 问题/动作 | 当前证据与解释 |
| --- | --- |
| [环境查询](experiments/environment/README.md) | [status.tsv](experiments/environment/log/status.tsv)：十项查询退出 0；工具版本、GPU、驱动分别记录。 |
| [共享程序构建](experiments/build/README.md) | [build.log](experiments/build/log/build.log)：设备资源信息；44,984 字节的动态链接程序。 |
| [普通运行](experiments/execution/README.md) | [vector.log](experiments/execution/log/vector.log)：vector/257/1，结果正确；4.096 us 是本次 event 区间值，不是稳定峰值性能。 |

## 2. 编译与链接各自回答什么

完整学习顺序及依赖见 [compile 课程导航](experiments/compile/README.md)。以下都已完成离线动作，不代表 GPU 运行。

| 单功能动作 | 读哪一类信息 |
| --- | --- |
| [compile-dryrun](experiments/compile-dryrun/README.md) | 拟执行的内部编译链；不生成大型中间文件。 |
| [compile-ptx](experiments/compile-ptx/README.md) | `.ptx` 的入口、虚拟寄存器与版本；本次 PTX 9.4。 |
| [compile-cubin](experiments/compile-cubin/README.md) | 原生 `sm_89` 镜像与资源诊断，保留重定位供 CFG 使用。 |
| [compile-fatbin](experiments/compile-fatbin/README.md) | 同一输入的设备镜像容器。 |
| [compile-object](experiments/compile-object/README.md) | 主机目标文件，尚未成为完整主机程序。 |
| [compile-host-link](experiments/compile-host-link/README.md) | 只链接已有 object，动态依赖 cudart。 |
| [compile-rdc-compile](experiments/compile-rdc-compile/README.md) | 两份源码的可重定位设备代码；不做链接。 |
| [compile-rdc-device-link](experiments/compile-rdc-device-link/README.md) | 设备符号解析。 |
| [compile-rdc-host-link](experiments/compile-rdc-host-link/README.md) | CPU 入口及已完成设备链接的主机链接。 |
| [compile-lto-compile](experiments/compile-lto-compile/README.md) | LTO 设备中间表示；首次不合法参数的错误保留为明确命名记录。 |
| [compile-lto-device-link](experiments/compile-lto-device-link/README.md) | 设备链接时优化；成功不保证内联或提速。 |
| [compile-lto-host-link](experiments/compile-lto-host-link/README.md) | 最终小型 LTO 主机程序；本轮未执行它的 GPU 计算。 |

每个动作的命令、原始输出、状态均在自己的 `log/`。共享产物位于 `build/sm_89/compile/artifacts/`；不再保留 `keep/` 和大型预处理输出。

## 3. 二进制与反汇编

| 单功能动作 | 当前观测及边界 |
| --- | --- |
| [binary-list](experiments/binary-list/README.md) | 列出 fatbin 的 cubin/PTX；列表不说明驱动选择。 |
| [binary-ptx](experiments/binary-ptx/README.md) | 导出内嵌 PTX 文本，含工具包装信息，不冒充独立提取文件。 |
| [binary-sass](experiments/binary-sass/README.md) | 原生指令；不是运行轨迹或性能。 |
| [binary-resources](experiments/binary-resources/README.md) | 小 probe 的静态 REG:10 等资源；不是 occupancy 实测。 |
| [binary-elf](experiments/binary-elf/README.md) | 设备 ELF 与符号，不与主机 ELF 混用。 |
| [nvdisasm-sass](experiments/nvdisasm-sass/README.md) | SASS 与源码行对应；优化后不是逐行执行。 |
| [nvdisasm-cfg](experiments/nvdisasm-cfg/README.md) | 987 字节 DOT，工具成功；Graphviz 未安装，未做渲染或其解析验证。 |
| [nvdisasm-live-ranges](experiments/nvdisasm-live-ranges/README.md) | 寄存器值的活跃范围，不等于动态耗时或占用率。 |

这八个动作已经实现并执行，不再是“nvdisasm 只有规划”。`cuobjdump` 镜像提取仍未接入，不为了生成 nvdisasm 输入恢复提取；直接使用编译器产生的独立 cubin。

## 4. 加载和正确性

| 动作 | 当前结果 |
| --- | --- |
| [module-build](experiments/module-build/README.md) / [module-loading](experiments/module-loading/README.md) | 分别只构建、只运行 Runtime，64 元素校验正确。 |
| [module-driver-build](experiments/module-driver-build/README.md) | 只构建 Driver 主机入口；stub 只用于链接。 |
| [module-cubin-build](experiments/module-cubin-build/README.md) / [module-driver-cubin](experiments/module-driver-cubin/README.md) | 分别只构建设备镜像、只加载运行；正确。 |
| [module-ptx-build](experiments/module-ptx-build/README.md) / [module-driver-ptx](experiments/module-driver-ptx/README.md) | PTX 编译成功，但加载失败：`.version 9.4` 超过驱动支持的 9.2；保留 API 222、程序退出 1。 |
| [validation](experiments/validation/README.md) | tiled/65/1 的正常 memcheck：0 errors，结果正确。 |
| [validation-racecheck](experiments/validation-racecheck/README.md) | 同输入正常 racecheck：0 hazards/0 errors/0 warnings。 |
| [sanitizer-build](experiments/sanitizer-build/README.md) / [sanitizer-static](experiments/sanitizer-static/README.md) | 分开故障调试构建与 SASS 检查，共用一份源码和二进制。 |
| [sanitizer-memcheck](experiments/sanitizer-memcheck/README.md) | 故意越界：原始退出 86，2 errors，诊断断言成功。 |
| [sanitizer-racecheck](experiments/sanitizer-racecheck/README.md) | 故意共享内存竞争：原始退出 86，1 error，诊断断言成功。 |
| [sanitizer-initcheck](experiments/sanitizer-initcheck/README.md) | 故意未初始化读取：原始退出 86，1 error，诊断断言成功。 |
| [sanitizer-synccheck](experiments/sanitizer-synccheck/README.md) | 故意同步错误：原始退出 86，17 errors，诊断断言成功。 |

故障实验的“成功”指检查器识别了指定错误，绝不是程序无错；只凭非零退出码不能通过验收。

## 5. 采集与读报告分开

| 动作 | 当前证据与阅读目标 |
| --- | --- |
| [ncu](experiments/ncu/README.md) | 基本采集成功，8 passes；[profile.log](experiments/ncu/log/profile.log) 与 `.ncu-repz`。 |
| [ncu-memory](experiments/ncu-memory/README.md) | 内存表格成功，32 passes；[profile.log](experiments/ncu-memory/log/profile.log)，解释 99 sectors 和尾部不满。 |
| [ncu-report](experiments/ncu-report/README.md) | 只导入基本报告；[details.log](experiments/ncu-report/log/details.log)，核对 2 blocks、256 threads/block、512 threads。 |
| [nsys](experiments/nsys/README.md) | 只采集时间线；[profile.log](experiments/nsys/log/profile.log) 与 `.nsys-rep`。 |
| [nsys-stats](experiments/nsys-stats/README.md) | 只统计已有报告，SQLite 留本动作；[stats.log](experiments/nsys-stats/log/stats.log)，2 次 kernel 与 API 区间分别读。 |

小输入、有限 section、单个 ncu 采集对象，不生成大应用或全量报告。分析器运行中的 event 时间受到插桩和重放影响，不能替代普通基线。报告更新后旧的导入/统计解释也要显式刷新。

## 6. 未执行范围与记录原则

Graphviz 渲染未做；NVRTC/nvJitLink、cuda-gdb、NVTX、cu++filt/nvprune 仍是 [总体导览](TOOLKIT_GUIDE.md) 中的后续方向，不造日志。所有当前结论区分官方语义、静态产物、实际运行与未知。

本轮已清理原环境记录并重采；被替代入口已删除，没有建立日期目录或 history。首次 LTO 参数拒绝、ncu 后缀判断失败属于本轮有用诊断证据，明确命名保留而不覆盖改写。当前结果以每动作 `log/status.tsv` 和对应原始输出为准。
