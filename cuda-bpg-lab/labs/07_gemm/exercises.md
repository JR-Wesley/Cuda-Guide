# Stage 7 综合练习

先独立完成，再阅读[答案](answers.md)。每题说明语义、源码模型、静态证据、动态指标分别来自哪里。

1. 写出Naive/Coalesced某个lane的A/B/C字节地址。M=N=K=256时，分别计算T16与T32首warp在k=0覆盖的32 B sector数；同址重复是否应乘lane数？
2. 512³、T16、R4下，计算direct/shared/register三类global源级请求字节、唯一数据量和有用FLOP。为什么requested GB/s最高的版本不一定最快？
3. 推导完整tile下Register Tiled近似AI=`R*T/[2(R+1)]`，并计算R→∞时的上限。它为什么不是随R无界增长？
4. M37、N65、K33、T16、R4：计算grid、有效global加载数、device分配、有用FMA和padding模型FMA；说明多余工作发生在哪些维度。
5. 为什么shared kernel不能在row≥M或col≥N时提前return？第二次barrier能不能删除？给出具体覆盖风险。
6. 对T16/T32分别推导shared计算阶段As和Bs的bank映射；为什么本布局不自动需要transpose中的+1 padding？SASS宽访存会带来什么核对要求？
7. 比较T16/R4与T32/R4的threads/block、shared/block、output tile和grid数量变化；为什么仅比较theoretical occupancy不足以选择tile？
8. K4096时求gamma_K。对某输出S=1000、FP64参考接近0，计算本实验允许的绝对误差。若最大相对误差很大，是否必须失败？怎样发现容差过宽掩盖索引错误？
9. 编写记录表：先预测register→unrolled是否更快，再检查SASS loop、代码长度、register/local及理论驻留量，最后做A/B/B/A；不能只报告最小耗时。
10. 选择一个小矩阵、一个矩形和一个非整tile矩阵，运行五版并报告Min/Median/P95、GFLOP/s、host E2E、误差指标。若某一步变慢，提出两个竞争解释并设计一次最小区分实验。
11. 用NSYS核对一个op一个kernel及实际grid/block，比较活跃kernel与Event span。说明没有NCU时不能确认的机制，给出以后可补充的指标类别。
12. 完成自己的优化报告：Theory→Model→Hypothesis→Experiment→Analysis→Conclusion，保留基线、源码/二进制哈希、命令与失败，不将本项目实测当成自己的环境结果。
