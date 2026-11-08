# Stage 6 练习

先阅读[指南](README.md)，独立完成后再打开[答案](answers.md)。回答中区分语义证明、模型、静态证据、普通计时和插桩时间线。

1. N=1,048,576、chunk=65,536、buffers=3、rounds=128：计算chunk数量、host双向传输字节、kernel requested字节、device分配、FLOP。为什么不能把前两类字节相加称为PCIe带宽？
2. 画出buffers=2、streams=3时前7个chunk的slot/worker映射，指出每次复用等待哪个chunk。若删除slot_done等待，哪些覆盖会破坏结果？
3. 证明coordinator上的timing stop覆盖所有worker；指出删除join后得到的时间为什么无效。重复记录gate为什么不会让上一个op的wait转而等待下一个op？
4. N=1,048,593、chunk=65,536时最后一个chunk有多少元素？输入/输出各传多少字节、block256要发多少block？测试仅用N为chunk整数倍会漏掉什么？
5. 假设16个chunk，各H=20 us、C=35 us、D=25 us。计算串行和理想流水时间及最大模型加速；至少列出四个现实偏差来源。
6. 固定N/rounds/chunk/buffers，扫描streams=1/2/3/4。普通时间接近但NSYS有copy-compute交集，应该如何表述？为下一轮选择一个变量并预先写判定条件。
7. H2D/D2H分别比较小/大copy的pageable和pinned。为什么Event span、单次host E2E、memcpy API持续时间和device copy持续时间不一定相同？可否把它们简单相减估算pin成本？
8. Kernel区间[0,10)、[5,15)，copy区间[2,7)、[12,20)（us）：求kernel活跃时间和、并集、kernel-kernel并发长度、copy-compute交集。哪些量可证明重叠，哪些不能直接等同加速？
9. N=4099、jobs=3时写出每个job的范围与block256 grid。为什么serial也必须发3个kernel？将N增加到65,536时，单job资源需求如何改变？
10. 完成一轮可复核干预：先记录预测，运行正确性检查，采集普通CSV，再取NSYS时间线检查数量/字节/依赖/交集，最后写出被支持、被反驳和仍无法区分的解释。保留不符合预期的样本，说明pin/unpin及CPU参考是否计时。
