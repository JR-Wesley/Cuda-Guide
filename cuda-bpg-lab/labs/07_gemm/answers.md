# Stage 7 参考答案

1. row-major地址是4(row*K+k)、4(k*N+col)、4(row*N+col)。T16 Naive的row=tx、col=ty，首warp有16个row、2个col：A/B/C sector16/1/16；Coalesced则2/2/4。T32时分别32/1/32和1/4/4。以地址集合计sector，同址lane不增加覆盖sector数；实际cache/DRAM行为不能从该集合唯一反推。
2. 有用FLOP=268,435,456；唯一数据3,145,728 B。direct请求1,074,790,400 B；shared请求68,157,440 B；register请求42,991,616 B。主性能比较分子2MNK相同的GFLOP/s；requested字节随算法复用变化，不能跨版把该GB/s当实际DRAM带宽。
3. 每个完整k-tile的FLOP=2R*T³，global输入字节4(R+1)T²，相除得到R*T/[2(R+1)]；R→∞时趋T/2。R增大减少重复B加载，但A输入需求也随输出行数增加，所以不是无限复用。
4. BM64/BN16，grid_x=5、grid_y=1。A加载37*33*5=6105、B加载33*65*1=2145，共8250 float；输出2405 float，requested42,620 B。device唯一数据4(1221+2145+2405)=23,084 B。有用FMA=79,365，源级padding FMA=5*1*64*16*48=245,760。M、N、K三方向都存在边缘工作，计算倍率约3.096，模型不代表硬件动态指令精确倍数。
5. 所有线程合作填shared并同步，越界输出线程仍可能负责有效输入的加载；提前return可能破坏barrier参与或漏填数据。计算后barrier防止快warp进入下一K tile覆盖慢warp仍在读的当前tile，不是多余的装饰。
6. bank=(row*T+column)%32。T32 As同址广播、Bs连续32bank；T16 As在同一warp的两半分别读ty0/1对应的两个bank地址，Bs读16地址并重复广播。它不是按列读32×32转置tile的32路同bank访问。若编译器生成多word指令，需按实际指令/子请求检查映射，不能只照源码标量表给出实测bank冲突结论。
7. T16/R4：256threads、shared5120 B、tile64×16；T32/R4：1024threads、shared20480 B、tile128×32。大整矩阵后者grid块数约少4倍，但每块资源更多，register分配、驻留block数、小grid不足和尾部浪费都可能抵消收益。理论occupancy只是一项静态约束。
8. Ku=4096/2^24=1/4096，gamma=1/4095≈0.0002442002442；阈值2*gamma*1000+1e-7≈0.4884005884。取消求和的相对误差可非常大而绝对误差仍符合界；单靠相对误差不判错。必须同时看实际最大abs/scaled error、random/identity和边界形状，不能只跑易相消输入；数学上界保守，不代表实际误差应该接近上界。
9. 没有预设速度排序。固定M/N/K/T/R/输入/编译和计时范围；两版都展开R循环，只有BK循环展开不同。区分内层BK回跳与外层K-tile回跳，代码列表含对齐padding不等于动态代码路径。把每次A/B/B/A全部样本和寄存器/资源变化记录下来；漂移大于差异时降低结论强度。
10. 合格答案必须实测和全量参考通过。可竞争解释：小问题launch成本；shared/barrier代价；grid不足；边缘padding；较多register降低驻留；更大tile减少global请求。每轮固定其他因素，例如同形状register/unrolled，或同T/R对整除与轻微尾部形状；后一种同时改变有用工作量，要用实际FLOP和padding模型解释。
11. common collect每variant kernel数为1+repeats*(warmup+iterations+1)，含初验、warmup、正式batch和E2E。按源码顺序核对再分别统计，不把所有活动的均值冒充正式batch。NSYS不能提供DRAM/cache命中、shared bank conflicts、SM发射/stall或achieved occupancy；NCU可用时按当前版本查询相关类别并独立保存，不替换普通计时。
12. 报告要求可由原始证据复算，并保留反例、未测项及适用范围。参考[实际分析](analysis.md)，但不得复制其数字充当其他设备的结果。
