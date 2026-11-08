# Stage 3 练习

1. 对 1024×1024 transpose，计算每次 requested bytes 和两个 tile 的 shared 大小。哪个是理论字节，哪个能由编译产物核验？
2. 对 tile[32][32] 和 tile[32][33] 的列访问，列出前8个 lane 的 bank 编号。为什么广播例外不适用？
3. 证明 37×65 矩阵最后一个 tile 的有效 store 不会读取未初始化 shared 元素。把边缘线程提前 return 会发生什么？
4. T=65536、steps=256，W=16384和16777216时 requested bytes 是否相同？输入唯一字节、算术强度和 E2E 传输量如何变化？
5. 为什么这个工作集实验不是 pointer-chase latency benchmark？绘制它无法单独区分的两种瓶颈解释。
6. 为什么 dynamic_index 与 static_index 可以共享同一个 CPU reference？数值求和顺序有哪些不能随意改变的部分？
7. 如何区分 addressable local 数组与后端 register spill？至少用两层证据，不只看一个 localSizeBytes 数值。
8. 若寄存器限制使 theoretical occupancy 上界更高，但 kernel 更慢，应如何解释和进一步验证？
9. 做一个小矩阵与大矩阵的对照；不要将 tiled/padded 的性能排序预先写进结论。

答案在 [answers.md](answers.md)。
