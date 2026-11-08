# Stage 2 练习

1. 对完整 warp，画 stride=1/2/3/8、offset=0 的地址和 sector 编号。gather 的写入会变吗？
2. 推导 stride=1、offset=1 的请求，说明为什么不能直接得出新增 DRAM bytes。
3. N=1,048,576、stride=8、offset=3、gather，算 requested bytes、input/output span 与 allocation bytes。
4. float4 N=7、block=1 启动多少线程？哪些线程复制 vector/tail？为什么不能只用 ceil((N/4)/block)？
5. 为什么 float4 每 warp 16 sectors 不代表比 scalar 的 4 sectors 更低效？
6. 做 small/large N 的 scalar/float2/float4 对照，观察何时收益消失或反转；保留负面结果。
7. 固定 N 扫 stride 时，哪些非 coalescing 因素也变化？提出后续对照逐项排除。
8. 分别写“模型预测”“静态证据”“运行事实”“仍未确认”。ncu 不可用时，哪些结论仍成立？

参考答案独立在 [answers.md](answers.md)。
