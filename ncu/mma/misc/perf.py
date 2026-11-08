"""
GEMM Performance Analysis Script
Analyzes theoretical performance bounds for C = alpha*A@B + beta*C
Matrix size: 4096 x 4096 (float32)
"""

import math

# -----------------------------
# 参数定义
# -----------------------------
M = N = K = 4096
dtype_size = 4  # float32 = 4 bytes

# -----------------------------
# 1. 计算总 FLOPs
# -----------------------------
# GEMM: C = alpha * A@B + beta * C
# A@B: 每个 C[i,j] 需要 K 次乘加 (2 FLOPs)，共 M*N*K 次乘加 → 2*M*N*K FLOPs
# 逐元素加法: M*N 次乘加（beta * C[i,j]）
flops_gemm = (2 * K - 1 ) * M * N
flops_update = M * N
total_flops = flops_gemm + flops_update
total_gflops = total_flops / 1e9

print("=== 性能分析：4096² 矩阵 GEMM ===")
print(f"矩阵大小: {M} x {N} (float32)")
print(f"FLOPs (A@B): 2 * {M}^3 = {flops_gemm / 1e9:.2f} GFLOPs")
print(f"FLOPs (beta*C + ...): {flops_update / 1e6:.2f} MFLOPs")
print(f"总计 FLOPs: {total_gflops:.2f} GFLOPs\n")

# -----------------------------
# 2. 最小内存访问量（理想情况，完美缓存）
# -----------------------------
# 读取: A(M×K), B(K×N), C(M×N) → 3 * M*N*4 bytes
# 写入: C(M×N) → 1 * M*N*4 bytes
min_data_read_bytes = (M * K + N * K + M * N) * dtype_size
min_data_write_bytes = 1 * M * N * dtype_size
total_min_memory_bytes = min_data_read_bytes + min_data_write_bytes

min_data_read_mb = min_data_read_bytes / 1e6
min_data_write_mb = min_data_write_bytes / 1e6
total_min_memory_mb = total_min_memory_bytes / 1e6

print("=== 最小内存访问（理论下限）===")
print(f"读取 (A,B,C): {min_data_read_mb:.0f} MB")
print(f"写入 (C): {min_data_write_mb:.0f} MB")
print(f"总计最小内存传输: {total_min_memory_mb:.0f} MB\n")

# -----------------------------
# 3. 理论执行时间
# -----------------------------
# 计算时间 (基于 FP32 吞吐)
# GPU 参数 NVIDIA GeForce RTX 4060 Laptop GPU
# Max FLOPS (FP32) = CUDA Cores × 2 × Boost Clock (GHz)
print("=== 硬件参数 ===")
compute_tflops_fp32_compute = 3072 * 2 * 2250 * 1e6 / 1e12
compute_tflops_fp32 = 11.7      # TFLOPs/s
print(f"FLOPS (FP32)计算值: {compute_tflops_fp32_compute:.2f} TFLOPs/s")
print(f"FLOPS (FP32): {compute_tflops_fp32:.2f} TFLOPs/s")
# 带宽 (GB/s) = (显存频率 Gbps × 显存位宽 × 2) / 8 (DDR)
memory_bw_gbps_compute = 8001 * 1e6 * 128 * 2/ 8 / 1e9 # GB/s
memory_bw_gbps = 255.7          # GB/s HBM
print(f"显存带宽计算值: {memory_bw_gbps_compute:.2f} GB/s")
print(f"显存带宽: {memory_bw_gbps:.2f} GB/s (HBM)\n")
tensor_core_tflops = 309.0      # TFLOPs/s TODO：确认具体数值

compute_time_ms = (total_flops / (compute_tflops_fp32 * 1e12)) * 1000  # 秒 → 毫秒

# 内存传输时间
memory_transfer_time_ms = (total_min_memory_bytes / (memory_bw_gbps * 1e9)) * 1000

print("=== 理论执行时间（若达到峰值）===")
print(f"计算时间: {compute_time_ms:.2f} ms (基于 {compute_tflops_fp32} TFLOPs/s)")
print(f"内存传输时间: {memory_transfer_time_ms:.2f} ms (基于 {memory_bw_gbps} GB/s)\n")

# -----------------------------
# 4. 判断瓶颈（Roofline 模型思想）
# -----------------------------
if compute_time_ms > memory_transfer_time_ms:
    bottleneck = "compute-bound"
    ratio = compute_time_ms / memory_transfer_time_ms
    print(f"- 计算时间是内存的 {ratio:.1f}x → 核心将是 **compute-bound**")
    print(f"- 只要实际内存访问量 < 10x 最小值，仍可保持 compute-bound")
else:
    bottleneck = "memory-bound"
    ratio = memory_transfer_time_ms / compute_time_ms
    print(f"❌ 内存时间是计算的 {ratio:.1f}x → 核心将是 **memory-bound**")

print()

# -----------------------------
# 5. Tensor Core 性能对比
# -----------------------------
tc_compute_time_ms = (total_flops / (tensor_core_tflops * 1e12)) * 1000
print("=== 使用 Tensor Core 时（假设支持 fp32）===")
print(f"计算时间: {tc_compute_time_ms:.2f} ms (基于 {tensor_core_tflops} TFLOPs/s)")
if tc_compute_time_ms < memory_transfer_time_ms:
    print("⚠️  使用 Tensor Core 后，计算极快，几乎一定是 **memory-bound**")
else:
    print("✅ 仍可能是 compute-bound，但差距缩小")
print()

# -----------------------------
# 6. Naive Kernel 内存流量（无缓存，最坏情况）
# -----------------------------
# 每个线程独立加载 A 的行和 B 的列
# 每个线程加载：K (A 的行) + K (B 的列) + 1 (C 的元素) = 2*K + 1 个 float
# 总线程数：M*N
per_thread_loads = 2 * K + 1
total_naive_memory_ops = M * N * per_thread_loads
total_naive_bytes = total_naive_memory_ops * dtype_size
total_naive_gb = total_naive_bytes / 1e9

# $4092^2 * 4092*2 * 4 \approx 548 * k^3 Bytes$

print("=== Naive Kernel 内存流量（无共享，最坏情况）===")
print(f"每个线程加载: {per_thread_loads} 个 float")
print(f"总内存操作数: {total_naive_memory_ops / 1e9:.2f} billion")
print(f"总内存流量: {total_naive_gb:.1f} GB")
print(f"（对比最小理论值 {total_min_memory_mb:.0f} MB → 高出约 {total_naive_gb * 1000 / total_min_memory_mb:.1f}x）")

# -----------------------------
# 7. 当前实现性能对比
# ---------------------
naive_time = 254.74 # 254 ms
naive_gflops = total_gflops / (naive_time / 1000)  # 实测 300 GFLOPs
naive_efficiency = (naive_gflops / compute_tflops_fp32) / 1e3  # TF → GF
print("\n=== 当前实现性能 ===")
print(f"实测时间: {naive_time:.2f} ms")
print(f"实测性能: {naive_gflops:.2f} GFLOPs/s")
print(f"理论峰值利用率: {naive_efficiency:.1%}")

# sharemem 49152 B
shmem_time = 164.42 # 164 ms
shmem_gflops = total_gflops / (shmem_time / 1000)  # 实测 300 GFLOPs
shmem_efficiency = (shmem_gflops / compute_tflops_fp32) / 1e3  # TF → GF
print("\n=== Shared Memory Cache-Blocking 性能 ===")
print(f"实测时间: {shmem_time:.2f} ms")
print(f"实测性能: {shmem_gflops:.2f} GFLOPs/s")
print(f"理论峰值利用率: {shmem_efficiency:.1%}")
print(f"Imporved: {shmem_efficiency / naive_efficiency:.1%}")