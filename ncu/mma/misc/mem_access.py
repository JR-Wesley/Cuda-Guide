"""
GEMM Memory Access Pattern Analysis 
"""

M = N = K = 1024
blockDim_x = 32
blockDim_y = 32

# Focus on block (0, 0, 0)
blockIdx_x = 0
blockIdx_y = 0

print("Simulating sgemm_naive: 2D block (32x32), warp 0 accesses")
print("="*90)
print(f"{'Thread':<10} {'x':<4} {'y':<4} {'C index':<8} {'A row'} {'B col'} "
      f"{'First 5 A accesses'} {'| First 5 B accesses'}")
print("-"*90)

blockDim_x = 32
blockDim_y = 32

for tx in range(4):  # threadIdx.x = 0,1,2,3
    ty = 0           # threadIdx.y = 0 → all threads in same column

    x = blockIdx_x * blockDim_x + tx   # row in C
    y = blockIdx_y * blockDim_y + ty   # col in C → always 0

    c_idx = x * N + y                  # memory location in C

    # first few accesses to A[x][i] and B[i][y]
    a_indices = [f"A[{x*K + i}]" for i in range(5)]
    b_indices = [f"B[{i*N + y}]" for i in range(5)]

    print(f"T({tx},{ty})     {x:<4} {y:<4} C[{c_idx:<8}] A[{x}]  B[{y}]  "
          f"{','.join(a_indices[:3])}... | {','.join(b_indices[:3])}...")
print("threadIdx.x相邻的线程是同一warp")
print("对于任一线程：访问A是连续的，B是不连续的")
print("warp内不同线程：访问A是不连续的、无法合并，B是相同的、同一warp内广播")

print("\n\nSimulating sgemm_coalescing: 1D block (1024), warp 0 accesses")
print("="*90)
print(f"{'Thread':<10} {'x':<4} {'y':<4} {'C index':<8} {'A row'} {'B col'} "
      f"{'First 5 A accesses'} {'| First 5 B accesses'}")
print("-"*90)

# Now blockDim is 1D: 1024 threads per block
# We simulate threadIdx.x from 0 to 3 (first 4 threads of warp 0)

BLOCKSIZE = 32
for t_id in range(4):  # threadIdx.x = 0,1,2,3
    # Decode into 2D coordinate within the tile
    x = blockIdx_x * BLOCKSIZE + (t_id // BLOCKSIZE)  # t_id//32 = 0
    y = blockIdx_y * BLOCKSIZE + (t_id %  BLOCKSIZE)  # t_id%32 = 0,1,2,3

    c_idx = x * N + y

    a_indices = [f"A[{x*K + i}]" for i in range(5)]
    b_indices = [f"B[{i*N + y}]" for i in range(5)]

    print(f"T({t_id})      {x:<4} {y:<4} C[{c_idx:<8}] A[{x}]  B[{y}]  "
          f"{','.join(a_indices[:3])}... | {','.join(b_indices[:3])}...")