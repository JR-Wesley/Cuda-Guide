M = 4096
N = 4096
K = 4096

grid_dim = 32

read_num = grid_dim * grid_dim *(M * K / grid_dim + N * K / grid_dim)
write_num = M * N

max_read_req = read_num / 32
max_write_req = write_num / 32

print(f"max_read_req: {max_read_req}")
print(f"max_write_req: {max_write_req}")

real_write_num = 335.54 * 1e6
real_read_num = 33.55 * 1e6
print(f"real_read_num: {real_read_num}")
print(f"real_write_num: {real_write_num}")

print(f"utilization: {max_read_req / real_read_num}")
print(f"utilization: {max_write_req / real_write_num}")

