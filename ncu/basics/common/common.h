#pragma once
#include <chrono>
#include <sys/time.h>

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
    }                                                                          \
}

// 高精度计时器，返回当前时间（秒）
inline double seconds() {
    // 使用steady_clock确保单调递增的时钟（不受系统时间调整影响）
    auto now = std::chrono::steady_clock::now();
    // 转换为微秒
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        now.time_since_epoch()).count();
    // 返回秒级时间
    return static_cast<double>(duration) * 1e-6;
}
