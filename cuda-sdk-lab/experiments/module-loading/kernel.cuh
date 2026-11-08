#pragma once

extern "C" __global__ void add_one(const int* input, int* output) {
    const int i = threadIdx.x;
    output[i] = input[i] + 1;
}
