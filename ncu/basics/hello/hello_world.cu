#include <stdio.h>

__global__ void helloWorld() {
    printf("Hello World from thread %d in block %d!\n", threadIdx.x, blockIdx.x);
}

int main() {
    helloWorld<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}