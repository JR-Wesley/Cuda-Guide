// gemm.h
#pragma once
#include <cuda_runtime.h>
#include <torch/torch.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

void sgemm_naive(torch::Tensor A, torch::Tensor B, torch::Tensor C, float alpha,
                 float beta);
void sgemm_shmem(torch::Tensor A, torch::Tensor B, torch::Tensor C, float alpha,
                 float beta);
void sgemm_2dtiling(torch::Tensor A, torch::Tensor B, torch::Tensor C,
                    float alpha, float beta);
