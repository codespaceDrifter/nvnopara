#ifndef OP_CUDA_HPP
#define OP_CUDA_HPP

#include "tensor.hpp"
#include <vector>
#include <cuda_runtime.h>

void CUDA_add(int n, float* a, float* b, float* result);

void CUDA_sub(int n, float* a, float* b, float* result);

void CUDA_mul(int n, float* a, float* b, float* result);

void CUDA_div(int n, float* a, float* b, float* result);

void CUDA_pow(int n, float* a, float* b, float* result);

void CUDA_equal(int n, float* a, float* b, float* result);

void CUDA_lessThan(int n, float* a, float* b, float* result);

void CUDA_greaterThan(int n, float* a, float* b, float* result);

void CUDA_matmul(int m, int n, int k, float* a, float* b, float* result);

void CUDA_sin(int n, float* a, float* result);

void CUDA_cos(int n, float* a, float* result);

void CUDA_sum(Tensor* a, Tensor* result);

void CUDA_max(Tensor* a, Tensor* result);

void CUDA_min(Tensor* a, Tensor* result);

#endif // OP_CUDA_HPP