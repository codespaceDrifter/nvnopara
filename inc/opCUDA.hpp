#ifndef OP_CUDA_HPP
#define OP_CUDA_HPP

#include "tensor.hpp"
#include <vector>
#include <cuda_runtime.h>

void CUDA_add(float* a, float* b, int n, float* result);

void CUDA_sub(float* a, float* b, int n, float* result);

void CUDA_mul(float* a, float* b, int n, float* result);

void CUDA_div(float* a, float* b, int n, float* result);

void CUDA_pow(float* a, float* b, int n, float* result);

void CUDA_equal(float* a, float* b, int n, float* result);

void CUDA_lessThan(float* a, float* b, int n, float* result);

void CUDA_greaterThan(float* a, float* b, int n, float* result);

void CUDA_matmul(float* a, float* b, int m, int n, int k, float* result);

void CUDA_sin(float* a, int n, float* result);

void CUDA_cos(float* a, int n, float* result);

void CUDA_sum(Tensor* a, Tensor* result);

void CUDA_max(Tensor* a, Tensor* result);

void CUDA_min(Tensor* a, Tensor* result);

#endif // OP_CUDA_HPP