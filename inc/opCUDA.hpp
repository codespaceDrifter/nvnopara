#ifndef OP_CUDA_HPP
#define OP_CUDA_HPP

#include "tensor.hpp"
#include <vector>
#include <cuda_runtime.h>

void CUDA_add(Tensor* a, Tensor* b, Tensor* result);

void CUDA_sub(Tensor* a, Tensor* b, Tensor* result);

void CUDA_mul(Tensor* a, Tensor* b, Tensor* result);

void CUDA_div(Tensor* a, Tensor* b, Tensor* result);

void CUDA_pow(Tensor* a, Tensor* b, Tensor* result);

void CUDA_equal(Tensor* a, Tensor* b, Tensor* result);

void CUDA_lessThan(Tensor* a, Tensor* b, Tensor* result);

void CUDA_greaterThan(Tensor* a, Tensor* b, Tensor* result);

void CUDA_matmul(int m, int n, int k, float* a, float* b, float* result);

void CUDA_sin(Tensor* a, Tensor* result);

void CUDA_cos(Tensor* a, Tensor* result);

void CUDA_sum(Tensor* a, Tensor* result);

void CUDA_max(Tensor* a, Tensor* result);

void CUDA_min(Tensor* a, Tensor* result);

#endif // OP_CUDA_HPP