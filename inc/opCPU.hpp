#ifndef OP_CPU_HPP
#define OP_CPU_HPP

#include "tensor.hpp"
#include <vector>
#include <cmath>

void CPU_add(float* a, float* b, int n, float* result);

void CPU_sub(float* a, float* b, int n, float* result);

void CPU_mul(float* a, float* b, int n, float* result);

void CPU_div(float* a, float* b, int n, float* result);

void CPU_pow(float* a, float* b, int n, float* result);

void CPU_equal(float* a, float* b, int n, float* result);

void CPU_lessThan(float* a, float* b, int n, float* result);

void CPU_greaterThan(float* a, float* b, int n, float* result);

void CPU_matmul(float* a, float* b, int m, int n, int k, float* result);

void CPU_sin(float* a, int n, float* result);

void CPU_cos(float* a, int n, float* result);

void CPU_sum(Tensor* a,  Tensor* result);

void CPU_max(Tensor* a, Tensor* result);

void CPU_min(Tensor* a, Tensor* result);

#endif // OP_CPU_HPP