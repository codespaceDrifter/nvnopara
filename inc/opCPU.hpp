#ifndef OP_CPU_HPP
#define OP_CPU_HPP

#include "tensor.hpp"
#include <vector>
#include <cmath>

void CPU_add(int n, float* a, float* b, float* result);

void CPU_sub(int n, float* a, float* b, float* result);

void CPU_mul(int n, float* a, float* b, float* result);

void CPU_div(int n, float* a, float* b, float* result);

void CPU_pow(int n, float* a, float* b, float* result);

void CPU_equal(int n, float* a, float* b, float* result);

void CPU_lessThan(int n, float* a, float* b, float* result);

void CPU_greaterThan(int n, float* a, float* b, float* result);

void CPU_matmul(int m, int n, int k, float* a, float* b, float* result);

void CPU_sin(int n, float* a, float* result);

void CPU_cos(int n, float* a, float* result);

void CPU_sum(Tensor* a,  Tensor* result);

void CPU_max(Tensor* a, Tensor* result);

void CPU_min(Tensor* a, Tensor* result);

#endif // OP_CPU_HPP