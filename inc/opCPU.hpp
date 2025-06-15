#ifndef OP_CPU_HPP
#define OP_CPU_HPP

#include "tensor.hpp"
#include <vector>

void CPU_add(Tensor* a, Tensor* b, Tensor* result);

void CPU_sub(Tensor* a, Tensor* b, Tensor* result);

void CPU_mul(Tensor* a, Tensor* b, Tensor* result);

void CPU_div(Tensor* a, Tensor* b, Tensor* result);

void CPU_pow(Tensor* a, Tensor* b, Tensor* result);

void CPU_equal(Tensor* a, Tensor* b, Tensor* result);

void CPU_lessThan(Tensor* a, Tensor* b, Tensor* result);

void CPU_greaterThan(Tensor* a, Tensor* b, Tensor* result);

void CPU_matmul(Tensor* a, Tensor* b, Tensor* result);

void CPU_sin(Tensor* a, Tensor* result);

void CPU_cos(Tensor* a, Tensor* result);

void CPU_sum(Tensor* a,  Tensor* result);

void CPU_max(Tensor* a, Tensor* result);

void CPU_min(Tensor* a, Tensor* result);

#endif // OP_CPU_HPP