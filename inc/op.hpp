#ifndef OP_HPP
#define OP_HPP

// checks if shape compatible then pass parameters to separate device functions
// assumes all of these are NOT inplace

#include <vector>
#include <cassert>
#include "tensor.hpp"
#include "opCPU.hpp"
#include "opCUDA.hpp"

Device opDevice (Tensor* a, Tensor* b);

Tensor* assertReducibleCreate(Tensor* originalTensor, std::vector<int>newShape);

Tensor* assertBroadcastableCreate (Tensor* a, Tensor* b);

Tensor* add(Tensor* a, Tensor* b);

Tensor* sub(Tensor* a, Tensor* b);

Tensor* mul(Tensor* a, Tensor* b);

Tensor* div(Tensor* a, Tensor* b);

Tensor* pow(Tensor* a, Tensor* b);

Tensor* equal(Tensor* a, Tensor* b);

Tensor* lessThan(Tensor* a, Tensor* b);

Tensor* greaterThan(Tensor* a, Tensor* b);

Tensor* matmul(Tensor* a, Tensor* b);

Tensor* sin(Tensor* a);

Tensor* cos(Tensor* a);

Tensor* sum(Tensor* a, std::vector<int> axes);

Tensor* max(Tensor* a, std::vector<int> axes);

Tensor* min(Tensor* a, std::vector<int> axes);

Tensor* matmulByElemul(Tensor* a, Tensor* b);

#endif // OP_HPP
