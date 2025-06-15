#include "op.hpp"
#include "tensor.hpp"
#include "backend.hpp"

Tensor* assertShapeSameCreate (Tensor* a, Tensor* b){
    assert (a->shape == b->shape);
    Tensor* result = new Tensor (a->shape);
    return result;
}

Tensor* assertReducibleCreate(Tensor* originalTensor, std::vector<int>newShape){
    std::vector<int> oldShape = originalTensor->shape;
    assert(oldShape.size() == newShape.size());
    std::vector<int> resultShape (oldShape.size());
    for (int i = 0; i < oldShape.size(); i++) {
        assert(oldShape[i] == newShape[i] || newShape[i] == 1);
        if (oldShape[i] == newShape[i]){
            resultShape[i] = oldShape[i];
        } else if (newShape[i] == 1){
            resultShape[i] = 1;
        }
    }
    Tensor* result = new Tensor (resultShape);
    return result;
}


Tensor* add(Tensor* a, Tensor* b) {
    Tensor* result = assertShapeSameCreate(a,b);
    switch (currentBackend) {
        case Backend::CPU:
            CPU_add(a, b, result);
            break;
        case Backend::CUDA:
            CUDA_add(a, b, result);
            break;
    }
    return result;
}

Tensor* sub(Tensor* a, Tensor* b) {
    Tensor* result = assertShapeSameCreate(a,b);
    switch (currentBackend) {
        case Backend::CPU:
            CPU_sub(a, b, result);
            break;
        case Backend::CUDA:
            CUDA_sub(a, b, result);
            break;
    }
    return result;
}

Tensor* mul(Tensor* a, Tensor* b) {
    Tensor* result = assertShapeSameCreate(a,b);
    switch (currentBackend) {
        case Backend::CPU:
            CPU_mul(a, b, result);
            break;
        case Backend::CUDA:
            CUDA_mul(a, b, result);
            break;
    }
    return result;
}

Tensor* div(Tensor* a, Tensor* b) {
    Tensor* result = assertShapeSameCreate(a,b);
    switch (currentBackend) {
        case Backend::CPU:
            CPU_div(a, b, result);
            break;
        case Backend::CUDA:
            CUDA_div(a, b, result);
            break;
    }
    return result;
}

Tensor* pow(Tensor* a, Tensor* b) {
    Tensor* result = assertShapeSameCreate(a,b);
    switch (currentBackend) {
        case Backend::CPU:
            CPU_pow(a, b, result);
            break;
        case Backend::CUDA:
            CUDA_pow(a, b, result);
            break;
    }
    return result;
}

Tensor* equal(Tensor* a, Tensor* b) {
    Tensor* result = assertShapeSameCreate(a,b);
    switch (currentBackend) {
        case Backend::CPU:
            CPU_equal(a, b, result);
            break;
        case Backend::CUDA:
            CUDA_equal(a, b, result);
            break;
    }
    return result;
}

Tensor* lessThan(Tensor* a, Tensor* b) {
    Tensor* result = assertShapeSameCreate(a,b);
    switch (currentBackend) {
        case Backend::CPU:
            CPU_lessThan(a, b, result);
            break;
        case Backend::CUDA:
            CUDA_lessThan(a, b, result);
            break;
    }
    return result;
}

Tensor* greaterThan(Tensor* a, Tensor* b) {
    Tensor* result = assertShapeSameCreate(a,b);
    switch (currentBackend) {
        case Backend::CPU:
            CPU_greaterThan(a, b, result);
            break;
        case Backend::CUDA:
            CUDA_greaterThan(a, b, result);
            break;
    }
    return result;
}

Tensor* sin(Tensor* a) {
    Tensor* result = new Tensor(a->shape);
    switch (currentBackend) {
        case Backend::CPU:
            CPU_sin(a, result);
            break;
        case Backend::CUDA:
            CUDA_sin(a, result);
            break;
    }
    return result;
}

Tensor* cos(Tensor* a) {
    Tensor* result = new Tensor(a->shape);
    switch (currentBackend) {
        case Backend::CPU:
            CPU_cos(a, result);
            break;
        case Backend::CUDA:
            CUDA_cos(a, result);
            break;
    }
    return result;
}

Tensor* matmul(Tensor* a, Tensor* b) {
    assert (a->shape.size() == 2);
    assert (b->shape.size() == 2);
    assert(a->shape[1] && b->shape[0]);
    std::vector<int> resultShape = {a->shape[0], b->shape[1]};
    Tensor* result = new Tensor(resultShape);
    switch (currentBackend) {
        case Backend::CPU:
            CPU_matmul(a, b, result);
            break;
        case Backend::CUDA:
            CUDA_matmul(a, b, result);
            break;
    }
    return result;
}


Tensor* sum(Tensor* a, std::vector<int> newShape) {
    Tensor* result = assertReducibleCreate(a, newShape);
    switch (currentBackend) {
        case Backend::CPU:
            CPU_sum(a, result);
            break;
        case Backend::CUDA:
            CUDA_sum(a, result);
            break;
    }
    return result;
}

Tensor* max(Tensor* a, std::vector<int> newShape) {

    Tensor* result = assertReducibleCreate(a, newShape);

    switch (currentBackend) {
        case Backend::CPU:
            CPU_max(a, result);
            break;
        case Backend::CUDA:
            CUDA_max(a, result);
            break;
    }
    return result;
}

Tensor* min(Tensor* a, std::vector<int> newShape) {

    Tensor* result = assertReducibleCreate(a, newShape);

    switch (currentBackend) {
        case Backend::CPU:
            CPU_min(a, result);
            break;
        case Backend::CUDA:
            CUDA_min(a, result);
            break;
    }
    return result;
}