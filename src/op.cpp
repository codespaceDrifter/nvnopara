#include "op.hpp"
#include "tensor.hpp"


Device opDevice (Tensor* a, Tensor* b){
    assert (a->device == b->device);
    return a->device;
}

Tensor* assertShapeSameCreate (Tensor* a, Tensor* b){
    // Compare shapes element by element
    assert(a->dim == b->dim);
    for (int i = 0; i < a->dim; i++) {
        assert(a->shape[i] == b->shape[i]);
    }
    
    Tensor* result = new Tensor(a->shape, a->dim);
    return result;
}

Tensor* assertReducibleCreate(Tensor* originalTensor, std::vector<int> newShape){
    // Convert int* to vector for comparison
    std::vector<int> oldShape(originalTensor->shape, originalTensor->shape + originalTensor->dim);
    
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
    Tensor* result = new Tensor (resultShape.data(), resultShape.size());
    return result;
}


Tensor* add(Tensor* a, Tensor* b) {
    Tensor* result = assertShapeSameCreate(a,b);
    float* aData = a->data;
    float* bData = b->data;
    float* resultData = result->data;
    int n = result->size;
    Device device = opDevice(a, b);
    switch (device) {
        case Device::CPU:
            CPU_add(n, aData, bData, resultData);
            break;
        case Device::CUDA:
            result->toCUDA();
            CUDA_add(n, aData, bData, result->data);
            break;
    }
    return result;
}

Tensor* sub(Tensor* a, Tensor* b) {
    Tensor* result = assertShapeSameCreate(a,b);
    float* aData = a->data;
    float* bData = b->data;
    float* resultData = result->data;
    int n = result->size;
    Device device = opDevice(a, b);
    switch (device) {
        case Device::CPU:
            CPU_sub(n, aData, bData, resultData);
            break;
        case Device::CUDA:
            result->toCUDA();
            CUDA_sub(n, aData, bData, result->data);
            break;
    }
    return result;
}

Tensor* mul(Tensor* a, Tensor* b) {
    Tensor* result = assertShapeSameCreate(a,b);
    float* aData = a->data;
    float* bData = b->data;
    float* resultData = result->data;
    int n = result->size;
    Device device = opDevice(a, b);
    switch (device) {
        case Device::CPU:
            CPU_mul(n, aData, bData, resultData);
            break;
        case Device::CUDA:
            result->toCUDA();
            CUDA_mul(n, aData, bData, result->data);
            break;
    }
    return result;
}

Tensor* div(Tensor* a, Tensor* b) {
    Tensor* result = assertShapeSameCreate(a,b);
    float* aData = a->data;
    float* bData = b->data;
    float* resultData = result->data;
    int n = result->size;
    Device device = opDevice(a, b);
    switch (device) {
        case Device::CPU:
            CPU_div(n, aData, bData, resultData);
            break;
        case Device::CUDA:
            result->toCUDA();
            CUDA_div(n, aData, bData, result->data);
            break;
    }
    return result;
}

Tensor* pow(Tensor* a, Tensor* b) {
    Tensor* result = assertShapeSameCreate(a,b);
    float* aData = a->data;
    float* bData = b->data;
    float* resultData = result->data;
    int n = result->size;
    Device device = opDevice(a, b);
    switch (device) {
        case Device::CPU:
            CPU_pow(n, aData, bData, resultData);
            break;
        case Device::CUDA:
            result->toCUDA();
            CUDA_pow(n, aData, bData, result->data);
            break;
    }
    return result;
}

Tensor* equal(Tensor* a, Tensor* b) {
    Tensor* result = assertShapeSameCreate(a,b);
    float* aData = a->data;
    float* bData = b->data;
    float* resultData = result->data;
    int n = result->size;
    Device device = opDevice(a, b);
    switch (device) {
        case Device::CPU:
            CPU_equal(n, aData, bData, resultData);
            break;
        case Device::CUDA:
            result->toCUDA();
            CUDA_equal(n, aData, bData, result->data);
            break;
    }
    return result;
}

Tensor* lessThan(Tensor* a, Tensor* b) {
    Tensor* result = assertShapeSameCreate(a,b);
    float* aData = a->data;
    float* bData = b->data;
    float* resultData = result->data;
    int n = result->size;
    Device device = opDevice(a, b);
    switch (device) {
        case Device::CPU:
            CPU_lessThan(n, aData, bData, resultData);
            break;
        case Device::CUDA:
            result->toCUDA();
            CUDA_lessThan(n, aData, bData, result->data);
            break;
    }
    return result;
}

Tensor* greaterThan(Tensor* a, Tensor* b) {
    Tensor* result = assertShapeSameCreate(a,b);
    float* aData = a->data;
    float* bData = b->data;
    float* resultData = result->data;
    int n = result->size;
    Device device = opDevice(a, b);
    switch (device) {
        case Device::CPU:
            CPU_greaterThan(n, aData, bData, resultData);
            break;
        case Device::CUDA:
            result->toCUDA();
            CUDA_greaterThan(n, aData, bData, result->data);
            break;
    }
    return result;
}

Tensor* sin(Tensor* a) {
    Tensor* result = new Tensor(a->shape, a->dim);
    float* aData = a->data;
    float* resultData = result->data;
    int n = result->size;
    switch (a->device) {
        case Device::CPU:
            CPU_sin(n, aData, resultData);
            break;
        case Device::CUDA:
            result->toCUDA();
            CUDA_sin(n, aData, result->data);
            break;
    }
    return result;
}

Tensor* cos(Tensor* a) {
    Tensor* result = new Tensor(a->shape, a->dim);
    float* aData = a->data;
    float* resultData = result->data;
    int n = result->size;
    switch (a->device) {
        case Device::CPU:
            CPU_cos(n, aData, resultData);
            break;
        case Device::CUDA:
            result->toCUDA();
            CUDA_cos(n, aData, result->data);
            break;
    }
    return result;
}

Tensor* matmul(Tensor* a, Tensor* b) {
    assert (a->dim == 2);
    assert (b->dim == 2);
    assert(a->shape[1] == b->shape[0]);
    Tensor* result = new Tensor({a->shape[0], b->shape[1]});

    float* aData = a->data;
    float* bData = b->data;
    float* resultData = result->data;

    //k is the shared common dimension not in the result shape
    int m = a->shape[0];
    int n = b->shape[1];
    int k = a->shape[1];

    Device device = opDevice(a, b);
    switch (device) {
        case Device::CPU:
            CPU_matmul(m, n, k, aData, bData, resultData);
            break;
        case Device::CUDA:
            result->toCUDA();
            CUDA_matmul(m, n, k, aData, bData, result->data);
            break;
    }
    return result;
}


Tensor* sum(Tensor* a, std::vector<int> newShape) {
    Tensor* result = assertReducibleCreate(a, newShape);
    float* aData = a->data;
    float* resultData = result->data;
    switch (a->device) {
        case Device::CPU:
            CPU_sum(a, result);
            break;
        case Device::CUDA:
            result->toCUDA();
            CUDA_sum(a, result);
            break;
    }
    return result;
}

Tensor* max(Tensor* a, std::vector<int> newShape) {
    Tensor* result = assertReducibleCreate(a, newShape);
    float* aData = a->data;
    float* resultData = result->data;
    switch (a->device) {
        case Device::CPU:
            CPU_max(a, result);
            break;
        case Device::CUDA:
            result->toCUDA();
            CUDA_max(a, result);
            break;
    }
    return result;
}

Tensor* min(Tensor* a, std::vector<int> newShape) {
    Tensor* result = assertReducibleCreate(a, newShape);
    float* aData = a->data;
    float* resultData = result->data;
    switch (a->device) {
        case Device::CPU:
            CPU_min(a, result);
            break;
        case Device::CUDA:
            result->toCUDA();
            CUDA_min(a, result);
            break;
    }
    return result;
}
