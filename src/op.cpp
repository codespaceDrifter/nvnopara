#include "op.hpp"
#include "tensor.hpp"


Device opDevice (Tensor* a, Tensor* b){
    assert (a->device == b->device);
    return a->device;
}

Tensor* assertBroadcastableCreate(Tensor* a, Tensor* b) {
    int aDim = a->dim;
    int bDim = b->dim;
    int maxDim = (aDim > bDim) ? aDim : bDim;

    // Use vectors for padded shapes
    std::vector<int> aPadded(maxDim, 1);
    std::vector<int> bPadded(maxDim, 1);

    // Pad with 1s in front
    for (int i = 0; i < aDim; ++i) aPadded[maxDim - aDim + i] = a->shape[i];
    for (int i = 0; i < bDim; ++i) bPadded[maxDim - bDim + i] = b->shape[i];

    // Compute broadcasted shape
    std::vector<int> resultShape(maxDim);
    for (int i = 0; i < maxDim; ++i) {
        int aS = aPadded[i];
        int bS = bPadded[i];
        assert(aS == bS || aS == 1 || bS == 1);
        resultShape[i] = (aS > bS) ? aS : bS;
    }

    // Allocate a new int* for the shape, as Tensor expects a raw pointer
    int* shapeArr = new int[maxDim];
    for (int i = 0; i < maxDim; ++i) shapeArr[i] = resultShape[i];

    Tensor* result = new Tensor(shapeArr, maxDim);
    return result;
}

Tensor* assertReducibleCreate(Tensor* originalTensor, std::vector<int> newShape){
    // Convert int* to vector for comparison
    std::vector<int> oldShape(originalTensor->shape, originalTensor->shape + originalTensor->dim);

    assert(oldShape.size() == newShape.size());
    int dim = oldShape.size();
    int* resultShape = new int[dim];
    for (int i = 0; i < dim; i++) {
        assert(oldShape[i] == newShape[i] || newShape[i] == 1);
        if (oldShape[i] == newShape[i]){
            resultShape[i] = oldShape[i];
        } else if (newShape[i] == 1){
            resultShape[i] = 1;
        }
    }
    Tensor* result = new Tensor(resultShape, dim);
    return result;
}


Tensor* add(Tensor* a, Tensor* b) {
    Tensor* result = assertBroadcastableCreate(a,b);
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
            CUDA_add(a, b, result);
            break;
    }
    return result;
}

Tensor* sub(Tensor* a, Tensor* b) {
    Tensor* result = assertBroadcastableCreate(a,b);
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
            CUDA_sub(a, b, result);
            break;
    }
    return result;
}

Tensor* mul(Tensor* a, Tensor* b) {
    Tensor* result = assertBroadcastableCreate(a,b);
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
            CUDA_mul(a, b, result);
            break;
    }
    return result;
}

Tensor* div(Tensor* a, Tensor* b) {
    Tensor* result = assertBroadcastableCreate(a,b);
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
            CUDA_div(a, b, result);
            break;
    }
    return result;
}

Tensor* pow(Tensor* a, Tensor* b) {
    Tensor* result = assertBroadcastableCreate(a,b);
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
            CUDA_pow(a, b, result);
            break;
    }
    return result;
}

Tensor* equal(Tensor* a, Tensor* b) {
    Tensor* result = assertBroadcastableCreate(a,b);
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
            CUDA_equal(a, b, result);
            break;
    }
    return result;
}

Tensor* lessThan(Tensor* a, Tensor* b) {
    Tensor* result = assertBroadcastableCreate(a,b);
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
            CUDA_lessThan(a, b, result);
            break;
    }
    return result;
}

Tensor* greaterThan(Tensor* a, Tensor* b) {
    Tensor* result = assertBroadcastableCreate(a,b);
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
            CUDA_greaterThan(a, b, result);
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
            CUDA_sin(a, result);
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
            CUDA_cos(a, result);
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

Tensor* matmulByElemul (Tensor* a, Tensor* b) {
// asserts here. not coded yet
    a->unsqueeze(a->dim - 1);
    b->transpose(b->dim - 1, b->dim - 2);

    printf("a unsqueezed: ");
    a->print();

    printf("b transposed: ");
    b->print();


    Tensor* temp = mul(a, b);

    printf("temp: ");
    temp->print();


    std::vector<int> newShape(temp->shape, temp->shape + temp->dim);
    newShape.back() = 1;

    printf("newShape: ");
    for (int i = 0; i < newShape.size(); i++) {
        printf("%d ", newShape[i]);
    }
    printf("\n");

    Tensor* result = sum (temp, newShape);
    printf("result: ");
    result->print();

    result->squeeze(result->dim - 1);

    a->squeeze(a->dim - 2);
    b->transpose(b->dim - 2, b->dim - 1);
    delete temp;
    return result;
}