#include "opCUDA.hpp"

#define MAX_DIM 16
#define BLOCK_SIZE 256

__constant__ int CM_resultDim;

//all the same length. the stride is pre padded with zero to match shape dim
__constant__ int CM_resultShape[MAX_DIM];
__constant__ int CM_aStride[MAX_DIM];
__constant__ int CM_bStride[MAX_DIM];


__device__ void idxConversion (int idx, int* aIdx, int* bIdx){
    int indice[MAX_DIM];
    int resultDimMinusOne = CM_resultDim - 1;
    for (int i = resultDimMinusOne; i >= 0; i--) {
        indice[i] = idx % CM_resultShape[i];
        idx = idx / CM_resultShape[i];
    }

    int aResultIdx = 0;
    for (int i = 0; i < CM_resultDim; i++) {
        aResultIdx += indice[i] * CM_aStride[i];
    }
    *aIdx = aResultIdx;

    if (bIdx){
        int bResultIdx = 0;
        for (int i = 0; i < CM_resultDim; i++) {
            bResultIdx += indice[i] * CM_bStride[i];
        }
        *bIdx = bResultIdx;
    }

}


__global__ void fillKernel(int n, float* result, float value){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        result[idx] = value;
    }
}


__global__ void addKernel(int n, float*a, float*b, float*result){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        int aIdx, bIdx;
        idxConversion(idx, &aIdx, &bIdx);
        result[idx] = a[aIdx] + b[bIdx];
    }
}


__global__ void subKernel(int n, float*a, float*b, float*result){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        int aIdx, bIdx;
        idxConversion(idx, &aIdx, &bIdx);
        result[idx] = a[aIdx] - b[bIdx];
    }
}

__global__ void mulKernel(int n, float*a, float*b, float*result){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        int aIdx, bIdx;
        idxConversion(idx, &aIdx, &bIdx);
        result[idx] = a[aIdx] * b[bIdx];
    }
}

__global__ void divKernel(int n, float*a, float*b, float*result){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        int aIdx, bIdx;
        idxConversion(idx, &aIdx, &bIdx);
        result[idx] = a[aIdx] / b[bIdx];
    }
}

__global__ void powKernel(int n, float*a, float*b, float*result){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        int aIdx, bIdx;
        idxConversion(idx, &aIdx, &bIdx);
        result[idx] = pow(a[aIdx], b[bIdx]);
    }
}

__global__ void equalKernel(int n, float*a, float*b, float*result){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        int aIdx, bIdx;
        idxConversion(idx, &aIdx, &bIdx);
        result[idx] = a[aIdx] == b[bIdx] ? 1.0f : 0.0f;
    }
}

__global__ void lessThanKernel(int n, float*a, float*b, float*result){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        int aIdx, bIdx;
        idxConversion(idx, &aIdx, &bIdx);
        result[idx] = a[aIdx] < b[bIdx] ? 1.0f : 0.0f;
    }
}

__global__ void greaterThanKernel(int n, float*a, float*b, float*result){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        int aIdx, bIdx;
        idxConversion(idx, &aIdx, &bIdx);
        result[idx] = a[aIdx] > b[bIdx] ? 1.0f : 0.0f;
    }
}

__global__ void sinKernel(int n, float*a, float*result){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        int aIdx;
        idxConversion(idx, &aIdx, nullptr);
        result[idx] = sinf(a[aIdx]);
    }
}

__global__ void cosKernel(int n, float*a, float*result){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        int aIdx;
        idxConversion(idx, &aIdx, nullptr);
        result[idx] = cosf(a[aIdx]);
    }
}

__global__ void sumKernel(int n, float*a, float*result){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n){
        int aIdx;
        idxConversion(idx, &aIdx, nullptr);
        atomicAdd(&result[idx], a[aIdx]);
    }
}

// atmoically determine the max between a result (address) and a value (from a)
__device__ void atomicMax(float* address, float val){
    int* address_as_int = (int*)address;  // atomicCAS needs ints. 

    // need a assume and an old because during the loop
    // because another thread could have changed the value at the address and the comparison would not be correct 
    int old = *address_as_int; // old value. this is NOT a int conversion this is a bit reinterpretation
    int assumed; 

    do {
        //the value at address_as_int could change here in the loop
        assumed = old;
        float fAssumed = __int_as_float(assumed);
        float curMax = fmaxf(fAssumed, val);

        // atomic CAS logic:
        // returns actual value at address_as_int no matter the comparison result
        // if value at addres_as_int == assumed, then set it to val

        // old gets set to value at address_as_int in this line
        old = atomicCAS(address_as_int, assumed, __float_as_int(curMax));
    } while (assumed != old);
}

__global__ void maxKernel(int n, float*a, float*result){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        int aIdx;
        idxConversion(idx, &aIdx, nullptr);
        atomicMax(&result[idx], a[aIdx]);
    }
}

__device__ void atomicMin(float* address, float val){
    int* address_as_int = (int*)address;
    int old = *address_as_int;
    int assumed;

    do {
        assumed = old;
        float fAssumed = __int_as_float(assumed);
        float curMin = fminf(fAssumed, val);
        old = atomicCAS(address_as_int, assumed, __float_as_int(curMin));
    } while (assumed != old);
}

__global__ void minKernel(int n, float*a, float*result){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        int aIdx;
        idxConversion(idx, &aIdx, nullptr);
        atomicMin(&result[idx], a[aIdx]);
    }
}

//this is a naive and slow matmul kernel. no tiles, no tensor cores

__global__ void matmulKernel(int M, int N, int K, float* A, float* B, float* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}




static void prepCM(Tensor* a, Tensor* b, Tensor* result){
    int resultDim = result->dim;
    assert (resultDim <= MAX_DIM);

    // this is not passing in a int this is passing in CM_resultDim as a symbol
    // which the compiler makes it into an address
    cudaMemcpyToSymbol(CM_resultDim, &resultDim, sizeof(int));
    cudaMemcpyToSymbol(CM_resultShape, result->shape, resultDim * sizeof(int));

    int aStridePadded[MAX_DIM] = {0};
    int aOffset = resultDim - a->dim;
    assert (aOffset >= 0);
    memcpy(aStridePadded + aOffset, a->stride, a->dim * sizeof(int));
    cudaMemcpyToSymbol(CM_aStride, aStridePadded, MAX_DIM * sizeof(int));


    if (b){
        int bStridePadded[MAX_DIM] = {0};
        int bOffset = resultDim - b->dim;
        assert (bOffset >= 0);
        memcpy(bStridePadded + bOffset, b->stride, b->dim * sizeof(int));
        cudaMemcpyToSymbol(CM_bStride, bStridePadded, MAX_DIM * sizeof(int));
    }
}



void CUDA_add(Tensor* a, Tensor* b, Tensor* result) {
    int n = result->size;
    prepCM(a, b, result);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    addKernel<<<numBlocks, blockSize>>>(n, a->data, b->data, result->data);
}


void CUDA_sub(Tensor* a, Tensor* b, Tensor* result) {
    int n = result->size;
    prepCM(a, b, result);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    subKernel<<<numBlocks, blockSize>>>(n, a->data, b->data, result->data);

}

void CUDA_mul(Tensor* a, Tensor* b, Tensor* result) {
    int n = result->size;
    prepCM(a, b, result);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    mulKernel<<<numBlocks, blockSize>>>(n, a->data, b->data, result->data);
}

void CUDA_div(Tensor* a, Tensor* b, Tensor* result) {
    int n = result->size;
    prepCM(a, b, result);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    divKernel<<<numBlocks, blockSize>>>(n, a->data, b->data, result->data);
}

void CUDA_pow(Tensor* a, Tensor* b, Tensor* result) {
    int n = result->size;
    prepCM(a, b, result);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    powKernel<<<numBlocks, blockSize>>>(n, a->data, b->data, result->data);
}

void CUDA_equal(Tensor* a, Tensor* b, Tensor* result) {
    int n = result->size;
    prepCM(a, b, result);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    equalKernel<<<numBlocks, blockSize>>>(n, a->data, b->data, result->data);
}

void CUDA_lessThan(Tensor* a, Tensor* b, Tensor* result) {
    int n = result->size;
    prepCM(a, b, result);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    lessThanKernel<<<numBlocks, blockSize>>>(n, a->data, b->data, result->data);
}

void CUDA_greaterThan(Tensor* a, Tensor* b, Tensor* result) {
    int n = result->size;
    prepCM(a, b, result);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    greaterThanKernel<<<numBlocks, blockSize>>>(n, a->data, b->data, result->data);
}

void CUDA_matmul(int m, int n, int k, float* a, float* b, float* result) {
    dim3 blockDim(16, 16);
    dim3 gridDim((n+15)/16, (m+15)/16);
    matmulKernel<<<gridDim, blockDim>>>(m, n, k, a, b, result);
}

void CUDA_sin(Tensor* a, Tensor* result) {
    int n = result->size;
    prepCM(a, nullptr, result);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    sinKernel<<<numBlocks, blockSize>>>(n, a->data, result->data);
}

void CUDA_cos(Tensor* a, Tensor* result) {
    int n = result->size;
    prepCM(a, nullptr, result);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    cosKernel<<<numBlocks, blockSize>>>(n, a->data, result->data);
}

void CUDA_sum(Tensor* a, Tensor* result) {
    float* aData = a->data;
    float* resultData = result->data;
    prepCM(a, nullptr, result);

    int blockSize = 256;
    int numBlocks = (a->size + blockSize - 1) / blockSize;
    sumKernel<<<numBlocks, blockSize>>>(a->size, aData, resultData);
}

void CUDA_max(Tensor* a, Tensor* result) {
    float* aData = a->data;
    float* resultData = result->data;
    prepCM(a, nullptr, result);

    int blockSize = 256;

    int resultNumBlocks = (result->size + blockSize - 1) / blockSize;
    fillKernel<<<resultNumBlocks, blockSize>>>(result->size, resultData, -1e10f);

    int numBlocks = (a->size + blockSize - 1) / blockSize;
    maxKernel<<<numBlocks, blockSize>>>(a->size, aData, resultData);

    cudaDeviceSynchronize();
}

void CUDA_min(Tensor* a, Tensor* result) {
    float* aData = a->data;
    float* resultData = result->data;
    prepCM(a, nullptr, result);

    int blockSize = 256;

    int resultNumBlocks = (result->size + blockSize - 1) / blockSize;
    fillKernel<<<resultNumBlocks, blockSize>>>(result->size, resultData, 1e10f);

    int numBlocks = (a->size + blockSize - 1) / blockSize;
    minKernel<<<numBlocks, blockSize>>>(a->size, aData, resultData);

    cudaDeviceSynchronize();
}
