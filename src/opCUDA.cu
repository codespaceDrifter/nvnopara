#include "opCUDA.hpp"

#define MAX_DIM 6
#define BLOCK_SIZE 256


__device__ void flatToIndice (int idx, int dim, int* shape, int* indice){
    int lastDim = dim - 1;
    for (int i = lastDim; i >= 0; i--) {
        indice[i] = idx % shape[i];
        idx = idx / shape[i];
    }
}

// indiceDim must be greater or equal than strideDim
__device__ int indiceToFlat (int indiceDim, int* indice, int strideDim, int* stride){
    int resultIdx = 0;
    int offset = indiceDim - strideDim;
    for (int i = 0; i < strideDim; i++) {
        resultIdx += indice[i + offset] * stride[i];
    }
    return resultIdx;
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
        result[idx] = a[idx] + b[idx];
    }
}


__global__ void subKernel(int n, float*a, float*b, float*result){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        result[idx] = a[idx] - b[idx];
    }
}

__global__ void mulKernel(int n, float*a, float*b, float*result){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        result[idx] = a[idx] * b[idx];
    }
}

__global__ void divKernel(int n, float*a, float*b, float*result){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        result[idx] = a[idx] / b[idx];
    }
}

__global__ void powKernel(int n, float*a, float*b, float*result){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        result[idx] = pow(a[idx], b[idx]);
    }
}

__global__ void equalKernel(int n, float*a, float*b, float*result){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        result[idx] = a[idx] == b[idx] ? 1.0f : 0.0f;
    }
}

__global__ void lessThanKernel(int n, float*a, float*b, float*result){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        result[idx] = a[idx] < b[idx] ? 1.0f : 0.0f;
    }
}

__global__ void greaterThanKernel(int n, float*a, float*b, float*result){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        result[idx] = a[idx] > b[idx] ? 1.0f : 0.0f;
    }
}

__global__ void sinKernel(int n, float*a, float*result){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        result[idx] = sinf(a[idx]);
    }
}

__global__ void cosKernel(int n, float*a, float*result){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        result[idx] = cosf(a[idx]);
    }
}

__global__ void sumKernel(int n, float*a, float*result, int shapeDim, int* aShape,int strideDim, int* resultStride){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n){
        int indice[MAX_DIM];
        flatToIndice(idx, shapeDim, aShape, indice);
        int resultIdx = indiceToFlat(shapeDim, indice, strideDim, resultStride);
        atomicAdd(&result[resultIdx], a[idx]);
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

__global__ void maxKernel(int n, float*a, float*result, int shapeDim, int* aShape,int strideDim, int* resultStride){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        int indice[MAX_DIM];
        flatToIndice(idx, shapeDim, aShape, indice);
        int resultIdx = indiceToFlat(shapeDim, indice, strideDim, resultStride);
        atomicMax(&result[resultIdx], a[idx]);
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

__global__ void minKernel(int n, float*a, float*result, int shapeDim, int* aShape,int strideDim, int* resultStride){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        int indice[MAX_DIM];
        flatToIndice(idx, shapeDim, aShape, indice);
        int resultIdx = indiceToFlat(shapeDim, indice, strideDim, resultStride);
        atomicMin(&result[resultIdx], a[idx]);
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

void CUDA_add(int n, float* a, float* b, float* result) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    addKernel<<<numBlocks, blockSize>>>(n, a, b, result);
}


void CUDA_sub(int n, float* a, float* b, float* result) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    subKernel<<<numBlocks, blockSize>>>(n, a, b, result);
}

void CUDA_mul(int n, float* a, float* b, float* result) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    mulKernel<<<numBlocks, blockSize>>>(n, a, b, result);
}

void CUDA_div(int n, float* a, float* b, float* result) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    divKernel<<<numBlocks, blockSize>>>(n, a, b, result);
}

void CUDA_pow(int n, float* a, float* b, float* result) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    powKernel<<<numBlocks, blockSize>>>(n, a, b, result);
}

void CUDA_equal(int n, float* a, float* b, float* result) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    equalKernel<<<numBlocks, blockSize>>>(n, a, b, result);
}

void CUDA_lessThan(int n, float* a, float* b, float* result) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    lessThanKernel<<<numBlocks, blockSize>>>(n, a, b, result);
}

void CUDA_greaterThan(int n, float* a, float* b, float* result) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    greaterThanKernel<<<numBlocks, blockSize>>>(n, a, b, result);
}

void CUDA_matmul(int m, int n, int k, float* a, float* b, float* result) {
    dim3 blockDim(16, 16);
    dim3 gridDim((n+15)/16, (m+15)/16);
    matmulKernel<<<gridDim, blockDim>>>(m, n, k, a, b, result);
}

void CUDA_sin(int n, float* a, float* result) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    sinKernel<<<numBlocks, blockSize>>>(n, a, result);
}

void CUDA_cos(int n, float* a, float* result) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    cosKernel<<<numBlocks, blockSize>>>(n, a, result);
}

void CUDA_sum(Tensor* a, Tensor* result) {
    float* aData = a->data;
    float* resultData = result->data;

    int blockSize = 256;
    int numBlocks = (a->size + blockSize - 1) / blockSize;
    sumKernel<<<numBlocks, blockSize>>>(a->size, aData, resultData, a->dim, a->d_shape, result->dim, result->d_stride);
}

void CUDA_max(Tensor* a, Tensor* result) {
    float* aData = a->data;
    float* resultData = result->data;

    int blockSize = 256;

    int resultNumBlocks = (result->size + blockSize - 1) / blockSize;
    fillKernel<<<resultNumBlocks, blockSize>>>(result->size, resultData, -1e10f);

    int numBlocks = (a->size + blockSize - 1) / blockSize;
    maxKernel<<<numBlocks, blockSize>>>(a->size, aData, resultData, a->dim, a->d_shape, result->dim, result->d_stride);

    cudaDeviceSynchronize();
}

void CUDA_min(Tensor* a, Tensor* result) {
    float* aData = a->data;
    float* resultData = result->data;

    int blockSize = 256;

    int resultNumBlocks = (result->size + blockSize - 1) / blockSize;
    fillKernel<<<resultNumBlocks, blockSize>>>(result->size, resultData, 1e10f);

    int numBlocks = (a->size + blockSize - 1) / blockSize;
    minKernel<<<numBlocks, blockSize>>>(a->size, aData, resultData, a->dim, a->d_shape, result->dim, result->d_stride);

    cudaDeviceSynchronize();
}
