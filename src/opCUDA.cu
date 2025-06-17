#include "opCUDA.hpp"


__device__ int idxConversion (int idx, int dim, int* oldShape, int* newStride){

    int resultIdx = 0;
    int endStart = dim - 1;
    for (int i = endStart; i >= 0; i--) {
        int tempResultIdx = idx % oldShape[i] * newStride[i];
        resultIdx += tempResultIdx;
        idx = idx / oldShape[i];
    }


    return resultIdx;
}

__global__ void fillKernel(float* result, float value, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        result[idx] = value;
    }
}


__global__ void addKernel(float*a, float*b, float*result, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        result[idx] = a[idx] + b[idx];
    }
}

__global__ void subKernel(float*a, float*b, float*result, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        result[idx] = a[idx] - b[idx];
    }
}

__global__ void mulKernel(float*a, float*b, float*result, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        result[idx] = a[idx] * b[idx];
    }
}

__global__ void divKernel(float*a, float*b, float*result, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        result[idx] = a[idx] / b[idx];
    }
}

__global__ void powKernel(float*a, float*b, float*result, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        result[idx] = pow(a[idx], b[idx]);
    }
}

__global__ void equalKernel(float*a, float*b, float*result, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        result[idx] = a[idx] == b[idx] ? 1.0f : 0.0f;
    }
}

__global__ void lessThanKernel(float*a, float*b, float*result, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        result[idx] = a[idx] < b[idx] ? 1.0f : 0.0f;
    }
}

__global__ void greaterThanKernel(float*a, float*b, float*result, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        result[idx] = a[idx] > b[idx] ? 1.0f : 0.0f;
    }
}

__global__ void sinKernel(float*a, float*result, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        result[idx] = sinf(a[idx]);
    }
}

__global__ void cosKernel(float*a, float*result, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        result[idx] = cosf(a[idx]);
    }
}

__global__ void sumKernel(float*a, float*result, int dim, int* aShape, int* resultStride, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n){
        int resultIdx = idxConversion(idx, dim, aShape, resultStride);
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

__global__ void maxKernel(float*a, float*result, int dim, int* aShape, int* resultStride, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        int resultIdx = idxConversion(idx, dim, aShape, resultStride);
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

__global__ void minKernel(float*a, float*result, int dim, int* aShape, int* resultStride, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        int resultIdx = idxConversion(idx, dim, aShape, resultStride);
        result[resultIdx] = fminf(result[resultIdx], a[idx]);
    }
}

//this is a naive and slow matmul kernel. no tiles, no tensor cores

__global__ void matmulKernel(float* A, float* B, float* C, int M, int N, int K) {
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

void reduceOpParameterGen (Tensor* a, Tensor* result, int** d_aShape, int** d_resultStride){

    int* aShape = a->shape.data();
    int dim = a->shape.size();

    std::vector<int> extendResultStride(dim, 0);
    std::copy(result->stride.begin(), result->stride.end(),
        extendResultStride.end() - result->stride.size());
    int* resultStride = extendResultStride.data();

    cudaMalloc(d_aShape, dim * sizeof(int));
    cudaMalloc(d_resultStride, dim * sizeof(int));

    cudaMemcpy(*d_aShape, aShape, dim * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_resultStride, resultStride, dim * sizeof(int), cudaMemcpyHostToDevice);

}


void CUDA_add(float* a, float* b, int n, float* result) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    addKernel<<<numBlocks, blockSize>>>(a, b, result, n);
}


void CUDA_sub(float* a, float* b, int n, float* result) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    subKernel<<<numBlocks, blockSize>>>(a, b, result, n);
}

void CUDA_mul(float* a, float* b, int n, float* result) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    mulKernel<<<numBlocks, blockSize>>>(a, b, result, n);
}

void CUDA_div(float* a, float* b, int n, float* result) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    divKernel<<<numBlocks, blockSize>>>(a, b, result, n);
}

void CUDA_pow(float* a, float* b, int n, float* result) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    powKernel<<<numBlocks, blockSize>>>(a, b, result, n);
}

void CUDA_equal(float* a, float* b, int n, float* result) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    equalKernel<<<numBlocks, blockSize>>>(a, b, result, n);
}

void CUDA_lessThan(float* a, float* b, int n, float* result) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    lessThanKernel<<<numBlocks, blockSize>>>(a, b, result, n);
}

void CUDA_greaterThan(float* a, float* b, int n, float* result) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    greaterThanKernel<<<numBlocks, blockSize>>>(a, b, result, n);
}

void CUDA_matmul(float* a, float* b, int m, int n, int k, float* result) {
    dim3 blockDim(16, 16);
    dim3 gridDim((n+15)/16, (m+15)/16);
    matmulKernel<<<gridDim, blockDim>>>(a, b, result, m, n, k);
}

void CUDA_sin(float* a, int n, float* result) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    sinKernel<<<numBlocks, blockSize>>>(a, result, n);
}

void CUDA_cos(float* a, int n, float* result) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    cosKernel<<<numBlocks, blockSize>>>(a, result, n);
}

void CUDA_sum(Tensor* a, Tensor* result) {
    float* aData = a->data;


    float* resultData = result->data;

    int* d_aShape;
    int* d_resultStride;

    reduceOpParameterGen(a, result, &d_aShape, &d_resultStride);

    int dim = a->shape.size();

    int blockSize = 256;

    int numBlocks = (a->size + blockSize - 1) / blockSize;
    sumKernel<<<numBlocks, blockSize>>>(aData, resultData, dim, d_aShape, d_resultStride, a->size);

    cudaFree(d_aShape);
    cudaFree(d_resultStride);
}

void CUDA_max(Tensor* a, Tensor* result) {

    float* aData = a->data;
    float* resultData = result->data;

    int* d_aShape;
    int* d_resultStride;

    reduceOpParameterGen(a, result, &d_aShape, &d_resultStride);

    int dim = a->shape.size();

    int blockSize = 256;

    int resultNumBlocks = (result->size + blockSize - 1) / blockSize;
    fillKernel<<<resultNumBlocks, blockSize>>>(resultData, -1e10f, result->size);

    int numBlocks = (a->size + blockSize - 1) / blockSize;

    maxKernel<<<numBlocks, blockSize>>>(aData, resultData, dim, d_aShape, d_resultStride, a->size);

    cudaDeviceSynchronize();

    cudaFree(d_aShape);
    cudaFree(d_resultStride);
}

void CUDA_min(Tensor* a, Tensor* result) {
    float* aData = a->data;
    float* resultData = result->data;

    int* d_aShape;
    int* d_resultStride;

    reduceOpParameterGen(a, result, &d_aShape, &d_resultStride);

    int dim = a->shape.size();

    int blockSize = 256;

    int resultNumBlocks = (result->size + blockSize - 1) / blockSize;
    fillKernel<<<resultNumBlocks, blockSize>>>(resultData, 1e10f, result->size);

    int numBlocks = (a->size + blockSize - 1) / blockSize;

    minKernel<<<numBlocks, blockSize>>>(aData, resultData, dim, d_aShape, d_resultStride, a->size);

    cudaDeviceSynchronize();

    cudaFree(d_aShape);
    cudaFree(d_resultStride);
}
