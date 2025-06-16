#include "opCUDA.hpp"
#include <cuda_runtime.h>

void CUDA_add(float* a, float* b, int n, float* result) {
    float *d_a, *d_b, *d_result;
    size_t size = n * sizeof(float);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_result, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    addKernel<<<1, n>>>(d_a, d_b, d_result, n);
}

__global__ void addKernel(float*a, float*b, float*result, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        result[idx] = a[idx] + b[idx];
    }
}

void CUDA_sub(float* a, float* b, int n, float* result) { }

void CUDA_mul(float* a, float* b, int n, float* result) { }

void CUDA_div(float* a, float* b, int n, float* result) { }

void CUDA_pow(float* a, float* b, int n, float* result) { }

void CUDA_equal(float* a, float* b, int n, float* result) { }

void CUDA_lessThan(float* a, float* b, int n, float* result) { }

void CUDA_greaterThan(float* a, float* b, int n, float* result) { }

void CUDA_matmul(float* a, float* b, int m, int n, int k, float* result) { }

void CUDA_sin(float* a, int n, float* result) { }

void CUDA_cos(float* a, int n, float* result) { }

void CUDA_sum(Tensor* a, Tensor* result) { }

void CUDA_max(Tensor* a, Tensor* result) { }

void CUDA_min(Tensor* a, Tensor* result) { }
