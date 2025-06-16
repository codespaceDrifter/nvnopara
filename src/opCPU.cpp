#include "opCPU.hpp"

void CPU_add(float* a, float* b, int n, float* result) {
    for (int i = 0; i < n; i++) {
        result[i] = a[i] + b[i];
    }
}

void CPU_sub(float* a, float* b, int n, float* result) {
    for (int i = 0; i < n; i++) {
        result[i] = a[i] - b[i];
    }
}

void CPU_mul(float* a, float* b, int n, float* result) {
    for (int i = 0; i < n; i++) {
        result[i] = a[i] * b[i];
    }
}

void CPU_div(float* a, float* b, int n, float* result) {
    for (int i = 0; i < n; i++) {
        result[i] = a[i] / b[i];
    }
}

void CPU_pow(float* a, float* b, int n, float* result) {
    for (int i = 0; i < n; i++) {
        result[i] = std::pow(a[i], b[i]);
    }
}

void CPU_equal(float* a, float* b, int n, float* result) {
    for (int i = 0; i < n; i++) {
        result[i] = a[i] == b[i];
    }
}

void CPU_lessThan(float* a, float* b, int n, float* result) {
    for (int i = 0; i < n; i++) {
        result[i] = a[i] < b[i];
    }
}

void CPU_greaterThan(float* a, float* b, int n, float* result) {
    for (int i = 0; i < n; i++) {
        result[i] = a[i] > b[i];
    }
}

void CPU_matmul(float* a, float* b, int m, int n, int k, float* result) {
    for (int i = 0; i < m; i++) {           // For each row of A
        for (int j = 0; j < n; j++) {       // For each column of B
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {   // Dot product
                sum += a[i * k + l] * b[l * n + j];
            }
            result[i * n + j] = sum;
        }
    }
}

void CPU_sin(float* a, int n, float* result) {
    for (int i = 0; i < n; i++) {
        result[i] = std::sin(a[i]);
    }
}

void CPU_cos(float* a, int n, float* result) {
    for (int i = 0; i < n; i++) {
        result[i] = std::cos(a[i]);
    }
}

void CPU_sum(Tensor* a, Tensor* result) {
    int n = a->size;
    float* aData = a->data;
    for (int i = 0; i < n; i++) {
        std::vector<int> aIndices = a->flatToIndices(i);
        result->idx(aIndices) += aData[i];
    }
}

void CPU_max(Tensor* a, Tensor* result) {
    int n = a->size;
    float* aData = a->data;
    for (int i = 0; i < n; i++) {
        std::vector<int> aIndices = a->flatToIndices(i);
        result->idx(aIndices) = std::max(result->idx(aIndices), aData[i]);
    }
}

void CPU_min(Tensor* a, Tensor* result) {
    int n = a->size;
    float* aData = a->data;
    for (int i = 0; i < n; i++) {
        std::vector<int> aIndices = a->flatToIndices(i);
        result->idx(aIndices) = std::min(result->idx(aIndices), aData[i]);
    }
}