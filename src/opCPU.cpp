#include "opCPU.hpp"

void CPU_add(int n, float* a, float* b, float* result) {
    for (int i = 0; i < n; i++) {
        result[i] = a[i] + b[i];
    }
}

void CPU_sub(int n, float* a, float* b, float* result) {
    for (int i = 0; i < n; i++) {
        result[i] = a[i] - b[i];
    }
}

void CPU_mul(int n, float* a, float* b, float* result) {
    for (int i = 0; i < n; i++) {
        result[i] = a[i] * b[i];
    }
}

void CPU_div(int n, float* a, float* b, float* result) {
    for (int i = 0; i < n; i++) {
        result[i] = a[i] / b[i];
    }
}

void CPU_pow(int n, float* a, float* b, float* result) {
    for (int i = 0; i < n; i++) {
        result[i] = std::pow(a[i], b[i]);
    }
}

void CPU_equal(int n, float* a, float* b, float* result) {
    for (int i = 0; i < n; i++) {
        result[i] = a[i] == b[i];
    }
}

void CPU_lessThan(int n, float* a, float* b, float* result) {
    for (int i = 0; i < n; i++) {
        result[i] = a[i] < b[i];
    }
}

void CPU_greaterThan(int n, float* a, float* b, float* result) {
    for (int i = 0; i < n; i++) {
        result[i] = a[i] > b[i];
    }
}

void CPU_matmul(int m, int n, int k, float* a, float* b, float* result) {
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

void CPU_sin(int n, float* a, float* result) {
    for (int i = 0; i < n; i++) {
        result[i] = std::sin(a[i]);
    }
}

void CPU_cos(int n, float* a, float* result) {
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