#include <iostream>
#include "tensor.hpp"
#include <vector>
#include <cmath>
#include "op.hpp"

int main() {
    bool allTestsPassed = true;
    
    // Create consistent tensors A and B outside all test blocks
    // A is (1,2) for broadcasting test, B is (2,2)
    Tensor* A = new Tensor({1,2});
    A->data[0] = 1.0f;
    A->data[1] = 2.0f;  // A = [1, 2]
    
    Tensor* B = new Tensor({2,2});
    B->data[0] = 3.0f;
    B->data[1] = 4.0f;
    B->data[2] = 5.0f;
    B->data[3] = 6.0f;  // B = [[3, 4], [5, 6]]
    
    // Move to CUDA once and keep them there
    A->toCUDA();
    B->toCUDA();
    
    // Test CUDA add: A + B
    {
        Tensor* expected = new Tensor({2,2});
        std::vector<float> expectedData = {4.0f, 6.0f, 6.0f, 8.0f};  // [[4, 6], [6, 8]]
        for (int i = 0; i < expectedData.size(); i++) {
            expected->data[i] = expectedData[i];
        }
        
        Tensor* result = add(A, B);
        result->toCPU();
        
        if (!result->equal(expected)) {
            std::cout << "Add test failed" << std::endl;
            allTestsPassed = false;
        }
        
        delete expected;
        delete result;
    }
    
    // Test CUDA subtract: B - A
    {
        Tensor* expected = new Tensor({2,2});
        std::vector<float> expectedData = {2.0f, 2.0f, 4.0f, 4.0f};  // [[2, 2], [4, 4]]
        for (int i = 0; i < expectedData.size(); i++) {
            expected->data[i] = expectedData[i];
        }
        
        Tensor* result = sub(B, A);
        result->toCPU();
        
        if (!result->equal(expected)) {
            std::cout << "Sub test failed" << std::endl;
            allTestsPassed = false;
        }
        
        delete expected;
        delete result;
    }
    
    // Test CUDA multiply: A * B
    {
        Tensor* expected = new Tensor({2,2});
        std::vector<float> expectedData = {3.0f, 8.0f, 5.0f, 12.0f};  // [[3, 8], [5, 12]]
        for (int i = 0; i < expectedData.size(); i++) {
            expected->data[i] = expectedData[i];
        }
        
        Tensor* result = mul(A, B);
        result->toCPU();
        
        if (!result->equal(expected)) {
            std::cout << "Mul test failed" << std::endl;
            allTestsPassed = false;
        }
        
        delete expected;
        delete result;
    }
    
    // Test CUDA divide: B / A
    {
        Tensor* expected = new Tensor({2,2});
        std::vector<float> expectedData = {3.0f, 2.0f, 5.0f, 3.0f};  // [[3, 2], [5, 3]]
        for (int i = 0; i < expectedData.size(); i++) {
            expected->data[i] = expectedData[i];
        }
        
        Tensor* result = div(B, A);
        result->toCPU();
        
        if (!result->equal(expected)) {
            std::cout << "Div test failed" << std::endl;
            allTestsPassed = false;
        }
        
        delete expected;
        delete result;
    }
    
    // Test CUDA power: A ^ B
    {
        Tensor* expected = new Tensor({2,2});
        std::vector<float> expectedData = {1.0f, 16.0f, 1.0f, 64.0f};  // [[1^3, 2^4], [1^5, 2^6]]
        for (int i = 0; i < expectedData.size(); i++) {
            expected->data[i] = expectedData[i];
        }
        
        Tensor* result = pow(A, B);
        result->toCPU();
        
        if (!result->equal(expected)) {
            std::cout << "Pow test failed" << std::endl;
            allTestsPassed = false;
        }
        
        delete expected;
        delete result;
    }
    
    // Test CUDA equal: A == B
    {
        Tensor* expected = new Tensor({2,2});
        std::vector<float> expectedData = {0.0f, 0.0f, 0.0f, 0.0f};  // All false
        for (int i = 0; i < expectedData.size(); i++) {
            expected->data[i] = expectedData[i];
        }
        
        Tensor* result = equal(A, B);
        result->toCPU();
        
        if (!result->equal(expected)) {
            std::cout << "Equal test failed" << std::endl;
            allTestsPassed = false;
        }
        
        delete expected;
        delete result;
    }
    
    // Test CUDA lessThan: A < B
    {
        Tensor* expected = new Tensor({2,2});
        std::vector<float> expectedData = {1.0f, 1.0f, 1.0f, 1.0f};  // All true
        for (int i = 0; i < expectedData.size(); i++) {
            expected->data[i] = expectedData[i];
        }
        
        Tensor* result = lessThan(A, B);
        result->toCPU();
        
        if (!result->equal(expected)) {
            std::cout << "LessThan test failed" << std::endl;
            allTestsPassed = false;
        }
        
        delete expected;
        delete result;
    }
    
    // Test CUDA greaterThan: B > A
    {
        Tensor* expected = new Tensor({2,2});
        std::vector<float> expectedData = {1.0f, 1.0f, 1.0f, 1.0f};  // All true
        for (int i = 0; i < expectedData.size(); i++) {
            expected->data[i] = expectedData[i];
        }
        
        Tensor* result = greaterThan(B, A);
        result->toCPU();
        
        if (!result->equal(expected)) {
            std::cout << "GreaterThan test failed" << std::endl;
            allTestsPassed = false;
        }
        
        delete expected;
        delete result;
    }
    
    // Test CUDA sin: sin(A)
    {
        Tensor* expected = new Tensor({1,2});
        std::vector<float> expectedData = {(float)sin(1.0f), (float)sin(2.0f)};
        for (int i = 0; i < expectedData.size(); i++) {
            expected->data[i] = expectedData[i];
        }
        
        Tensor* result = sin(A);
        result->toCPU();
        
        if (!result->equal(expected)) {
            std::cout << "Sin test failed" << std::endl;
            allTestsPassed = false;
        }
        
        delete expected;
        delete result;
    }
    
    // Test CUDA cos: cos(B)
    {
        Tensor* expected = new Tensor({2,2});
        std::vector<float> expectedData = {(float)cos(3.0f), (float)cos(4.0f), (float)cos(5.0f), (float)cos(6.0f)};
        for (int i = 0; i < expectedData.size(); i++) {
            expected->data[i] = expectedData[i];
        }
        
        Tensor* result = cos(B);
        result->toCPU();
        
        if (!result->equal(expected)) {
            std::cout << "Cos test failed" << std::endl;
            allTestsPassed = false;
        }
        
        delete expected;
        delete result;
    }
    
    // Test CUDA matmul: A.T @ B (need to create proper matrices for matmul)
    {
        // Create 2x2 matrices for matmul test
        Tensor* matA = new Tensor({2,2});
        matA->data[0] = 1.0f; matA->data[1] = 2.0f;
        matA->data[2] = 3.0f; matA->data[3] = 4.0f;  // [[1, 2], [3, 4]]
        
        Tensor* matB = new Tensor({2,2});  
        matB->data[0] = 5.0f; matB->data[1] = 6.0f;
        matB->data[2] = 7.0f; matB->data[3] = 8.0f;  // [[5, 6], [7, 8]]
        
        Tensor* expected = new Tensor({2,2});
        std::vector<float> expectedData = {19.0f, 22.0f, 43.0f, 50.0f};  // [[19, 22], [43, 50]]
        for (int i = 0; i < expectedData.size(); i++) {
            expected->data[i] = expectedData[i];
        }
        
        matA->toCUDA();
        matB->toCUDA();
        
        Tensor* result = matmul(matA, matB);
        result->toCPU();
        
        if (!result->equal(expected)) {
            std::cout << "Matmul test failed" << std::endl;
            allTestsPassed = false;
        }
        
        delete matA;
        delete matB;
        delete expected;
        delete result;
    }
    
    // Clean up main tensors
    delete A;
    delete B;
    
    if (allTestsPassed) {
        std::cout << "All tests passed" << std::endl;
    } else {
        std::cout << "Some tests failed" << std::endl;
    }
    
    return 0;
}