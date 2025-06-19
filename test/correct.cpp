#include <iostream>
#include "tensor.hpp"
#include <vector>
#include "op.hpp"

int main() {
    bool allTestsPassed = true;
    
    // Test CUDA add
    {
        Tensor* a = new Tensor({2,2});
        a->arrange(0, 1);  // [0, 1, 2, 3]
        
        Tensor* b = new Tensor({2,2});
        b->arrange(1, 1);  // [1, 2, 3, 4]
        
        // Expected result: [1, 3, 5, 7]
        Tensor* expected = new Tensor({2,2});
        expected->arrange(1, 2);
        
        // Move to CUDA
        a->toCUDA();
        b->toCUDA();
        
        // Perform CUDA add
        Tensor* result = add(a, b);
        result->toCPU();
        
        // Test
        if (!result->equal(expected)) {
            allTestsPassed = false;
        }
        
        delete a;
        delete b;
        delete expected;
        delete result;
    }
    
    // Test CUDA subtract
    {
        Tensor* a = new Tensor({2,2});
        a->arrange(2, 1);  // [2, 3, 4, 5]
        
        Tensor* b = new Tensor({2,2});
        b->arrange(0, 1);  // [0, 1, 2, 3]
        
        // Expected result: [2, 2, 2, 2]
        Tensor* expected = new Tensor({2,2});
        for (int i = 0; i < expected->size; i++) {
            expected->data[i] = 2.0f;
        }
        
        // Move to CUDA
        a->toCUDA();
        b->toCUDA();
        
        // Perform CUDA subtract
        Tensor* result = sub(a, b);
        result->toCPU();
        
        // Test
        if (!result->equal(expected)) {
            allTestsPassed = false;
        }
        
        delete a;
        delete b;
        delete expected;
        delete result;
    }
    
    // Test CUDA multiply
    {
        Tensor* a = new Tensor({2,2});
        a->arrange(1, 1);  // [1, 2, 3, 4]
        
        Tensor* b = new Tensor({2,2});
        b->arrange(2, 1);  // [2, 3, 4, 5]
        
        // Expected result: [2, 6, 12, 20]
        Tensor* expected = new Tensor({2,2});
        expected->data[0] = 2.0f;
        expected->data[1] = 6.0f;
        expected->data[2] = 12.0f;
        expected->data[3] = 20.0f;
        
        // Move to CUDA
        a->toCUDA();
        b->toCUDA();
        
        // Perform CUDA multiply
        Tensor* result = mul(a, b);
        result->toCPU();
        
        // Test
        if (!result->equal(expected)) {
            allTestsPassed = false;
        }
        
        delete a;
        delete b;
        delete expected;
        delete result;
    }
    
    // Test CUDA divide
    {
        Tensor* a = new Tensor({2,2});
        a->arrange(4, 2);  // [4, 6, 8, 10]
        
        Tensor* b = new Tensor({2,2});
        b->arrange(2, 1);  // [2, 3, 4, 5]
        
        // Expected result: [2, 2, 2, 2]
        Tensor* expected = new Tensor({2,2});
        for (int i = 0; i < expected->size; i++) {
            expected->data[i] = 2.0f;
        }
        
        // Move to CUDA
        a->toCUDA();
        b->toCUDA();
        
        // Perform CUDA divide
        Tensor* result = div(a, b);
        result->toCPU();
        
        // Test
        if (!result->equal(expected)) {
            allTestsPassed = false;
        }
        
        delete a;
        delete b;
        delete expected;
        delete result;
    }
    
    // Test CUDA power
    {
        Tensor* a = new Tensor({2,2});
        a->arrange(2, 1);  // [2, 3, 4, 5]
        
        Tensor* b = new Tensor({2,2});
        for (int i = 0; i < b->size; i++) {
            b->data[i] = 2.0f;  // [2, 2, 2, 2]
        }
        
        // Expected result: [4, 9, 16, 25]
        Tensor* expected = new Tensor({2,2});
        expected->data[0] = 4.0f;
        expected->data[1] = 9.0f;
        expected->data[2] = 16.0f;
        expected->data[3] = 25.0f;
        
        // Move to CUDA
        a->toCUDA();
        b->toCUDA();
        
        // Perform CUDA power
        Tensor* result = pow(a, b);
        result->toCPU();
        
        // Test
        if (!result->equal(expected)) {
            allTestsPassed = false;
        }
        
        delete a;
        delete b;
        delete expected;
        delete result;
    }
    
    // Test CUDA equal
    {
        Tensor* a = new Tensor({2,2});
        a->arrange(1, 1);  // [1, 2, 3, 4]
        
        Tensor* b = new Tensor({2,2});
        b->data[0] = 1.0f;
        b->data[1] = 5.0f;
        b->data[2] = 3.0f;
        b->data[3] = 7.0f;  // [1, 5, 3, 7]
        
        // Expected result: [1, 0, 1, 0]
        Tensor* expected = new Tensor({2,2});
        expected->data[0] = 1.0f;
        expected->data[1] = 0.0f;
        expected->data[2] = 1.0f;
        expected->data[3] = 0.0f;
        
        // Move to CUDA
        a->toCUDA();
        b->toCUDA();
        
        // Perform CUDA equal
        Tensor* result = equal(a, b);
        result->toCPU();
        
        // Test
        if (!result->equal(expected)) {
            allTestsPassed = false;
        }
        
        delete a;
        delete b;
        delete expected;
        delete result;
    }
    
    // Test CUDA lessThan
    {
        Tensor* a = new Tensor({2,2});
        a->arrange(1, 1);  // [1, 2, 3, 4]
        
        Tensor* b = new Tensor({2,2});
        b->data[0] = 2.0f;
        b->data[1] = 2.0f;
        b->data[2] = 5.0f;
        b->data[3] = 3.0f;  // [2, 2, 5, 3]
        
        // Expected result: [1, 0, 1, 0]
        Tensor* expected = new Tensor({2,2});
        expected->data[0] = 1.0f;
        expected->data[1] = 0.0f;
        expected->data[2] = 1.0f;
        expected->data[3] = 0.0f;
        
        // Move to CUDA
        a->toCUDA();
        b->toCUDA();
        
        // Perform CUDA lessThan
        Tensor* result = lessThan(a, b);
        result->toCPU();
        
        // Test
        if (!result->equal(expected)) {
            allTestsPassed = false;
        }
        
        delete a;
        delete b;
        delete expected;
        delete result;
    }
    
    // Test CUDA greaterThan
    {
        Tensor* a = new Tensor({2,2});
        a->arrange(1, 1);  // [1, 2, 3, 4]
        
        Tensor* b = new Tensor({2,2});
        b->data[0] = 2.0f;
        b->data[1] = 2.0f;
        b->data[2] = 2.0f;
        b->data[3] = 3.0f;  // [2, 2, 2, 3]
        
        // Expected result: [0, 0, 1, 1]
        Tensor* expected = new Tensor({2,2});
        expected->data[0] = 0.0f;
        expected->data[1] = 0.0f;
        expected->data[2] = 1.0f;
        expected->data[3] = 1.0f;
        
        // Move to CUDA
        a->toCUDA();
        b->toCUDA();
        
        // Perform CUDA greaterThan
        Tensor* result = greaterThan(a, b);
        result->toCPU();
        
        // Test
        if (!result->equal(expected)) {
            allTestsPassed = false;
        }
        
        delete a;
        delete b;
        delete expected;
        delete result;
    }
    
    // Test CUDA matmul
    {
        Tensor* a = new Tensor({2,2});
        a->arrange(1, 1);  // [1, 2, 3, 4]
        
        Tensor* b = new Tensor({2,2});
        b->arrange(1, 1);  // [1, 2, 3, 4]
        
        // Expected result: [7, 10, 15, 22] for matrix multiplication
        Tensor* expected = new Tensor({2,2});
        expected->data[0] = 7.0f;
        expected->data[1] = 10.0f;
        expected->data[2] = 15.0f;
        expected->data[3] = 22.0f;
        
        // Move to CUDA
        a->toCUDA();
        b->toCUDA();
        
        // Perform CUDA matmul
        Tensor* result = matmul(a, b);
        result->toCPU();
        
        // Test
        if (!result->equal(expected)) {
            allTestsPassed = false;
        }
        
        delete a;
        delete b;
        delete expected;
        delete result;
    }
    
    // Test CUDA sin
    {
        Tensor* a = new Tensor({2,2});
        a->data[0] = 0.0f;
        a->data[1] = 1.5708f;  // pi/2
        a->data[2] = 3.1416f;  // pi
        a->data[3] = 4.7124f;  // 3*pi/2
        
        // Expected result: [0, 1, 0, -1]
        Tensor* expected = new Tensor({2,2});
        expected->data[0] = 0.0f;
        expected->data[1] = 1.0f;
        expected->data[2] = 0.0f;
        expected->data[3] = -1.0f;
        
        // Move to CUDA
        a->toCUDA();
        
        // Perform CUDA sin
        Tensor* result = sin(a);
        result->toCPU();
        
        // Test (using looser tolerance for trigonometric functions)
        for (int i = 0; i < result->size; i++) {
            if (abs(result->data[i] - expected->data[i]) > 1e-3) {
                allTestsPassed = false;
                break;
            }
        }
        
        delete a;
        delete expected;
        delete result;
    }
    
    // Test CUDA cos
    {
        Tensor* a = new Tensor({2,2});
        a->data[0] = 0.0f;
        a->data[1] = 1.5708f;  // pi/2
        a->data[2] = 3.1416f;  // pi
        a->data[3] = 4.7124f;  // 3*pi/2
        
        // Expected result: [1, 0, -1, 0]
        Tensor* expected = new Tensor({2,2});
        expected->data[0] = 1.0f;
        expected->data[1] = 0.0f;
        expected->data[2] = -1.0f;
        expected->data[3] = 0.0f;
        
        // Move to CUDA
        a->toCUDA();
        
        // Perform CUDA cos
        Tensor* result = cos(a);
        result->toCPU();
        
        // Test (using looser tolerance for trigonometric functions)
        for (int i = 0; i < result->size; i++) {
            if (abs(result->data[i] - expected->data[i]) > 1e-3) {
                allTestsPassed = false;
                break;
            }
        }
        
        delete a;
        delete expected;
        delete result;
    }
    
    if (allTestsPassed) {
        std::cout << "All tests passed" << std::endl;
    } else {
        std::cout << "Some tests failed" << std::endl;
    }
    
    return 0;
}

