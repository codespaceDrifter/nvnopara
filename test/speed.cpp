#include <iostream>
#include <vector>
#include "tensor.hpp"
#include "op.hpp"
#include <chrono>




int main() {

    Tensor* a = new Tensor({1'000, 1'000});
    a->randomize(0, 10);

    Tensor* b = new Tensor({1'000, 1'000});
    b->randomize(0, 10);

    auto start = std::chrono::high_resolution_clock::now();

    Tensor* c_CPU = add(a, b);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Elapsed time: " << duration.count() << " seconds\n";

    a->toCUDA();
    b->toCUDA();

    auto start2 = std::chrono::high_resolution_clock::now();

    Tensor* c_CUDA = add(a, b);

    auto end2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration2 = end2 - start2;
    std::cout << "Elapsed time: " << duration2.count() << " seconds\n";






    return 0;
}