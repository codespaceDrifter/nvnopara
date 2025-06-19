#include <iostream>
#include <vector>
#include "tensor.hpp"
#include "op.hpp"
#include <chrono>




int main() {

    Tensor* a = new Tensor({1'000'000'000, 1});
    a->arrange(0,0.1);

    Tensor* b = new Tensor({1'000'000'000, 1});
    b->arrange(0,0.1);


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

    // CUDA minus test

    auto start3 = std::chrono::high_resolution_clock::now();

    Tensor* d_CUDA = sub(a, b);

    auto end3 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration3 = end3 - start3;
    std::cout << "Elapsed time for CUDA minus: " << duration3.count() << " seconds\n";

    delete d_CUDA;






    return 0;
}