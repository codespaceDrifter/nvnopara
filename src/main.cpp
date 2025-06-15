#include <iostream>
#include "tensor.hpp"
#include <vector>
#include "backend.hpp"

int main() {
    printBackend();

    float* data = new float[10];
    for (int i = 0; i < 10; i++) {
        data[i] = i;
    }

    Tensor tensor({2, 5});
    tensor.data = data;

    std::cout << tensor.idx(std::vector<int>{1, 2}) << std::endl;

    return 0;
}

