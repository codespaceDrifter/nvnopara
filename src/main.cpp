#include <iostream>
#include "tensor.hpp"
#include <vector>
#include "backend.hpp"
#include "op.hpp"

int main() {
    printBackend();
    currentBackend = Backend::CPU;
    printBackend();



    return 0;
}

