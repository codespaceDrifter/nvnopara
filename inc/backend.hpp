#ifndef BACKEND_HPP
#define BACKEND_HPP

#include <iostream>
#include <cuda_runtime.h>

enum class Backend {
    CPU,
    CUDA,
};

Backend detectBackend();

void printBackend();

extern Backend currentBackend;

#endif // BACKEND_HPP