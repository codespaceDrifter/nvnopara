#include "backend.hpp"


void printBackend() {
    switch (currentBackend) {
        case Backend::CPU:
            std::cout << "Backend: CPU\n"; break;
        case Backend::CUDA:
            std::cout << "Backend: CUDA\n"; break;
        default:
            std::cout << "Backend: Unknown\n"; break;
    }
}

Backend detectBackend() {
    if (cudaGetDeviceCount(nullptr) > 0) {
        return Backend::CUDA;
    }
    return Backend::CPU;
}

Backend currentBackend = detectBackend();