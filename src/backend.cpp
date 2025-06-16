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
    //seed the random number generator, not related to the backend
    srand(time(nullptr));

    //check if cuda available
    if (cudaGetDeviceCount(nullptr) > 0) {
        return Backend::CUDA;
    }
    return Backend::CPU;
}

Backend currentBackend = detectBackend();