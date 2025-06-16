#ifndef DEVICE_HPP
#define DEVICE_HPP

#include <iostream>
#include <cuda_runtime.h>

enum class Device {
    CPU,
    CUDA,
};

Device detectDevice();

void printDevice();

extern Device currentDevice;

#endif // DEVICE_HPP
