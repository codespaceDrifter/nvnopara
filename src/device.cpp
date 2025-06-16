#include "device.hpp"


void printDevice() {
    switch (currentDevice) {
        case Device::CPU:
            std::cout << "Device: CPU\n"; break;
        case Device::CUDA:
            std::cout << "Device: CUDA\n"; break;
        default:
            std::cout << "Device: Unknown\n"; break;
    }
}

Device detectDevice() {
    //seed the random number generator, not related to the device
    srand(time(nullptr));

    //check if cuda available
    if (cudaGetDeviceCount(nullptr) > 0) {
        return Device::CUDA;
    }
    return Device::CPU;
}

Device currentDevice = detectDevice();
