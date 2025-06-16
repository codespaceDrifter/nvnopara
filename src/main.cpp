#include <iostream>
#include "tensor.hpp"
#include <vector>
#include "device.hpp"
#include "op.hpp"

int main() {
    printDevice();
    currentDevice = Device::CPU;
    printDevice();



    return 0;
}

