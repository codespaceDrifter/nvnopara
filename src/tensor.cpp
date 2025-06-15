#include "tensor.hpp"

Tensor::Tensor(std::vector<int> shape) {
    this->shape = shape;

    this->size = 1;
    for (int i = 0; i < shape.size(); i++) {
        this->size *= shape[i];
    }

    this->data = new float[this->size];

    this->stride = std::vector<int>(shape.size());
    this->stride[shape.size() - 1] = 1;
    for (int i = shape.size() - 2; i >= 0; i--) {
        if (shape[i] == 1) {
            this->stride[i] = 0;
        } else {
            this->stride[i] = this->stride[i + 1] * shape[i + 1];
        }
    }
}



Tensor::~Tensor() {
    delete[] this->data;
}