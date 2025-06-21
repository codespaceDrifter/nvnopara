#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cassert>
#include <cuda_runtime.h>

enum class Device {
    CPU,
    CUDA,
};

//seed the RNG
static int _ = (srand(time(nullptr)),0);

class Tensor {
public:

    Tensor(int* shape, int dim);
    Tensor(std::initializer_list<int> shape)
        : Tensor(const_cast<int*>(shape.begin()), static_cast<int>(shape.size())) {}
    

    // do NOT use these shape indexes for normal element OPs
    inline __attribute__((always_inline)) std::vector<int> flatToIndices (int data_idx){
        std::vector<int> result;
        int cur_group = data_idx;
        for (int i = this->dim - 1; i >= 0; --i){
            result.insert(result.begin(), cur_group % this->shape[i]);
            cur_group = cur_group / this->shape[i];
        }
        return result;
    }

    inline __attribute__((always_inline)) float& idx (std::vector<int> indices_vec){
        indices_vec.erase(indices_vec.begin(), indices_vec.begin() + indices_vec.size() - this->dim);
        int data_idx = 0;
        for (int i = 0; i < indices_vec.size(); ++i){
            data_idx += indices_vec[i] * this->stride[i];
        }
        return this->data[data_idx];
    }

    template <typename... Indices>
    inline __attribute__((always_inline)) float& idx (Indices... indices){
        std::vector<int> indices_vec = {indices...};
        return this->idx(indices_vec);
    }


    void toCPU();
    void toCUDA();

    void transpose(int dim1, int dim2);
    void squeeze (int dimIdx);
    void unsqueeze (int dimIdx);

    void randomize(float min, float max);
    void arrange(float start = 0, float step = 1);
    void print();
    bool equal(Tensor* other);

    ~Tensor();
    
    float* data;
    int size;
    int* shape;    // CPU shape
    int* stride;   // CPU stride
    int dim;
    Device device;
};

#endif // TENSOR_HPP