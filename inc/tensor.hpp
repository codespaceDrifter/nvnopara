#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>


class Tensor {
public:
    Tensor(std::vector<int> shape);
    

    // do NOT use these shape indexes for normal element OPs
    inline __attribute__((always_inline)) std::vector<int> flatToIndices (int data_idx){
        std::vector<int> result;
        int cur_group = data_idx;
        for (int i = this->shape.size() - 1; i >= 0; --i){
            result.insert(result.begin(), cur_group % this->shape[i]);
            cur_group = cur_group / this->shape[i];
        }
        return result;
    }

    inline __attribute__((always_inline)) float& idx (std::vector<int> indices_vec){
        indices_vec.erase(indices_vec.begin(), indices_vec.begin() + indices_vec.size() - this->shape.size());
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


    ~Tensor();
    
    float* data;
    std::vector<int> shape;
    int size;
    std::vector<int> stride;
};

#endif // TENSOR_HPP