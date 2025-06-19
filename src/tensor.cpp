#include "tensor.hpp"


Tensor::Tensor(int* shape, int dim) {
    this->dim = dim;
    
    // Allocate memory for shape and stride arrays
    this->shape = new int[this->dim];
    this->stride = new int[this->dim];
    
    // Copy shape data
    for (int i = 0; i < this->dim; i++) {
        this->shape[i] = shape[i];
    }

    this->size = 1;
    for (int i = 0; i < this->dim; i++) {
        this->size *= this->shape[i];
    }

    this->data = new float[this->size]();

    // Calculate stride
    this->stride[this->dim - 1] = 1;
    for (int i = this->dim - 2; i >= 0; i--) {
        this->stride[i] = this->stride[i + 1] * this->shape[i + 1];
    }
    for (int i = 0; i < this->dim; i++) {
        if (this->shape[i] == 1) {
            this->stride[i] = 0;
        }
    }
    
    this->device = Device::CPU;
    this->d_shape = nullptr;
    this->d_stride = nullptr;
}


Tensor::~Tensor() {
    if (this->device == Device::CPU) {
        delete[] this->data;
        delete[] this->shape;
        delete[] this->stride;
    } else if (this->device == Device::CUDA) {
        cudaFree(this->data);
        delete[] this->shape;  // CPU copies
        delete[] this->stride;
        if (this->d_shape) cudaFree(this->d_shape);    // GPU copies
        if (this->d_stride) cudaFree(this->d_stride);
    }
}

void Tensor::arrange(float start, float step) {
    for (int i = 0; i < this->size; ++i){
        this->data[i] = start + i * step;
    }
}

void Tensor::randomize(float min, float max) {
    for (int i = 0; i< this->size; ++i){
        float cur = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) * (max - min) +min;
        this->data[i] = cur;
    }
}

void Tensor::toCPU() {
    switch (this->device) {
        case Device::CPU:
            break;
        case Device::CUDA:
            // Move data back to CPU
            float* tempData = new float[this->size];
            cudaMemcpy(tempData, this->data, this->size * sizeof(float), cudaMemcpyDeviceToHost);
            cudaFree(this->data);
            this->data = tempData;
            
            // Clean up GPU shape and stride (CPU copies already exist)
            if (this->d_shape) {
                cudaFree(this->d_shape);
                this->d_shape = nullptr;
            }
            if (this->d_stride) {
                cudaFree(this->d_stride);
                this->d_stride = nullptr;
            }
            
            break;
    }

    this->device = Device::CPU;
}

void Tensor::toCUDA() {
    if (this->device == Device::CPU) {
        // Move data to CUDA
        float* cudaData;
        cudaMalloc(&cudaData, this->size * sizeof(float));
        cudaMemcpy(cudaData, this->data, this->size * sizeof(float), cudaMemcpyHostToDevice);
        delete[] this->data;
        this->data = cudaData;
        
        // Create GPU copies of shape and stride (keep CPU copies)
        cudaMalloc(&this->d_shape, this->dim * sizeof(int));
        cudaMemcpy(this->d_shape, this->shape, this->dim * sizeof(int), cudaMemcpyHostToDevice);
        
        cudaMalloc(&this->d_stride, this->dim * sizeof(int));
        cudaMemcpy(this->d_stride, this->stride, this->dim * sizeof(int), cudaMemcpyHostToDevice);
    }
    this->device = Device::CUDA;
}

void Tensor::print(){
    assert(this->device == Device::CPU);
    //outputs the shape
    std::cout << "shape: (";
    for (int i = 0; i < this->dim; ++i) {
        std::cout << this->shape[i];
        if (i != this->dim - 1) std::cout << ", ";
    }
    std::cout <<")"<<std::endl;

    //outputs the formatted data
    std::cout <<"data: "<<std::endl;

    std::string result;

    if (this->dim == 0 ) result = "";

    // makes all the last dimension of the data into "[1,2,3]" formatted strings and put them in lineVec
    std::vector<std::string> lineVec;
    std::string tempStr;
    int lastShape = this->shape[this->dim-1];
    for (int i = 0; i < this->size; ++i){
        if (lastShape == 1 || i != 0 && (i+1) % lastShape == 0){
            tempStr += std::to_string(this->data[i]);
            tempStr = "[" + tempStr + "]";
            lineVec.push_back(tempStr);
            tempStr = "";
        } else {
            tempStr = tempStr + std::to_string(this->data[i]) + ", ";
        }
    }

    //formats the data to be according to a indented tensor format
    //the last two dimensions are like a matrix form
    //each dimension before that except the first divides the tensor and creates indents
    //uses max divide to not create a extra bracket at the end
    for (int i = this->dim - 2; i > 0; --i){

        int curDivide = this->shape[i];
        int curCount = 0;
        int curMaxDivides = 1;
        int curDivideTimes = 1;

        for (int z = 0; z < i; ++z){
            curMaxDivides *= this->shape[z];
        }

        for (int j = 0; j < lineVec.size(); ++j) lineVec[j] = "  " + lineVec[j];

        for (int j = 0; j < lineVec.size(); ++j){
            if (i == this->dim -2) curCount++;
            else if (lineVec[j] == "  ]") curCount++;
            if (curCount / curDivide == curDivideTimes && curCount % curDivide == 0 && curDivideTimes < curMaxDivides){
                ++curDivideTimes;
                lineVec.insert(lineVec.begin()+j+1, "]");
                j++;
                lineVec.insert(lineVec.begin()+j+1, "[");
                j++;
            }
        }
        lineVec.insert(lineVec.begin(), "[");
        lineVec.push_back("]");
    }

    for (int i = 0; i < lineVec.size(); ++i){
        result = result + lineVec[i] + "\n";
    }

    std::cout << result << std::endl;
}

bool Tensor::equal(Tensor* other) {
    assert(this->device == Device::CPU);
    assert(other->device == Device::CPU);
    
    if (this->dim != other->dim) return false;
    if (this->size != other->size) return false;
    
    for (int i = 0; i < this->dim; i++) {
        if (this->shape[i] != other->shape[i]) return false;
    }
    
    for (int i = 0; i < this->size; i++) {
        if (abs(this->data[i] - other->data[i]) > 1e-6) return false;
    }
    
    return true;
}


