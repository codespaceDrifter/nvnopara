#include "tensor.hpp"

Tensor::Tensor(std::vector<int> shape) {
    this->shape = shape;

    this->size = 1;
    for (int i = 0; i < shape.size(); i++) {
        this->size *= shape[i];
    }

    this->data = new float[this->size]();

    this->stride = std::vector<int>(shape.size());
    this->stride[shape.size() - 1] = 1;
    for (int i = shape.size() - 2; i >= 0; i--) {
        this->stride[i] = this->stride[i + 1] * shape[i + 1];
    }
    for (int i = 0; i < shape.size(); i++) {
        if (shape[i] == 1) {
            this->stride[i] = 0;
        }
    }
}

Tensor::~Tensor() {
    delete[] this->data;
}

void Tensor::randomize(float min, float max) {
    for (int i = 0; i< this->size; ++i){
        float cur = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) * (max - min) +min;
        this->data[i] = cur;
    }
}


void Tensor::print(){
    //outputs the shape
    std::cout << "shape: (";
    for (int i = 0; i < this->shape.size(); ++i) {
        std::cout << this->shape[i];
        if (i != this->shape.size() - 1) std::cout << ", ";
    }
    std::cout <<")"<<std::endl;

    //outputs the formatted data
    std::cout <<"data: "<<std::endl;

    std::string result;

    if (this->shape.size() == 0 ) result = "";

    // makes all the last dimension of the data into "[1,2,3]" formatted strings and put them in lineVec
    std::vector<std::string> lineVec;
    std::string tempStr;
    int lastShape = this->shape[this->shape.size()-1];
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
    for (int i = this->shape.size() - 2; i > 0; --i){

        int curDivide = this->shape[i];
        int curCount = 0;
        int curMaxDivides = 1;
        int curDivideTimes = 1;

        for (int z = 0; z < i; ++z){
            curMaxDivides *= this->shape[z];
        }

        for (int j = 0; j < lineVec.size(); ++j) lineVec[j] = "  " + lineVec[j];

        for (int j = 0; j < lineVec.size(); ++j){
            if (i == this->shape.size() -2) curCount++;
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


Tensor* Tensor::broadcast(std::vector<int> newShape){
    std::vector<int> thisPadded = this->shape;
    std::vector<int> newPadded = newShape; 
    thisPadded.insert(thisPadded.begin(), std::max(0, static_cast<int>(newPadded.size() - thisPadded.size()) ), 1);
    newPadded.insert(newPadded.begin(), std::max(0, static_cast<int>(thisPadded.size() - newPadded.size())),1);

    for (int i = 0; i < thisPadded.size(); i++){
        assert(thisPadded[i] == newPadded[i] || thisPadded[i] == 1);
    }

    Tensor* result = new Tensor(newShape);

    for (int i = 0; i < result->size; i++){
        std::vector<int> indices = result->flatToIndices(i);
        result->data[i] = this->idx(indices);
    }
    return result;
}
