C++ library that does Tensor Calculations  
nvnopara stands for nano parallel: a minimum functionality library that stills allow GPU tensor ops
Targets CUDA or CPU  
there are NO views, every tensor owns its own memory 
there are NO codegen. element op kernels are fused with abstract syntax tree and switch op statements. 

used in my svmbolsolve, phvsicsim, and mvndspace libraries  


#to do  

0: fix transpose, fix the matmulElemul function and speed test

1: kernel fusion  
__global__ void fusedKernel(float* inputs[], float* result, int n, int* ops, int numOps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float temp = inputs[0][idx];
        
        for(int i = 0; i < numOps; i++) {
            switch(ops[i]) {
                case ADD: temp += inputs[i+1][idx]; break;
                case MUL: temp *= inputs[i+1][idx]; break;
                case SIN: temp = sinf(temp); break;
                case COS: temp = cosf(temp); break;
                // etc...
            }
        }
        
        result[idx] = temp;
    }
}


2: tiled matmul  

3: specialized functions i.e. mesh collision