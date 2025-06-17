C++ library that does Tensor Calculations  
nvnopara stands for nano parallel: a minimum functionality library that stills allow GPU tensor ops
Targets CUDA or CPU  
this is meant for physics simulations or math this is NOT meant for Machine Learning
there are NO broadcasting
there is still shape to be passed in for Matmul and reduceops but normal element ops should NOT use stride calculations
there are NO views, every tensor memory is contiguous
no codegen/ laziness / fused operations for now just simple basic kernels  

used in my svmbolsolve, phvsicsim, and mvndspace libraries  


#to do

spend whole week on this make this actually good and worthy of being actually used in other projects (phvsicsim)

kernel fusion without codegen. use sth like

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


allow broadcasting, but do it with checks like a bool passed in whether to use broadcasting or not

tiled matmul