C++ library that does Tensor Calculations  
nvnopara stands for nano parallel: a minimum functionality library that stills allow GPU tensor ops
Targets CUDA or CPU  
this is meant for physics simulations or math this is NOT meant for Machine Learning
there are NO broadcasting
there is still shape to be passed in for Matmul and reduceops but normal element ops should NOT use stride calculations
there are NO views, every tensor memory is contiguous
no codegen/ laziness / fused operations for now just simple basic kernels  

used in my svmbolsolve, phvsicsim, and mvndspace libraries  