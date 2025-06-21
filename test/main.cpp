#include <iostream>
#include "tensor.hpp"
#include <vector>
#include "op.hpp"


//add this line at the start
//srand(time(nullptr));
int main() {

    Tensor* a = new Tensor({2,2});
    a->arrange();

    Tensor* b = new Tensor({2,2});
    b->arrange();


    a->toCUDA();
    b->toCUDA();

    Tensor* c = matmul(a, b);




    Tensor* d = matmulByElemul(a, b);



    delete a;
    delete b;
    delete c;
    delete d;


    return 0;
}

