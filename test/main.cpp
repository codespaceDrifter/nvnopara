#include <iostream>
#include "tensor.hpp"
#include <vector>
#include "op.hpp"


//add this line at the start
//srand(time(nullptr));
int main() {

    Tensor* a = new Tensor({2,2});
    a->arrange();
    a->print();

    Tensor* b = new Tensor({2,2});
    b->arrange();
    b->print();

    Tensor* c = matmul(a, b);
    c->print();

    a->toCUDA();
    b->toCUDA();

    Tensor* d = matmul(a, b);
    d->toCPU();
    d->print();



    return 0;
}

