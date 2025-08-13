#include <stdio.h>
#include <iostream>
#include "nn.hpp"

int main(void) {
    Tensor a({0.0f, 0.0f}, shape_t{2}, true);
    Tensor b({4.0f, 5.0f}, shape_t{2}, false);

    // print b
    printf("b: ");
    print_tensor(b);
    printf("\n");

    // We want a -> b
    
    for (int i = 0; i < 1024; ++i) {
        Tensor d = (a - b) * (a - b);

        //std::cout << "GC Count: " << Tensor_GC_Count() << std::endl;

        d.backward();

        a.update(0.1f);
        d.zero_grad();
        Tensor_GC();
    }

    // print a
    printf("a: ");
    std::cout << a << '\n';
    printf("\n");
    return 0;
}