#include <stdio.h>
#include <iostream>
#include "nn.hpp"

int main(void) {
    Tensor a({2.0f, 2.0f}, shape_t{1, 2}, true);
    Tensor b({1.0f, 3.0f}, shape_t{2, 1}, true);

    // We want a -> b
    
    for (int i = 0; i < 1024; ++i) {
        Tensor c  = nn::npow(a ^ b, 2); // Matrix multiplication

        std::cout << c << '\n';

        //std::cout << "GC Count: " << Tensor_GC_Count() << std::endl;

        c.backward();

        a.update(0.01f);
        c.zero_grad();
        Tensor_GC();
    }

    printf("a: ");
    std::cout << a << '\n';
    printf("\n");
    return 0;
}