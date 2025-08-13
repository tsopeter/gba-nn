#include <stdio.h>
#include <iostream>
#include "nn.hpp"

int main(void) {
    Tensor a({0.0f, 0.0f}, shape_t{2}, true);

    // We want a -> b
    
    for (int i = 0; i < 10'000; ++i) {
        Tensor c  = nn::npow((a - 10), 2.0f);

        std::cout << c << '\n';

        //std::cout << "GC Count: " << Tensor_GC_Count() << std::endl;

        c.backward();

        a.update(0.05f);
        c.zero_grad();
        Tensor_GC();
    }

    printf("a: ");
    std::cout << a << '\n';
    printf("\n");
    return 0;
}