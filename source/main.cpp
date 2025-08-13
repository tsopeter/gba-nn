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
    
    for (int i = 0; i < 10'000; ++i) {
        Tensor c0 = a - b;
        Tensor c2 = c0 * c0;

        c2.backward();

        a.update(0.1f);
        c2.zero_grad();

    }

    // print a
    printf("a: ");
   std::cout << a << '\n';
    printf("\n");

    return 0;
}