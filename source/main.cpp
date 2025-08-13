#include <stdio.h>
#include <iostream>
#include "nn.hpp"

int main(void) {
    auto a = std::make_shared<Tensor>(std::vector<float>{2.0f}, shape_t{2}, true);
    auto b = std::make_shared<Tensor>(std::vector<float>{4.0f}, shape_t{2}, false);

    printf("b: ");
    print_tensor(*b);
    printf("\n");

    {
        auto c_ = a->operator-(b);
    }

    for (int i = 0; i < 1; ++i) {

        auto c = (a->operator-(b))->operator*(a->operator-(b));
        c->backward();

        printf("grad a: ");
        print_tensor_grad(*a);
        printf("\n");

        a->update(0.1f);
        c->zero_grad();
    }

    printf("a: ");
    print_tensor(*a);
    printf("\n");

    return 0;
}