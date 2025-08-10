extern "C" {
    #include <gba_console.h>
    #include <gba_video.h>
    #include <gba_interrupt.h>
    #include <gba_systemcalls.h>
}
#include <stdio.h>
#include "nn.hpp"

int main(void) {
    irqInit();
    irqEnable(IRQ_VBLANK);
    consoleDemoInit();

    iprintf("Backpropagation example\n");

    uint16_t shape[2] = {1, 1};

    Tensor a, b, c, d, e;

    a.initf(2, shape, 1.f);
    b.initf(2, shape, 3.f);
    e.initf(2, shape, 0.01f);
    e.set_requires_grad(true);
    for (int i = 0; i < 10; ++i) {
        a.set_requires_grad(true);
        b.set_requires_grad(true);

        a.zero_grad();
        b.zero_grad();

        c = sub(a, b);
        d = mul(c, c);

        d.backward();

        Tensor grad  = extract_grad(a);
        grad.set_requires_grad(false);
        Tensor nabla = mul(a, a);

        print_tensor(nabla);
        
    }
    print_tensor(a);

    while (1) {
        VBlankIntrWait();
    }
}