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

    Tensor a({2.0f, 1.0f}, shape_t{2}, true);
    Tensor b({4.0f, -1.0f}, shape_t{2}, false);

    iprintf("Initial Tensor a:\n");
    print_tensor(a);

    iprintf("Tensor b:\n");
    print_tensor(b);

    iprintf("Loss function\nL=sum((a-b)^2)\n");

    int step = 0;
    while (1) {
        VBlankIntrWait();

        Tensor c = nn::sum((a - b) * (a - b));
        c.backward();
        a.update(0.01f);
        c.zero_grad();

        iprintf("\x1b[7;0H");  
        iprintf("Step: %d Tensor a:\n", step);
        iprintf("\x1b[8;0H");
        print_tensor(a);

        ++step;
    }
}