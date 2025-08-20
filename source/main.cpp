extern "C" {
    #include <gba_console.h>
    #include <gba_video.h>
    #include <gba_interrupt.h>
    #include <gba_systemcalls.h>
}
#include <stdio.h>
#include "nn.hpp"


void goto_line (int line) {
    iprintf("\x1b[%d;0H", line);
}

int main(void) {
    irqInit();
    irqEnable(IRQ_VBLANK);
    consoleDemoInit();

    iprintf("Backpropagation example with XOR\n");

    double lr = 1e-2;

    Tensor d0({0.0f, 0.0f}, shape_t{1, 2}, false);
    Tensor d1({0.0f, 1.0f}, shape_t{1, 2}, false);
    Tensor d2({1.0f, 0.0f}, shape_t{1, 2}, false);
    Tensor d3({1.0f, 1.0f}, shape_t{1, 2}, false);

    Tensor l0({-1.0f}, shape_t{1, 1}, false);
    Tensor l1({1.0f}, shape_t{1, 1}, false);
    Tensor l2({1.0f}, shape_t{1, 1}, false);
    Tensor l3({-1.0f}, shape_t{1, 1}, false);

    // First Hidden layer with 3 output neurons
    Tensor W11 ({-1.0f, 0.1f}, shape_t{1, 2}, true);
    Tensor W12 ({0.5f, -0.5f}, shape_t{1, 2}, true);
    Tensor W13 ({0.3f, 0.8f}, shape_t{1, 2}, true);

    Tensor b11 ({0.02f}, shape_t{1, 1}, true);
    Tensor b12 ({3.0f}, shape_t{1, 1}, true);
    Tensor b13 ({-5.0f}, shape_t{1, 1}, true);

    Tensor W21 ({2.0f}, shape_t{1, 1}, true);
    Tensor W22 ({-0.9f}, shape_t{1, 1}, true);
    Tensor W23 ({3.0f}, shape_t{1, 1}, true);

    Tensor b21 ({3.0f}, shape_t{1, 1}, true);

    // Add to update list
    Tensor *update_list[] = {
        &W11, &W12, &W13,
        &b11, &b12, &b13,
        &W21, &W22, &W23,
        &b21
    };

    int step = 0;

    Tensor *datasets[] = {&d0, &d1, &d2, &d3};
    Tensor *labels[] = {&l0, &l1, &l2, &l3};

    while (1) {
        VBlankIntrWait();

        int correct = 0;
        float avg_loss = 0.0f;

        int predictions[4];

        // Train on dataset
        for (int i = 0; i < 4; ++i) {
            Tensor &input = *datasets[i];
            Tensor &target = *labels[i];

            // Forward pass

            // First layer
            Tensor x00 = nn::sum(input * W11) + b11;
            Tensor x01 = nn::sum(input * W12) + b12;
            Tensor x02 = nn::sum(input * W13) + b13;

            Tensor x10 = nn::relu(x00);
            Tensor x11 = nn::relu(x01);
            Tensor x12 = nn::relu(x02);

            // Second layer
            Tensor y00 = (x10 * W21 + x11 * W22 + x12 * W23) + b21;
            Tensor output = nn::sigmoid(y00);

            // Loss calculation
            Tensor loss = nn::sum((target - output) * (target - output));

            // Backpropagation
            loss.backward();
            for (Tensor *param : update_list) {
                param->update(lr);
            }
            loss.zero_grad();
            
            avg_loss += loss.data()[0];

            // print accuracy
            if ((target[0] > 0.0f && output[0] > 0.0f) ||
                (target[0] < 0.0f && output[0] < 0.0f)) {
                ++correct;
            }

            predictions[i] = (output[0] > 0.0f) ? 1 : 0;
        }

        goto_line(4);
        iprintf("Step: %d\n",step);
        printf("Average Loss: %f\n", avg_loss / 4);
        iprintf("Accuracy: %d/4\n", correct);

        // Print the predictions
        iprintf("Predictions: ");
        for (int i = 0; i < 4; ++i) {
            iprintf("%d, ", predictions[i]);
        }
        iprintf("\n");

        // Print out the first Weights
        iprintf("W1:\n");
        print_tensor(W11);
        print_tensor(W12);
        print_tensor(W13);
        iprintf("W2:\n");
        print_tensor(W21);
        print_tensor(W22);
        print_tensor(W23);

        ++step;
        Tensor_GC();    // Garbage collection
    }
}