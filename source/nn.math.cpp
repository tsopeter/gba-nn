#include "nn.math.hpp"
#include <cmath>
#include <iostream>

Tensor nn::sum(Tensor& t) {
    Tensor *out = new Tensor(
        {1}, false, true
    );
    Tensor_GC_add(out);

    BASE_VALUE_TYPE sum = 0;
    for (uint16_t i = 0; i < t.size(); ++i) {
        sum += t[i];
    }
    (*out)[0] = sum;

    if (t.requires_grad()) {
        out->set_requires_grad(true);
        auto back_fn = [=](Tensor& self) {
            if (!self.get_parent1()) return;
            for (uint16_t i = 0; i < self.get_parent1()->size(); ++i) {
                if (self.get_parent1() && self.get_parent1()->requires_grad())
                    self.get_parent1()->grad()[i] += self.grad()[0];
            }
        };
        out->set_creator(back_fn, const_cast<Tensor*>(&t), nullptr);
    }

    return *out;
}

Tensor nn::npow(Tensor& t, float exponent) {
    Tensor *out = new Tensor(t.shape(), false, true);
    Tensor_GC_add(out);

    BASE_VALUE_TYPE exp_quant = float2quant(exponent);
    for (uint16_t i = 0; i < t.size(); ++i) {
        (*out)[i] = static_cast<BASE_VALUE_TYPE>(pow(quant2float(t[i]), exponent));
    }

    std::cout << "exponent: " << exponent << ", exp_quant: " << exp_quant << '\n';

    if (t.requires_grad()) {
        out->set_requires_grad(true);
        auto back_fn = [=](Tensor& self) {
            for (uint16_t i = 0; i < self.size(); ++i) {
                if (self.get_parent1() && self.get_parent1()->requires_grad()) {
                    BASE_VALUE_TYPE a_val = quant2float(self.get_parent1()->data()[i]);
                    BASE_VALUE_TYPE dL_da = self.grad()[i] * exp_quant * pow(a_val, exp_quant - 1);
                    self.get_parent1()->grad()[i] += dL_da;
                }
            }
        };
        out->set_creator(back_fn, const_cast<Tensor*>(&t), nullptr);
    }

    return *out;
}

Tensor nn::nexp(Tensor& t) {
    Tensor *out = new Tensor(t.shape(), false, true);
    Tensor_GC_add(out);

    for (uint16_t i = 0; i < t.size(); ++i) {
        (*out)[i] = static_cast<BASE_VALUE_TYPE>(exp(quant2float(t[i])));
    }

    if (t.requires_grad()) {
        out->set_requires_grad(true);
        auto back_fn = [=](Tensor& self) {
            for (uint16_t i = 0; i < self.size(); ++i) {
                if (self.get_parent1() && self.get_parent1()->requires_grad()) {
                    BASE_VALUE_TYPE a_val = quant2float(self.get_parent1()->data()[i]);
                    BASE_VALUE_TYPE dL_da = self.grad()[i] * exp(a_val);
                    self.get_parent1()->grad()[i] += dL_da;
                }
            }
        };
        out->set_creator(back_fn, const_cast<Tensor*>(&t), nullptr);
    }

    return *out;
}