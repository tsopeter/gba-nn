#include "nn.math.hpp"
#include <cmath>
#include <iostream>

Tensor &nn::sum(Tensor& t) {
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

Tensor &nn::npow(Tensor& t, float exponent) {
    Tensor *out = new Tensor(t.shape(), false, true);
    Tensor_GC_add(out);

    BASE_VALUE_TYPE exp_quant = float2quant(exponent);
    for (uint16_t i = 0; i < t.size(); ++i) {
        (*out)[i] = static_cast<BASE_VALUE_TYPE>(pow(quant2float(t[i]), exponent));
    }

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

Tensor &nn::nexp(Tensor& t) {
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

Tensor &nn::relu(Tensor& t) {
    Tensor *out = new Tensor(t.shape(), false, true);
    Tensor_GC_add(out);

    for (uint16_t i = 0; i < t.size(); ++i) {
        (*out)[i] = std::max(quant2float(t[i]), 0.0f);
    }

    if (t.requires_grad()) {
        out->set_requires_grad(true);
        auto back_fn = [=](Tensor& self) {
            for (uint16_t i = 0; i < self.size(); ++i) {
                if (self.get_parent1() && self.get_parent1()->requires_grad()) {
                    BASE_VALUE_TYPE a_val = quant2float(self.get_parent1()->data()[i]);
                    BASE_VALUE_TYPE dL_da = (a_val > 0) ? self.grad()[i] : 0;
                    self.get_parent1()->grad()[i] += dL_da;
                }
            }
        };
        out->set_creator(back_fn, const_cast<Tensor*>(&t), nullptr);
    }

    return *out;
}

Tensor &nn::sigmoid(Tensor& t, float alpha) {
    Tensor *out = new Tensor(t.shape(), false, true);
    Tensor_GC_add(out);

    for (uint16_t i = 0; i < t.size(); ++i) {
        BASE_VALUE_TYPE a_val = quant2float(t[i]);
        (*out)[i] = static_cast<BASE_VALUE_TYPE>(1.0f / (1.0f + exp(-alpha * a_val)));
    }

    if (t.requires_grad()) {
        out->set_requires_grad(true);
        auto back_fn = [=](Tensor& self) {
            for (uint16_t i = 0; i < self.size(); ++i) {
                if (self.get_parent1() && self.get_parent1()->requires_grad()) {
                    BASE_VALUE_TYPE sigmoid_val = quant2float(self[i]);
                    BASE_VALUE_TYPE dL_da = sigmoid_val * (1 - sigmoid_val) * alpha * self.grad()[i];
                    self.get_parent1()->grad()[i] += dL_da;
                }
            }
        };
        out->set_creator(back_fn, const_cast<Tensor*>(&t), nullptr);
    }

    return *out;
}

Tensor &nn::loss::mse(Tensor& prediction, Tensor& target) {
    Tensor &out1 = (prediction - target) * (prediction - target);
    Tensor &out2 = nn::sum(out1);
    return out2;
}