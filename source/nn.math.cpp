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
            //std::cout<<"SUM_Backward\n";
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

Tensor &nn::cat(Tensor& t1, Tensor& t2) {
    // Concat to last dimension
    if (t1.ndim() != t2.ndim()) {
        throw std::invalid_argument("Tensors must have the same number of dimensions to concatenate.");
    }
    shape_t new_shape = t1.shape();
    new_shape.back() += t2.shape().back();

    Tensor *out = new Tensor(new_shape, false, true);
    Tensor_GC_add(out);

    uint16_t size1 = t1.size();
    uint16_t size2 = t2.size();

    for (uint16_t i = 0; i < size1; ++i) {
        (*out)[i] = t1[i];
    }
    for (uint16_t i = 0; i < size2; ++i) {
        (*out)[size1 + i] = t2[i];
    }

    if (t1.requires_grad() || t2.requires_grad()) {
        out->set_requires_grad(true);
        auto back_fn = [=](Tensor& self) {
            std::cout << "CAT_Backward\n";
            if (self.get_parent1() && self.get_parent1()->requires_grad()) {
                for (uint16_t i = 0; i < t1.size(); ++i) {
                    self.get_parent1()->grad()[i] += self.grad()[i];
                }
            }
            if (self.get_parent2() && self.get_parent2()->requires_grad()) {
                for (uint16_t i = 0; i < t2.size(); ++i) {
                    self.get_parent2()->grad()[i] += self.grad()[t1.size() + i];
                }
            }
        };
        out->set_creator(back_fn, const_cast<Tensor*>(&t1), const_cast<Tensor*>(&t2));
    }

    return *out;
}

Tensor &nn::nsin(Tensor& t) {
    Tensor *out = new Tensor(t.shape(), false, true);
    Tensor_GC_add(out);

    for (uint16_t i = 0; i < t.size(); ++i) {
        (*out)[i] = static_cast<BASE_VALUE_TYPE>(sin(quant2float(t[i])));
    }

    if (t.requires_grad()) {
        out->set_requires_grad(true);
        auto back_fn = [=](Tensor& self) {
            std::cout << "NSIN_Backward\n";
            for (uint16_t i = 0; i < self.size(); ++i) {
                if (self.get_parent1() && self.get_parent1()->requires_grad()) {
                    BASE_VALUE_TYPE a_val = quant2float(self.get_parent1()->data()[i]);
                    BASE_VALUE_TYPE dL_da = cos(a_val) * self.grad()[i];
                    self.get_parent1()->grad()[i] += dL_da;
                }
            }
        };
        out->set_creator(back_fn, const_cast<Tensor*>(&t), nullptr);
    }

    return *out;
}

Tensor &nn::ncos(Tensor& t) {
    Tensor *out = new Tensor(t.shape(), false, true);
    Tensor_GC_add(out);

    for (uint16_t i = 0; i < t.size(); ++i) {
        (*out)[i] = static_cast<BASE_VALUE_TYPE>(cos(quant2float(t[i])));
    }

    if (t.requires_grad()) {
        out->set_requires_grad(true);
        std::cout<<"NCOS_Backward\n";
        auto back_fn = [=](Tensor& self) {
            for (uint16_t i = 0; i < self.size(); ++i) {
                if (self.get_parent1() && self.get_parent1()->requires_grad()) {
                    BASE_VALUE_TYPE a_val = quant2float(self.get_parent1()->data()[i]);
                    BASE_VALUE_TYPE dL_da = -sin(a_val) * self.grad()[i];
                    self.get_parent1()->grad()[i] += dL_da;
                }
            }
        };
        out->set_creator(back_fn, const_cast<Tensor*>(&t), nullptr);
    }

    return *out;
}

// helper function: wraps angle to [-PI, PI]
Tensor &nn::wrap_angle(Tensor &angle_diff) {
    static constexpr float PI = 3.14159265358979323846f;
    for (int i = 0; i < angle_diff.size(); ++i) {
        float a = quant2float(angle_diff[i]);
        while (a > PI)  a -= 2 * PI;
        while (a < -PI) a += 2 * PI;
        angle_diff[i] = a;
    }
    return angle_diff;
}

Tensor &nn::nrand(shape_t shape, bool requires_grad) {
    Tensor *out = new Tensor(shape, requires_grad, true);
    Tensor_GC_add(out);

    for (uint16_t i = 0; i < out->size(); ++i) {
        (*out)[i] = static_cast<BASE_VALUE_TYPE>(rand() % 256) / 255.0f; // Random float in [0, 1]
    }

    if (requires_grad) {
        out->set_requires_grad(true);
        auto back_fn = [](Tensor& self) {
            // No gradient for random tensors
        };
        out->set_creator(back_fn, nullptr, nullptr);
    }

    return *out;
}

Tensor &nn::loss::mse(Tensor& prediction, Tensor& target) {
    Tensor &out1 = (prediction - target) * (prediction - target);
    Tensor &out2 = nn::sum(out1);
    return out2;
}