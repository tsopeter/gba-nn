// tensor_ops.cpp
#include "nn.math.hpp"

extern "C" {
    #include <stdio.h>
}

// Helper: Q7.8 fixed-point multiply
inline int16_t q8_mul(int16_t a, int16_t b) {
    return (int32_t(a) * int32_t(b)) >> Q8_SHIFT;
}

// Helper: Q7.8 fixed-point divide (simple reciprocal method)
inline int16_t q8_div(int16_t a, int16_t b) {
    if (b == 0) return 0; // avoid div by zero
    return (int32_t(a) << Q8_SHIFT) / b;
}

Tensor add(const Tensor& a, const Tensor& b) {
    Tensor out;
    out.init(a.ndim, a.shape);
    for (uint16_t i = 0; i < a.size(); ++i)
        out[i] = a[i] + b[i];

    if (a.requires_grad || b.requires_grad) {
        out.set_requires_grad(true);
        auto back_fn = [](Tensor& self) {
            for (uint16_t i = 0; i < self.size(); ++i) {
                if (self.parent1 && self.parent1->requires_grad)
                    self.parent1->grad()[i] += self.grad()[i];
                if (self.parent2 && self.parent2->requires_grad)
                    self.parent2->grad()[i] += self.grad()[i];
            }
        };
        out._set_creator(back_fn, const_cast<Tensor*>(&a), const_cast<Tensor*>(&b));
    }
    return out;
}

Tensor sub(const Tensor& a, const Tensor& b) {
    Tensor out;
    out.init(a.ndim, a.shape);
    for (uint16_t i = 0; i < a.size(); ++i)
        out[i] = a[i] - b[i];

    if (a.requires_grad || b.requires_grad) {
        out.set_requires_grad(true);
        auto back_fn = [](Tensor& self) {
            for (uint16_t i = 0; i < self.size(); ++i) {
                if (self.parent1 && self.parent1->requires_grad)
                    self.parent1->grad()[i] += self.grad()[i];
                if (self.parent2 && self.parent2->requires_grad)
                    self.parent2->grad()[i] -= self.grad()[i];
            }
        };
        out._set_creator(back_fn, const_cast<Tensor*>(&a), const_cast<Tensor*>(&b));
    }
    return out;
}

Tensor mul(const Tensor& a, const Tensor& b) {
    Tensor out;
    out.init(a.ndim, a.shape);
    for (uint16_t i = 0; i < a.size(); ++i)
        out[i] = q8_mul(a[i], b[i]);

    if (a.requires_grad || b.requires_grad) {
        out.set_requires_grad(true);
        auto back_fn = [](Tensor& self) {
            for (uint16_t i = 0; i < self.size(); ++i) {
                if (self.parent1 && self.parent1->requires_grad)
                    self.parent1->grad()[i] += q8_mul(self.grad()[i], self.parent2->operator[](i));
                if (self.parent2 && self.parent2->requires_grad && self.parent1 != self.parent2)
                    self.parent2->grad()[i] += q8_mul(self.grad()[i], self.parent1->operator[](i));
            }
        };
        out._set_creator(back_fn, const_cast<Tensor*>(&a), const_cast<Tensor*>(&b));
    }
    return out;
}

Tensor mul(const Tensor& a, int16_t b_q8) {
    Tensor out;
    out.init(a.ndim, a.shape);

    for (uint16_t i = 0; i < a.size(); ++i)
        out[i] = q8_mul(a[i], b_q8);

    if (a.requires_grad) {
        out.set_requires_grad(true);
        auto back_fn = [b_q8](Tensor& self) {
            if (self.parent1 && self.parent1->requires_grad) {
                for (uint16_t i = 0; i < self.size(); ++i)
                    self.parent1->grad()[i] += q8_mul(self.grad()[i], b_q8);
            }
        };
        out._set_creator(back_fn, const_cast<Tensor*>(&a), nullptr);
    }

    return out;
}

Tensor mul(const Tensor& a, float b) {
    return mul(a, float_to_q8(b));
}

Tensor div(const Tensor& a, const Tensor& b) {
    Tensor out;
    out.init(a.ndim, a.shape);
    for (uint16_t i = 0; i < a.size(); ++i)
        out[i] = q8_div(a[i], b[i]);

    if (a.requires_grad || b.requires_grad) {
        out.set_requires_grad(true);
        auto back_fn = [](Tensor& self) {
            for (uint16_t i = 0; i < self.size(); ++i) {
                if (self.parent1 && self.parent1->requires_grad)
                    self.parent1->grad()[i] += q8_div(self.grad()[i], self.parent2->operator[](i));
                if (self.parent2 && self.parent2->requires_grad) {
                    int16_t a_val = self.parent1->operator[](i);
                    int16_t b_val = self.parent2->operator[](i);
                    int16_t dL_db = -q8_mul(self.grad()[i], q8_div(a_val, q8_mul(b_val, b_val)));
                    self.parent2->grad()[i] += dL_db;
                }
            }
        };
        out._set_creator(back_fn, const_cast<Tensor*>(&a), const_cast<Tensor*>(&b));
    }
    return out;
}

Tensor transpose(const Tensor& t) {
    if (t.ndim != 2) {
        Tensor dummy;
        return dummy;
    }

    uint16_t rows = t.dim(0);
    uint16_t cols = t.dim(1);
    uint16_t shape[2] = {cols, rows};
    Tensor out;
    out.init(2, shape);

    for (uint16_t i = 0; i < rows; ++i) {
        for (uint16_t j = 0; j < cols; ++j) {
            out[out.offset(j, i)] = t[t.offset(i, j)];
        }
    }

    // No autograd logic because transpose is a view-like op in most frameworks
    return out;
}

// Dot product with autograd for 2D @ 2D
Tensor dot(const Tensor& a, const Tensor& b) {
    if (a.ndim != 2 || b.ndim != 2 || a.dim(1) != b.dim(0)) {
        Tensor dummy;
        return dummy;
    }

    uint16_t m = a.dim(0);
    uint16_t n = a.dim(1);
    uint16_t p = b.dim(1);
    uint16_t shape[2] = {m, p};
    Tensor out;
    out.init(2, shape, 0);

    for (uint16_t i = 0; i < m; ++i) {
        for (uint16_t j = 0; j < p; ++j) {
            int32_t acc = 0;
            for (uint16_t k = 0; k < n; ++k)
                acc += q8_mul(a[a.offset(i, k)], b[b.offset(k, j)]);
            out[out.offset(i, j)] = (int16_t)acc;
        }
    }

    if (a.requires_grad || b.requires_grad) {
        out.set_requires_grad(true);
        auto back_fn = [](Tensor& self) {
            Tensor* a = self.parent1;
            Tensor* b = self.parent2;

            uint16_t m = a->dim(0);
            uint16_t n = a->dim(1);
            uint16_t p = b->dim(1);

            for (uint16_t i = 0; i < m; ++i) {
                for (uint16_t j = 0; j < p; ++j) {
                    int16_t grad_val = self.grad()[self.offset(i, j)];
                    for (uint16_t k = 0; k < n; ++k) {
                        if (a->requires_grad)
                            a->grad()[a->offset(i, k)] += q8_mul(grad_val, b->operator[](b->offset(k, j)));
                        if (b->requires_grad)
                            b->grad()[b->offset(k, j)] += q8_mul(grad_val, a->operator[](a->offset(i, k)));
                    }
                }
            }
        };
        out._set_creator(back_fn, const_cast<Tensor*>(&a), const_cast<Tensor*>(&b));
    }

    return out;
}