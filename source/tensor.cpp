#include "tensor.hpp"
extern "C" {
    #include <stdio.h>  // for iprintf
}

// Fixed-point conversion
int16_t float_to_q8(float x) {
    return (int16_t)(x * (1 << Q8_SHIFT));
}

float q8_to_float(int16_t x) {
    return (float)x / (1 << Q8_SHIFT);
}

// Constructor
Tensor::Tensor()
    : ndim(0), total_size(0)
{
    for (int i = 0; i < MAX_DIMS; ++i)
        shape[i] = 0;

    for (int i = 0; i < MAX_TENSOR_SIZE; ++i)
        data[i] = 0;
}

void Tensor::init(uint16_t dims, const uint16_t* shape_in, int16_t fill) {
    ndim = dims;
    total_size = 1;

    for (int i = 0; i < dims; ++i) {
        shape[i] = shape_in[i];
        total_size *= shape[i];
    }

    for (uint16_t i = 0; i < total_size; ++i)
        data[i] = fill;

    zero_grad();
}

void Tensor::initf(uint16_t dims, const uint16_t* shape_in, float fill) {
    int16_t f = float_to_q8(fill);
    this->init(dims, shape_in, f);
}

int16_t& Tensor::operator[](uint16_t i) {
    return data[i];
}

int16_t Tensor::operator[](uint16_t i) const {
    return data[i];
}

uint16_t Tensor::offset(uint16_t i, uint16_t j, uint16_t k, uint16_t l) const {
    if (ndim == 4)
        return ((i * shape[1] + j) * shape[2] + k) * shape[3] + l;
    else if (ndim == 3)
        return (i * shape[1] + j) * shape[2] + k;
    else if (ndim == 2)
        return i * shape[1] + j;
    else
        return i;
}

uint16_t Tensor::size() const {
    return total_size;
}

uint16_t Tensor::dim(uint16_t i) const {
    return shape[i];
}

void Tensor::zero_grad() {
    for (uint16_t i = 0; i < size(); ++i)
        grad_buf[i] = 0;
}

void Tensor::set_requires_grad(bool req) {
    requires_grad = req;
}

int16_t* Tensor::grad() {
    return grad_buf;
}

void Tensor::_set_creator(std::function<void(Tensor&)> back_fn, Tensor* p1, Tensor* p2) {
    backward_fn = back_fn;
    parent1 = p1;
    parent2 = p2;
}

void Tensor::backward(bool first) {
    if (!requires_grad) return;

    if (first) {
        for (uint16_t i = 0; i < size(); ++i)
            grad_buf[i] = float_to_q8(1.0f);
    }

    if (backward_fn)
        backward_fn(*this);

    if (parent1 != nullptr) {
        if (parent1->requires_grad)
            parent1->backward(false);
    }
    if (parent2 != nullptr) {
        if (parent2->requires_grad)
            parent2->backward(false);
    }
}

// Prints a 1D or 2D tensor
void print_tensor(const Tensor& t) {
    if (t.size() > 1024) {
        iprintf("Tensor too large to print\n");
        return;
    }

    if (t.dim(0) == 0) {
        iprintf("Empty tensor\n");
        return;
    }

    if (t.size() == 0) {
        iprintf("Tensor size 0\n");
        return;
    }

    if (t.dim(1) == 0 || t.ndim == 1) {
        // 1D tensor
        for (uint16_t i = 0; i < t.size(); ++i) {
            int16_t val = t[i];
            int integer = val >> 8;           // 3
            int fractional = (val & 0xFF) * 100 / 256;  // ≈ 14
            iprintf("%d.%02d ", integer, fractional);
        }
        iprintf("\n");
    } else if (t.ndim == 2) {
        // 2D tensor
        uint16_t H = t.dim(0);
        uint16_t W = t.dim(1);
        for (uint16_t i = 0; i < H; ++i) {
            for (uint16_t j = 0; j < W; ++j) {
                uint16_t idx = t.offset(i, j);
                int16_t val = t[idx];
                int integer = val >> 8;           // 3
                int fractional = (val & 0xFF) * 100 / 256;  // ≈ 14

                iprintf("%d.%02d ", integer, fractional);
            }
            iprintf("\n");
        }
    } else {
        iprintf("Tensor ndim > 2 not supported\n");
    }
}

void print_tensor_grad(Tensor& t) {
    iprintf("Gradients:\n");
    if (t.ndim == 1) {
        for (uint16_t i = 0; i < t.size(); ++i) {
            int16_t val = t.grad()[i];
            int integer = val >> Q8_SHIFT;
            int frac = (val & 0xFF) * 100 / 256;
            iprintf("%d.%02d ", integer, frac);
        }
        iprintf("\n");
    } else if (t.ndim == 2) {
        uint16_t H = t.dim(0);
        uint16_t W = t.dim(1);
        for (uint16_t i = 0; i < H; ++i) {
            for (uint16_t j = 0; j < W; ++j) {
                int16_t val = t.grad()[t.offset(i, j)];
                int integer = val >> Q8_SHIFT;
                int frac = (val & 0xFF) * 100 / 256;
                iprintf("%d.%02d ", integer, frac);
            }
            iprintf("\n");
        }
    } else {
        iprintf("ndim > 2 not supported\n");
    }
}

Tensor extract_grad(Tensor& t) {
    if (!t.requires_grad) {
        Tensor empty;
        return empty;
    }

    Tensor grad_tensor;
    grad_tensor.init(t.ndim, t.shape);  // same shape as t

    for (uint16_t i = 0; i < t.size(); ++i)
        grad_tensor[i] = t.grad()[i];

    return grad_tensor;
}