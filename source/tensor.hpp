#pragma once

#include <stdint.h>
#include <functional>

#define MAX_DIMS        4
#define MAX_TENSOR_SIZE 1024
#define Q8_SHIFT        8

// Fixed-point conversion
int16_t float_to_q8(float x);
float q8_to_float(int16_t x);

class Tensor {
public:
    Tensor();

    void init (uint16_t dims, const uint16_t* shape_in, int16_t fill = 0);
    void initf(uint16_t dims, const uint16_t* shape_in, float fill = 0.f);

    int16_t& operator[](uint16_t i);
    int16_t  operator[](uint16_t i) const;

    uint16_t offset(uint16_t i, uint16_t j = 0, uint16_t k = 0, uint16_t l = 0) const;

    uint16_t size() const;
    uint16_t dim(uint16_t i) const;

    uint16_t shape[MAX_DIMS];
    uint16_t ndim;
    uint16_t total_size;
    int16_t data[MAX_TENSOR_SIZE];

    // Backpropagation
    void zero_grad();
    void set_requires_grad(bool req);
    void backward(bool first=true);

    int16_t* grad();  // gradient pointer

    // Only called by autograd engine
    void _set_creator(std::function<void(Tensor&)>, Tensor* parent1, Tensor* parent2 = nullptr);

    bool requires_grad = false;
    int16_t grad_buf[MAX_TENSOR_SIZE] = {0};

    // Autograd fields
    std::function<void(Tensor&)> backward_fn;
    Tensor* parent1 = nullptr;
    Tensor* parent2 = nullptr;
};

Tensor extract_grad(Tensor& t);

void print_tensor(const Tensor& t);
void print_tensor_grad(Tensor& t);