#ifndef nn_math_hpp__
#define nn_math_hpp__

#include "tensor.hpp"

Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);

// scalar
Tensor mul(const Tensor& a, float);
Tensor mul(const Tensor& a, int16_t);

Tensor div(const Tensor& a, const Tensor& b);
Tensor dot(const Tensor& a, const Tensor& b);
Tensor transpose(const Tensor& t);

#endif
