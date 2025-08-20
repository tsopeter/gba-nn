#pragma once

#include "tensor.hpp"

/**
 * @brief nn
 * 
 * Neural network operations
 * This header file provides functions for common neural network operations
 * such as summation and power operations on tensors.
 */
namespace nn {

/**
 * @brief Sum of a tensor
 * 
 * Computes the sum of all elements in the tensor.
 */
Tensor &sum     (Tensor&);
Tensor &npow    (Tensor&, float exponent);
Tensor &nexp    (Tensor&);
Tensor &relu    (Tensor&);
Tensor &sigmoid (Tensor&, float alpha=1.0f);
Tensor &cat     (Tensor& t1, Tensor& t2);
Tensor &nsin     (Tensor& t);
Tensor &ncos     (Tensor& t);
Tensor &wrap_angle (Tensor&);
Tensor &nrand (shape_t shape, bool requires_grad=false);


namespace loss {

Tensor &mse (Tensor& prediction, Tensor& target);


}

}