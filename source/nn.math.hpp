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


namespace loss {

Tensor &mse (Tensor& prediction, Tensor& target);


}

}