#include "tensor.hpp"
#include <iostream>

BASE_VALUE_TYPE float2quant(float value) {
    // Convert float to Q7.8 fixed-point representation
    return static_cast<BASE_VALUE_TYPE>(value * BASE_VALUE_FRAC);
}

float quant2float(BASE_VALUE_TYPE value) {
    // Convert Q7.8 fixed-point representation to float
    return static_cast<float>(value) / BASE_VALUE_FRAC;
}

Tensor::Tensor() {
    // Default constructor
    ndim_ = 0;
    shape_.clear();
    data_.clear();
    requires_grad_ = false;
    grad_.clear();
    parent1_ = nullptr;
    parent2_ = nullptr;
}

Tensor::Tensor(const Tensor& other) {
    // Copy constructor
    ndim_ = other.ndim_;
    shape_ = other.shape_;
    data_ = other.data_;
    requires_grad_ = other.requires_grad_;
    grad_ = other.grad_;
    backward_fn_ = other.backward_fn_;
    parent1_ = other.parent1_;
    parent2_ = other.parent2_;
}

Tensor::Tensor(std::vector<float> data, shape_t shape) {
    // Constructor with data and shape
    ndim_ = shape.size();
    shape_ = shape;
    data_.resize(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        data_[i] = float2quant(data[i]);
    }
    requires_grad_ = false;
    grad_.resize(size(), 0);
    parent1_ = nullptr;
    parent2_ = nullptr;
}

Tensor::Tensor(std::vector<float> data, shape_t shape, bool requires_grad) {
    // Constructor with data, shape, and requires_grad flag
    ndim_ = shape.size();
    shape_ = shape;
    data_.resize(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        data_[i] = float2quant(data[i]);
    }
    requires_grad_ = requires_grad;
    if (requires_grad_) {
        grad_.resize(size(), 0);
    } else {
        grad_.clear();
    }
    parent1_ = nullptr;
    parent2_ = nullptr;
}

Tensor Tensor::operator+(const Tensor& other) const {
    return implt_operator_add_i(other);
}

Tensor Tensor::operator-(const Tensor& other) const {
    return implt_operator_sub_i(other);
}

Tensor Tensor::operator*(const Tensor& other) const {
    return implt_operator_mul_i(other);
}

Tensor Tensor::operator/(const Tensor& other) const {
    return implt_operator_div_i(other);
}

void print_tensor(Tensor& t) {
    for (const auto& val : t.data()) {
        printf("%f ", quant2float(val));
    }
    printf("\n");
}

void print_tensor_grad(Tensor& t) {
    for (const auto& grad_val : t.grad()) {
        printf("%f ", quant2float(grad_val));
    }
    printf("\n");
}

void Tensor::_set_creator(std::function<void(Tensor&)> fn, Tensor* parent1, Tensor* parent2) {
    backward_fn_ = std::move(fn);
    parent1_ = parent1;
    parent2_ = parent2;
}

void Tensor::set_requires_grad(bool requires_grad) {
    requires_grad_ = requires_grad;
    if (requires_grad_) {
        grad_.resize(size(), 0);
    }
}

uint16_t Tensor::size() const {
    uint16_t size = 1;
    for (const auto& dim : shape_) {
        size *= dim;
    }
    return size;
}

BASE_VALUE_TYPE& Tensor::operator[](size_t index) {
    return data_[index];
}

const BASE_VALUE_TYPE& Tensor::operator[](size_t index) const {
    return data_[index];
}

uint16_t Tensor::dim(uint16_t index) const {
    return shape_[index];
}

uint16_t Tensor::ndim() const {
    return ndim_;
}

void Tensor::backward(bool start) {
    if (start) {
        for (auto& g : grad()) g = float2quant(1.0f);  // seed with ∂L/∂c = 1
    }

    // calling the backward function if it exists
    if (backward_fn_) {
        backward_fn_(*this);
    }

    if (parent1_) parent1_->backward(false);
    if (parent2_) parent2_->backward(false);
}

void Tensor::zero_grad() {
    if (requires_grad_) {
        std::fill(grad_.begin(), grad_.end(), 0);
    }

    // zero's gradients of previous nodes in the graph
    if (parent1_) parent1_->zero_grad();
    if (parent2_) parent2_->zero_grad();
}

BASE_VALUE_VEC& Tensor::grad() {
    if (!requires_grad_) {
        throw std::runtime_error("Gradient not available for this tensor.");
    }
    return grad_;
}

bool Tensor::requires_grad() const {
    return requires_grad_;
}

BASE_VALUE_VEC& Tensor::data() {
    return data_;
}

void Tensor::update(float lr) {
    if (!requires_grad_) {
        throw std::runtime_error("Cannot update tensor without gradients.");
    }
    for (size_t i = 0; i < data_.size(); ++i) {
        data_[i] -= static_cast<BASE_VALUE_TYPE>(quant2float(grad_[i]) * lr);
    }
}

/// Math implementation
Tensor Tensor::implt_operator_add_i(const Tensor& other) const {
    Tensor out;

    out.ndim_ = ndim_;
    out.shape_ = shape_;
    out.data_.resize(size());

    for (uint16_t i = 0; i < size(); ++i) {
        out[i] = data_[i] + other.data_[i];
    }

    if (requires_grad_ || other.requires_grad_) {
        out.set_requires_grad(true);
        auto back_fn = [](Tensor& self) {
            for (uint16_t i = 0; i < self.size(); ++i) {
                if (self.parent1_ && self.parent1_->requires_grad_)
                    self.parent1_->grad()[i] += self.grad()[i];
                if (self.parent2_ && self.parent2_->requires_grad_)
                    self.parent2_->grad()[i] += self.grad()[i];
            }
        };
        out._set_creator(back_fn, const_cast<Tensor*>(&other), const_cast<Tensor*>(this));
    }
    return out;
}

Tensor Tensor::implt_operator_sub_i(const Tensor& other) const {
    Tensor out;

    out.ndim_ = ndim_;
    out.shape_ = shape_;
    out.data_.resize(size());

    for (uint16_t i = 0; i < size(); ++i) {
        out[i] = data_[i] - other.data_[i];
    }

    if (requires_grad_ || other.requires_grad_) {
        out.set_requires_grad(true);
        auto back_fn = [](Tensor& self) {
            for (uint16_t i = 0; i < self.size(); ++i) {
                if (self.parent1_ && self.parent1_->requires_grad_)
                    self.parent1_->grad()[i] -= self.grad()[i];
                if (self.parent2_ && self.parent2_->requires_grad_)
                    self.parent2_->grad()[i] += self.grad()[i];
            }
        };
        out._set_creator(back_fn, const_cast<Tensor*>(&other), const_cast<Tensor*>(this));
    }
    return out;
}

Tensor Tensor::implt_operator_mul_i(const Tensor& other) const {
    Tensor out;

    out.ndim_ = ndim_;
    out.shape_ = shape_;
    out.data_.resize(size());

    for (uint16_t i = 0; i < size(); ++i) {
        // Q7.8 fixed-point multiplication: (a * b) >> 8
        out[i] = static_cast<BASE_VALUE_TYPE>((static_cast<BASE_VALUE_TYPE_2>(data_[i]) * other.data_[i]) * static_cast<BASE_VALUE_TYPE_2>(BASE_VALUE_FRAC));
    }

    if (requires_grad_ || other.requires_grad_) {
        out.set_requires_grad(true);
        auto back_fn = [](Tensor& self) {
            for (uint16_t i = 0; i < self.size(); ++i) {
                if (self.parent1_ && self.parent1_->requires_grad_)
                    self.parent1_->grad()[i] += (self.grad()[i] * self.parent2_->operator[](i)) / static_cast<BASE_VALUE_TYPE>(BASE_VALUE_FRAC);
                if (self.parent2_ && self.parent2_->requires_grad_ && self.parent1_ != self.parent2_)
                    self.parent2_->grad()[i] += (self.grad()[i] * self.parent1_->operator[](i)) / static_cast<BASE_VALUE_TYPE>(BASE_VALUE_FRAC);
            }
        };
        out._set_creator(back_fn, const_cast<Tensor*>(&other), const_cast<Tensor*>(this));
    }
    return out;
}

Tensor Tensor::implt_operator_div_i(const Tensor& other) const {
    Tensor out;

    out.ndim_ = ndim_;
    out.shape_ = shape_;
    out.data_.resize(size());

    for (uint16_t i = 0; i < size(); ++i) {
        // Q7.8 fixed-point division: (a << 8) / b
        if (other.data_[i] != 0) {
            out[i] = static_cast<BASE_VALUE_TYPE>((static_cast<BASE_VALUE_TYPE_2>(data_[i]) * static_cast<BASE_VALUE_TYPE_2>(BASE_VALUE_FRAC)) / other.data_[i]);
        } else {
            out[i] = 0; // or handle division by zero as needed
        }
    }

    if (requires_grad_ || other.requires_grad_) {
        out.set_requires_grad(true);
        auto back_fn = [](Tensor& self) {
            for (uint16_t i = 0; i < self.size(); ++i) {
                if (self.parent1_ && self.parent1_->requires_grad_) {
                    // Fixed-point division: (a << 8) / b
                    BASE_VALUE_TYPE grad_val = self.grad()[i];
                    BASE_VALUE_TYPE b_val = self.parent2_->operator[](i);
                    BASE_VALUE_TYPE dL_da = (b_val != 0) ? ((grad_val * static_cast<BASE_VALUE_TYPE_2>(BASE_VALUE_FRAC)) / b_val) : 0;
                    self.parent1_->grad()[i] += dL_da;
                }
                if (self.parent2_ && self.parent2_->requires_grad_) {
                    BASE_VALUE_TYPE a_val = self.parent1_->operator[](i);
                    BASE_VALUE_TYPE b_val = self.parent2_->operator[](i);
                    // dL/db = -grad * (a / (b * b)), fixed-point
                    BASE_VALUE_TYPE_2 b_sq = static_cast<BASE_VALUE_TYPE_2>(b_val) * b_val;
                    BASE_VALUE_TYPE dL_db = 0;
                    if (b_sq != 0) {
                        BASE_VALUE_TYPE_2 num = static_cast<BASE_VALUE_TYPE_2>(self.grad()[i]) * a_val;
                        dL_db = -static_cast<BASE_VALUE_TYPE>((num * static_cast<BASE_VALUE_TYPE_2>(BASE_VALUE_FRAC)) / b_sq);
                    }
                    self.parent2_->grad()[i] += dL_db;
                }
            }
        };
        out._set_creator(back_fn, const_cast<Tensor*>(this), const_cast<Tensor*>(&other));
    }
    return out;
}
