#include "tensor.hpp"
#include <iostream>
#include <cassert>

static TensorResourceTracker _tensor_gc_internal_0134;

void Tensor_GC() {
    _tensor_gc_internal_0134.cleanup_tensors();
}

uint64_t Tensor_GC_Count() {
    return _tensor_gc_internal_0134.count_uncleaned();
}   

TensorResourceTracker::TensorResourceTracker() {}
TensorResourceTracker::~TensorResourceTracker() {
    cleanup_tensors();
}

void TensorResourceTracker::add_tensor_to_uncleaned(Tensor* tensor) {
    uncleaned_tensors_.push_back(tensor);
}

void TensorResourceTracker::cleanup_tensors() {
    //std::cout<<"Cleaning up tensors: " << uncleaned_tensors_.size() << std::endl;
    for (auto* tensor : uncleaned_tensors_) {
        if (!is_already_cleaned(tensor)) {
            cleaned_tensors_.push_back(tensor);
            delete tensor; // Clean up the tensor
        }
    }
    uncleaned_tensors_.clear();
    cleaned_tensors_.clear();
}

uint64_t TensorResourceTracker::count_uncleaned() const {
    return uncleaned_tensors_.size();
}

bool TensorResourceTracker::is_already_cleaned(Tensor* tensor) const {
    for (const auto* cleaned_tensor : cleaned_tensors_) {        
        if (cleaned_tensor == tensor) {
            return true; // Tensor has already been cleaned
        }
    }
    return false; // Tensor has not been cleaned
}

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

    children_.clear();
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

    children_ = other.children_;
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

    children_.clear();
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

    children_.clear();
}

Tensor::~Tensor() {

}

Tensor &Tensor::operator+(const Tensor& other) {
    return implt_operator_add_i(other);
}

Tensor &Tensor::operator+(float scalar) {
    return implt_operator_add_i(scalar);
}

Tensor &Tensor::operator-(const Tensor& other) {
    return implt_operator_sub_i(other);
}

Tensor &Tensor::operator-(float scalar) {
    return implt_operator_sub_i(scalar);
}

Tensor &Tensor::operator*(const Tensor& other) {
    return implt_operator_mul_i(other);
}

Tensor &Tensor::operator*(float scalar) {
    return implt_operator_mul_i(scalar);
}

Tensor &Tensor::operator/(const Tensor& other) {
    return implt_operator_div_i(other);
}

Tensor &Tensor::operator/(float scalar) {
    return implt_operator_div_i(scalar);
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
    //std::cout << "zerograd \n";
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
Tensor &Tensor::implt_operator_add_i(const Tensor& other) {
    Tensor *out = new Tensor();
    _tensor_gc_internal_0134.add_tensor_to_uncleaned(out);

    out->ndim_ = ndim_;
    out->shape_ = shape_;
    out->data_.resize(size());
    out->heap_allocated_ = true;

    for (uint16_t i = 0; i < size(); ++i) {
        (*out)[i] = data_[i] + other.data_[i];
    }

    if (requires_grad_ || other.requires_grad_) {
        out->set_requires_grad(true);
        auto back_fn = [](Tensor& self) {
            for (uint16_t i = 0; i < self.size(); ++i) {
                if (self.parent1_ && self.parent1_->requires_grad_)
                    self.parent1_->grad()[i] += self.grad()[i];
                if (self.parent2_ && self.parent2_->requires_grad_)
                    self.parent2_->grad()[i] += self.grad()[i];
            }
        };
        out->_set_creator(back_fn, const_cast<Tensor*>(&other), const_cast<Tensor*>(this));
    }

    return *out;
}

Tensor &Tensor::implt_operator_sub_i(const Tensor& other) {
    Tensor *out = new Tensor();
    _tensor_gc_internal_0134.add_tensor_to_uncleaned(out);

    out->ndim_ = ndim_;
    out->shape_ = shape_;
    out->data_.resize(size());
    out->heap_allocated_ = true;

    for (uint16_t i = 0; i < size(); ++i) {
        (*out)[i] = data_[i] - other.data_[i];
    }

    if (requires_grad_ || other.requires_grad_) {
        out->set_requires_grad(true);
        auto back_fn = [](Tensor& self) {
            //std::cout << "backward sub\n";
            for (uint16_t i = 0; i < self.size(); ++i) {
                if (self.parent1_ && self.parent1_->requires_grad_)
                    self.parent1_->grad()[i] -= self.grad()[i];
                if (self.parent2_ && self.parent2_->requires_grad_)
                    self.parent2_->grad()[i] += self.grad()[i];
            }
        };
        out->_set_creator(back_fn, const_cast<Tensor*>(&other), const_cast<Tensor*>(this));
    }

    return *out;
}

Tensor &Tensor::implt_operator_mul_i(const Tensor& other) {
    Tensor *out = new Tensor();
    _tensor_gc_internal_0134.add_tensor_to_uncleaned(out);

    out->ndim_ = ndim_;
    out->shape_ = shape_;
    out->data_.resize(size());
    out->heap_allocated_ = true;

    for (uint16_t i = 0; i < size(); ++i) {
        // Q7.8 fixed-point multiplication: (a * b) >> 8
        (*out)[i] = static_cast<BASE_VALUE_TYPE>((static_cast<BASE_VALUE_TYPE_2>(data_[i]) * other.data_[i]) * static_cast<BASE_VALUE_TYPE_2>(BASE_VALUE_FRAC));
    }

    if (requires_grad_ || other.requires_grad_) {
        out->set_requires_grad(true);
        auto back_fn = [](Tensor& self) {
            //std::cout << "backward mul" << std::endl;
            for (uint16_t i = 0; i < self.size(); ++i) {
                if (self.parent1_ && self.parent1_->requires_grad_)
                    self.parent1_->grad()[i] += (self.grad()[i] * self.parent2_->operator[](i)) / static_cast<BASE_VALUE_TYPE>(BASE_VALUE_FRAC);
                if (self.parent2_ && self.parent2_->requires_grad_ && self.parent1_ != self.parent2_)
                    self.parent2_->grad()[i] += (self.grad()[i] * self.parent1_->operator[](i)) / static_cast<BASE_VALUE_TYPE>(BASE_VALUE_FRAC);
            }
        };
        out->_set_creator(back_fn, const_cast<Tensor*>(&other), const_cast<Tensor*>(this));
    }

    return *out;
}

Tensor &Tensor::implt_operator_div_i(const Tensor& other) {
    Tensor *out = new Tensor();
    _tensor_gc_internal_0134.add_tensor_to_uncleaned(out);

    out->ndim_ = ndim_;
    out->shape_ = shape_;
    out->data_.resize(size());
    out->heap_allocated_ = true;

    for (uint16_t i = 0; i < size(); ++i) {
        // Q7.8 fixed-point division: (a << 8) / b
        if (other.data_[i] != 0) {
            (*out)[i] = static_cast<BASE_VALUE_TYPE>((static_cast<BASE_VALUE_TYPE_2>(data_[i]) * static_cast<BASE_VALUE_TYPE_2>(BASE_VALUE_FRAC)) / other.data_[i]);
        } else {
            (*out)[i] = 0; // or handle division by zero as needed
        }
    }

    if (requires_grad_ || other.requires_grad_) {
        out->set_requires_grad(true);
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
        out->_set_creator(back_fn, const_cast<Tensor*>(this), const_cast<Tensor*>(&other));
    }

    return *out;
}

Tensor &Tensor::implt_operator_add_i(float scalar) {
    Tensor *out = new Tensor();
    _tensor_gc_internal_0134.add_tensor_to_uncleaned(out);

    out->ndim_ = ndim_;
    out->shape_ = shape_;
    out->data_.resize(size());
    out->heap_allocated_ = true;

    BASE_VALUE_TYPE scalar_quant = float2quant(scalar);
    for (uint16_t i = 0; i < size(); ++i) {
        (*out)[i] = data_[i] + scalar_quant;
    }

    if (requires_grad_) {
        out->set_requires_grad(true);
        auto back_fn = [=](Tensor& self) {
            for (uint16_t i = 0; i < self.size(); ++i) {
                if (self.parent1_ && self.parent1_->requires_grad_)
                    self.parent1_->grad()[i] += self.grad()[i];
                if (self.parent2_ && self.parent2_->requires_grad_)
                    self.parent2_->grad()[i] += self.grad()[i];
            }
        };
        out->_set_creator(back_fn, const_cast<Tensor*>(this), nullptr);
    }

    return *out;
}

Tensor &Tensor::implt_operator_sub_i(float scalar) {
    Tensor *out = new Tensor();
    _tensor_gc_internal_0134.add_tensor_to_uncleaned(out);

    out->ndim_ = ndim_;
    out->shape_ = shape_;
    out->data_.resize(size());
    out->heap_allocated_ = true;

    BASE_VALUE_TYPE scalar_quant = float2quant(scalar);
    for (uint16_t i = 0; i < size(); ++i) {
        (*out)[i] = data_[i] - scalar_quant;
    }

    if (requires_grad_) {
        out->set_requires_grad(true);
        auto back_fn = [=](Tensor& self) {
            for (uint16_t i = 0; i < self.size(); ++i) {
                if (self.parent1_ && self.parent1_->requires_grad_)
                    self.parent1_->grad()[i] += self.grad()[i];
                if (self.parent2_ && self.parent2_->requires_grad_)
                    self.parent2_->grad()[i] -= self.grad()[i];
            }
        };
        out->_set_creator(back_fn, const_cast<Tensor*>(this), nullptr);
    }

    return *out;
}

Tensor &Tensor::implt_operator_mul_i(float scalar) {
    Tensor *out = new Tensor();
    _tensor_gc_internal_0134.add_tensor_to_uncleaned(out);

    out->ndim_ = ndim_;
    out->shape_ = shape_;
    out->data_.resize(size());
    out->heap_allocated_ = true;

    BASE_VALUE_TYPE scalar_quant = float2quant(scalar);
    for (uint16_t i = 0; i < size(); ++i) {
        // Q7.8 fixed-point multiplication: (a * b) >> 8
        (*out)[i] = static_cast<BASE_VALUE_TYPE>((static_cast<BASE_VALUE_TYPE_2>(data_[i]) * scalar_quant) * static_cast<BASE_VALUE_TYPE_2>(BASE_VALUE_FRAC));
    }

    if (requires_grad_) {
        out->set_requires_grad(true);
        auto back_fn = [=](Tensor& self) {
            for (uint16_t i = 0; i < self.size(); ++i) {
                if (self.parent1_ && self.parent1_->requires_grad_)
                    self.parent1_->grad()[i] += (self.grad()[i] * scalar) / static_cast<BASE_VALUE_TYPE>(BASE_VALUE_FRAC);
                if (self.parent2_ && self.parent2_->requires_grad_)
                    self.parent2_->grad()[i] += (self.grad()[i] * self.data_[i]) / static_cast<BASE_VALUE_TYPE>(BASE_VALUE_FRAC);
            }
        };
        out->_set_creator(back_fn, const_cast<Tensor*>(this), nullptr);
    }

    return *out;
}

Tensor &Tensor::implt_operator_div_i(float scalar) {
    Tensor *out = new Tensor();
    _tensor_gc_internal_0134.add_tensor_to_uncleaned(out);

    out->ndim_ = ndim_;
    out->shape_ = shape_;
    out->data_.resize(size());
    out->heap_allocated_ = true;

    BASE_VALUE_TYPE scalar_quant = float2quant(scalar);
    for (uint16_t i = 0; i < size(); ++i) {
        // Q7.8 fixed-point division: (a << 8) / b
        if (scalar_quant != 0) {
            (*out)[i] = static_cast<BASE_VALUE_TYPE>((static_cast<BASE_VALUE_TYPE_2>(data_[i]) * static_cast<BASE_VALUE_TYPE_2>(BASE_VALUE_FRAC)) / scalar_quant);
        } else {
            (*out)[i] = 0; // or handle division by zero as needed
        }
    }

    if (requires_grad_) {
        out->set_requires_grad(true);
        auto back_fn = [=](Tensor& self) {
            for (uint16_t i = 0; i < self.size(); ++i) {
                if (self.parent1_ && self.parent1_->requires_grad_) {
                    // Fixed-point division: (a << 8) / b
                    BASE_VALUE_TYPE grad_val = self.grad()[i];
                    BASE_VALUE_TYPE dL_da = (grad_val * static_cast<BASE_VALUE_TYPE_2>(BASE_VALUE_FRAC)) / scalar_quant;
                    self.parent1_->grad()[i] += dL_da;
                }
                if (self.parent2_ && self.parent2_->requires_grad_) {
                    BASE_VALUE_TYPE a_val = self.data_[i];
                    BASE_VALUE_TYPE dL_db = -self.grad()[i] * a_val / scalar_quant;
                    self.parent2_->grad()[i] += dL_db;
                }
            }
        };
        out->_set_creator(back_fn, const_cast<Tensor*>(this), nullptr);
    }

    return *out;
}



std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    os << "Tensor(shape: [";
    for (size_t i = 0; i < t.shape_.size(); ++i) {
        os << t.shape_[i];
        if (i < t.shape_.size() - 1) {
            os << ", ";
        }
    }
    os << "], data: [";
    for (size_t i = 0; i < t.data_.size(); ++i) {
        os << quant2float(t.data_[i]);
        if (i < t.data_.size() - 1) {
            os << ", ";
        }
    }
    os << "])";
    return os;
}