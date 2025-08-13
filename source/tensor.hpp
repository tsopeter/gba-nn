#pragma once

#include <stdint.h>
#include <vector>
#include <functional>

using BASE_VALUE_TYPE   = float;
using BASE_VALUE_TYPE_2 = float;
using BASE_VALUE_VEC  = std::vector<BASE_VALUE_TYPE>;
#define BASE_VALUE_Q 1
#define BASE_VALUE_FRAC (1 << BASE_VALUE_Q)

using shape_t = std::vector<uint16_t>;

BASE_VALUE_TYPE float2quant(float value);
float           quant2float(BASE_VALUE_TYPE value);

class Tensor {
    public:
        Tensor();
        Tensor(const Tensor&);
        Tensor(std::vector<float> data, shape_t shape);
        Tensor(std::vector<float> data, shape_t shape, bool requires_grad);
        ~Tensor();

        Tensor &operator+(const Tensor& other);
        Tensor &operator-(const Tensor& other);
        Tensor &operator*(const Tensor& other);
        Tensor &operator/(const Tensor& other);

        void set_requires_grad(bool requires_grad);
        bool requires_grad() const;
        uint16_t size() const;

        BASE_VALUE_TYPE& operator[](size_t index);
        const BASE_VALUE_TYPE& operator[](size_t index) const;

        uint16_t dim(uint16_t index) const;
        uint16_t ndim() const;

        void backward(bool start=true);
        void zero_grad();

        BASE_VALUE_VEC& data();
        BASE_VALUE_VEC& grad();

        void update (float lr);

        friend std::ostream& operator<<(std::ostream& os, const Tensor& t);

    private:

        void _set_creator(std::function<void(Tensor&)> fn, Tensor* parent1 = nullptr, Tensor* parent2 = nullptr);

        std::vector<BASE_VALUE_TYPE> data_;
        uint16_t ndim_;
        shape_t shape_;
        bool requires_grad_ = false;
        std::function<void(Tensor&)> backward_fn_;
        Tensor* parent1_ = nullptr;
        Tensor* parent2_ = nullptr;
        std::vector<BASE_VALUE_TYPE> grad_;
        bool heap_allocated_ = false;

        Tensor &implt_operator_add_i(const Tensor& other);
        Tensor &implt_operator_sub_i(const Tensor& other);
        Tensor &implt_operator_mul_i(const Tensor& other);
        Tensor &implt_operator_div_i(const Tensor& other);

        std::vector<Tensor*> children_;
};

void print_tensor(Tensor& t);
void print_tensor_grad(Tensor& t);

class TensorResourceTracker {
public:
    TensorResourceTracker();
    ~TensorResourceTracker();
    void add_tensor_to_uncleaned(Tensor* tensor);
    void cleanup_tensors();
    uint64_t count_uncleaned() const;
private:

    bool is_already_cleaned(Tensor* tensor) const;

    std::vector<Tensor*> uncleaned_tensors_;
    std::vector<Tensor*> cleaned_tensors_;
};

void     Tensor_GC();
uint64_t Tensor_GC_Count();