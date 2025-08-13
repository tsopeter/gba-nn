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

class Tensor : public std::enable_shared_from_this<Tensor> {
    public:
        Tensor();
        Tensor(const Tensor&);
        Tensor(std::vector<float> data, shape_t shape);
        Tensor(std::vector<float> data, shape_t shape, bool requires_grad);

        std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor>& other);
        std::shared_ptr<Tensor> operator-(std::shared_ptr<Tensor>& other);
        std::shared_ptr<Tensor> operator*(std::shared_ptr<Tensor>& other);
        std::shared_ptr<Tensor> operator/(std::shared_ptr<Tensor>& other);

        std::shared_ptr<Tensor> implt_operator_add_i(std::shared_ptr<Tensor>& other);
        std::shared_ptr<Tensor> implt_operator_sub_i(std::shared_ptr<Tensor>& other);
        std::shared_ptr<Tensor> implt_operator_mul_i(std::shared_ptr<Tensor>& other);
        std::shared_ptr<Tensor> implt_operator_div_i(std::shared_ptr<Tensor>& other);

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

    private:

        void _set_creator(std::function<void(Tensor&)> fn, 
        std::shared_ptr<Tensor> parent1 = nullptr, 
        std::shared_ptr<Tensor> parent2 = nullptr);

        std::vector<BASE_VALUE_TYPE> data_;
        uint16_t ndim_;
        shape_t shape_;
        bool requires_grad_ = false;
        std::function<void(Tensor&)> backward_fn_;
        std::shared_ptr<Tensor> parent1_ = nullptr;
        std::shared_ptr<Tensor> parent2_ = nullptr;
        std::vector<BASE_VALUE_TYPE> grad_;

};

void print_tensor(Tensor& t);
void print_tensor_grad(Tensor& t);

