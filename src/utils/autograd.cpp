#include "../../include/utils/autograd.hpp"
#include <algorithm>
#include <cmath>
#include <queue>
#include <unordered_set>

namespace utils {

    template<typename T>
    void Variable<T>::backward(const Tensor<T>& gradient) {
        if (!requires_grad_) {
            return;
        }
        
        // Initialize gradient if not provided
        Tensor<T> grad = gradient;
        if (gradient.rows() == 0 || gradient.cols() == 0) {
            // Scalar case - gradient is 1
            grad = Tensor<T>::ones(data_.rows(), data_.cols());
        }
        
        // Accumulate gradient
        grad_ = grad_ + grad;
        
        // If this variable has a gradient function, propagate backwards
        if (grad_fn_) {
            auto input_grads = grad_fn_->backward(grad);
            const auto& input_vars = grad_fn_->get_inputs();
            
            // Propagate gradients to input variables
            for (size_t i = 0; i < input_grads.size() && i < input_vars.size(); ++i) {
                if (input_vars[i] && input_vars[i]->requires_grad()) {
                    input_vars[i]->backward(input_grads[i]);
                }
            }
        }
    }
    
    template<typename T>
    Variable<T> Variable<T>::operator+(const Variable<T>& other) const {
        auto add_fn = std::make_shared<AddFunction<T>>();
        Tensor<T> result = add_fn->forward({*this, other});
        
        if (requires_grad_ || other.requires_grad_) {
            auto self_ptr = std::make_shared<Variable<T>>(*this);
            auto other_ptr = std::make_shared<Variable<T>>(other);
            return Variable<T>::create_with_grad_fn(result, add_fn, {self_ptr, other_ptr});
        }
        return Variable<T>(result, false);
    }
    
    template<typename T>
    Variable<T> Variable<T>::operator-(const Variable<T>& other) const {
        auto sub_fn = std::make_shared<SubFunction<T>>();
        Tensor<T> result = sub_fn->forward({*this, other});
        
        if (requires_grad_ || other.requires_grad_) {
            auto self_ptr = std::make_shared<Variable<T>>(*this);
            auto other_ptr = std::make_shared<Variable<T>>(other);
            return Variable<T>::create_with_grad_fn(result, sub_fn, {self_ptr, other_ptr});
        }
        return Variable<T>(result, false);
    }
    
    template<typename T>
    Variable<T> Variable<T>::operator*(const Variable<T>& other) const {
        auto mul_fn = std::make_shared<MulFunction<T>>();
        Tensor<T> result = mul_fn->forward({*this, other});
        
        if (requires_grad_ || other.requires_grad_) {
            auto self_ptr = std::make_shared<Variable<T>>(*this);
            auto other_ptr = std::make_shared<Variable<T>>(other);
            return Variable<T>::create_with_grad_fn(result, mul_fn, {self_ptr, other_ptr});
        }
        return Variable<T>(result, false);
    }
    
    template<typename T>
    Variable<T> Variable<T>::dot(const Variable<T>& other) const {
        auto dot_fn = std::make_shared<DotFunction<T>>();
        Tensor<T> result = dot_fn->forward({*this, other});
        
        if (requires_grad_ || other.requires_grad_) {
            auto self_ptr = std::make_shared<Variable<T>>(*this);
            auto other_ptr = std::make_shared<Variable<T>>(other);
            return Variable<T>::create_with_grad_fn(result, dot_fn, {self_ptr, other_ptr});
        }
        return Variable<T>(result, false);
    }
    
    template<typename T>
    Variable<T> Variable<T>::transpose() const {
        auto transpose_fn = std::make_shared<TransposeFunction<T>>();
        Tensor<T> result = transpose_fn->forward({*this});
        
        if (requires_grad_) {
            auto self_ptr = std::make_shared<Variable<T>>(*this);
            return Variable<T>::create_with_grad_fn(result, transpose_fn, {self_ptr});
        }
        return Variable<T>(result, false);
    }
    
    template<typename T>
    Variable<T> Variable<T>::sum() const {
        auto sum_fn = std::make_shared<SumFunction<T>>();
        Tensor<T> result = sum_fn->forward({*this});
        
        if (requires_grad_) {
            auto self_ptr = std::make_shared<Variable<T>>(*this);
            return Variable<T>::create_with_grad_fn(result, sum_fn, {self_ptr});
        }
        return Variable<T>(result, false);
    }
    
    template<typename T>
    Variable<T> Variable<T>::mean() const {
        auto sum_result = sum();
        T count = static_cast<T>(data_.rows() * data_.cols());
        Tensor<T> count_matrix(1, 1, count);
        Variable<T> count_var(count_matrix, false);
        
        // mean = sum / count
        return sum_result * Variable<T>(Tensor<T>(1, 1, 1.0 / count), false);
    }
    
    template<typename T>
    Variable<T> Variable<T>::sigmoid() const {
        auto sigmoid_fn = std::make_shared<SigmoidFunction<T>>();
        Tensor<T> result = sigmoid_fn->forward({*this});
        
        if (requires_grad_) {
            auto self_ptr = std::make_shared<Variable<T>>(*this);
            return Variable<T>::create_with_grad_fn(result, sigmoid_fn, {self_ptr});
        }
        return Variable<T>(result, false);
    }
    
    template<typename T>
    Variable<T> Variable<T>::tanh() const {
        // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
        Tensor<T> result(data_.rows(), data_.cols());
        for (size_t i = 0; i < data_.rows(); ++i) {
            for (size_t j = 0; j < data_.cols(); ++j) {
                result(i, j) = std::tanh(data_(i, j));
            }
        }
        return Variable<T>(result, requires_grad_);
    }
    
    template<typename T>
    Variable<T> Variable<T>::relu() const {
        Tensor<T> result(data_.rows(), data_.cols());
        for (size_t i = 0; i < data_.rows(); ++i) {
            for (size_t j = 0; j < data_.cols(); ++j) {
                result(i, j) = std::max(static_cast<T>(0), data_(i, j));
            }
        }
        return Variable<T>(result, requires_grad_);
    }
    
    template<typename T>
    Variable<T> Variable<T>::exp() const {
        Tensor<T> result(data_.rows(), data_.cols());
        for (size_t i = 0; i < data_.rows(); ++i) {
            for (size_t j = 0; j < data_.cols(); ++j) {
                result(i, j) = std::exp(data_(i, j));
            }
        }
        return Variable<T>(result, requires_grad_);
    }
    
    template<typename T>
    Variable<T> Variable<T>::log() const {
        Tensor<T> result(data_.rows(), data_.cols());
        for (size_t i = 0; i < data_.rows(); ++i) {
            for (size_t j = 0; j < data_.cols(); ++j) {
                result(i, j) = std::log(data_(i, j));
            }
        }
        return Variable<T>(result, requires_grad_);
    }

    // Explicit template instantiations
    template class Variable<float>;
    template class Variable<double>;
    template class Function<float>;
    template class Function<double>;
    template class AddFunction<float>;
    template class AddFunction<double>;
    template class SubFunction<float>;
    template class SubFunction<double>;
    template class MulFunction<float>;
    template class MulFunction<double>;
    template class DotFunction<float>;
    template class DotFunction<double>;
    template class TransposeFunction<float>;
    template class TransposeFunction<double>;
    template class SigmoidFunction<float>;
    template class SigmoidFunction<double>;
    template class SumFunction<float>;
    template class SumFunction<double>;

} // namespace utils