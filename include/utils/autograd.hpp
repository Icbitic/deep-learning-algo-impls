#pragma once

#include <functional>
#include <memory>
#include <vector>
#include <unordered_set>
#include "matrix.hpp"

/**
 * @file autograd.hpp
 * @brief PyTorch-like automatic differentiation engine
 * @author Kalenitid
 * @version 1.0.0
 */

namespace utils {

    template<typename T>
    class Variable;

    /**
     * @brief Function node in the computational graph
     */
    template<typename T>
    class Function {
    public:
        virtual ~Function() = default;
        
        /**
         * @brief Forward pass computation
         * @param inputs Input variables
         * @return Output matrix
         */
        virtual Matrix<T> forward(const std::vector<Variable<T>>& inputs) = 0;
        
        /**
         * @brief Backward pass computation
         * @param grad_output Gradient from the output
         * @return Gradients with respect to inputs
         */
        virtual std::vector<Matrix<T>> backward(const Matrix<T>& grad_output) = 0;
        
        /**
         * @brief Set saved tensors for backward pass
         */
        virtual void save_for_backward(const std::vector<Matrix<T>>& tensors) {
            saved_tensors_ = tensors;
        }
        
    protected:
        std::vector<Matrix<T>> saved_tensors_;
    };

    /**
     * @brief Variable class that supports automatic differentiation
     */
    template<typename T>
    class Variable {
    public:
        /**
         * @brief Constructor
         * @param data The matrix data
         * @param requires_grad Whether to compute gradients for this variable
         */
        Variable(const Matrix<T>& data, bool requires_grad = false)
            : data_(data), requires_grad_(requires_grad), grad_fn_(nullptr) {
            if (requires_grad_) {
                grad_ = Matrix<T>::zeros(data.rows(), data.cols());
            }
        }
        
        /**
         * @brief Constructor with gradient function
         */
        Variable(const Matrix<T>& data, std::shared_ptr<Function<T>> grad_fn)
            : data_(data), requires_grad_(true), grad_fn_(grad_fn) {
            grad_ = Matrix<T>::zeros(data.rows(), data.cols());
        }
        
        // Getters
        const Matrix<T>& data() const { return data_; }
        Matrix<T>& data() { return data_; }
        const Matrix<T>& grad() const { return grad_; }
        Matrix<T>& grad() { return grad_; }
        bool requires_grad() const { return requires_grad_; }
        std::shared_ptr<Function<T>> grad_fn() const { return grad_fn_; }
        
        /**
         * @brief Perform backward pass
         * @param gradient Optional gradient to start with
         */
        void backward(const Matrix<T>& gradient = Matrix<T>());
        
        /**
         * @brief Zero the gradients
         */
        void zero_grad() {
            if (requires_grad_) {
                grad_ = Matrix<T>::zeros(data_.rows(), data_.cols());
            }
        }
        
        /**
         * @brief Detach from computational graph
         */
        Variable<T> detach() const {
            return Variable<T>(data_, false);
        }
        
        // Arithmetic operations
        Variable<T> operator+(const Variable<T>& other) const;
        Variable<T> operator-(const Variable<T>& other) const;
        Variable<T> operator*(const Variable<T>& other) const;
        
        // Matrix operations
        Variable<T> dot(const Variable<T>& other) const;
        Variable<T> transpose() const;
        Variable<T> sum() const;
        Variable<T> mean() const;
        
        // Activation functions
        Variable<T> sigmoid() const;
        Variable<T> tanh() const;
        Variable<T> relu() const;
        Variable<T> exp() const;
        Variable<T> log() const;
        
        // Element access
        T& operator()(size_t row, size_t col) { return data_(row, col); }
        const T& operator()(size_t row, size_t col) const { return data_(row, col); }
        
        size_t rows() const { return data_.rows(); }
        size_t cols() const { return data_.cols(); }
        
    private:
        Matrix<T> data_;
        Matrix<T> grad_;
        bool requires_grad_;
        std::shared_ptr<Function<T>> grad_fn_;
        std::vector<Variable<T>> inputs_; // For backward pass
    };

    // Specific function implementations
    
    /**
     * @brief Addition function
     */
    template<typename T>
    class AddFunction : public Function<T> {
    public:
        Matrix<T> forward(const std::vector<Variable<T>>& inputs) override {
            return inputs[0].data() + inputs[1].data();
        }
        
        std::vector<Matrix<T>> backward(const Matrix<T>& grad_output) override {
            return {grad_output, grad_output};
        }
    };
    
    /**
     * @brief Subtraction function
     */
    template<typename T>
    class SubFunction : public Function<T> {
    public:
        Matrix<T> forward(const std::vector<Variable<T>>& inputs) override {
            return inputs[0].data() - inputs[1].data();
        }
        
        std::vector<Matrix<T>> backward(const Matrix<T>& grad_output) override {
            return {grad_output, grad_output * Matrix<T>(grad_output.rows(), grad_output.cols(), -1.0)};
        }
    };
    
    /**
     * @brief Element-wise multiplication function
     */
    template<typename T>
    class MulFunction : public Function<T> {
    public:
        Matrix<T> forward(const std::vector<Variable<T>>& inputs) override {
            this->save_for_backward({inputs[0].data(), inputs[1].data()});
            return inputs[0].data() * inputs[1].data();
        }
        
        std::vector<Matrix<T>> backward(const Matrix<T>& grad_output) override {
            return {grad_output * this->saved_tensors_[1], grad_output * this->saved_tensors_[0]};
        }
    };
    
    /**
     * @brief Matrix multiplication function
     */
    template<typename T>
    class DotFunction : public Function<T> {
    public:
        Matrix<T> forward(const std::vector<Variable<T>>& inputs) override {
            this->save_for_backward({inputs[0].data(), inputs[1].data()});
            return dot(inputs[0].data(), inputs[1].data());
        }
        
        std::vector<Matrix<T>> backward(const Matrix<T>& grad_output) override {
            return {
                dot(grad_output, this->saved_tensors_[1].transpose()),
                dot(this->saved_tensors_[0].transpose(), grad_output)
            };
        }
    };
    
    /**
     * @brief Transpose function
     */
    template<typename T>
    class TransposeFunction : public Function<T> {
    public:
        Matrix<T> forward(const std::vector<Variable<T>>& inputs) override {
            return inputs[0].data().transpose();
        }
        
        std::vector<Matrix<T>> backward(const Matrix<T>& grad_output) override {
            return {grad_output.transpose()};
        }
    };
    
    /**
     * @brief Sigmoid function
     */
    template<typename T>
    class SigmoidFunction : public Function<T> {
    public:
        Matrix<T> forward(const std::vector<Variable<T>>& inputs) override {
            Matrix<T> result(inputs[0].rows(), inputs[0].cols());
            for (size_t i = 0; i < inputs[0].rows(); ++i) {
                for (size_t j = 0; j < inputs[0].cols(); ++j) {
                    result(i, j) = 1.0 / (1.0 + std::exp(-inputs[0](i, j)));
                }
            }
            this->save_for_backward({result});
            return result;
        }
        
        std::vector<Matrix<T>> backward(const Matrix<T>& grad_output) override {
            const auto& sigmoid_output = this->saved_tensors_[0];
            Matrix<T> grad_input(sigmoid_output.rows(), sigmoid_output.cols());
            for (size_t i = 0; i < sigmoid_output.rows(); ++i) {
                for (size_t j = 0; j < sigmoid_output.cols(); ++j) {
                    grad_input(i, j) = grad_output(i, j) * sigmoid_output(i, j) * (1.0 - sigmoid_output(i, j));
                }
            }
            return {grad_input};
        }
    };
    
    /**
     * @brief Sum function
     */
    template<typename T>
    class SumFunction : public Function<T> {
    public:
        Matrix<T> forward(const std::vector<Variable<T>>& inputs) override {
            this->save_for_backward({inputs[0].data()});
            T sum_val = sum(inputs[0].data());
            return Matrix<T>(1, 1, sum_val);
        }
        
        std::vector<Matrix<T>> backward(const Matrix<T>& grad_output) override {
            const auto& input_shape = this->saved_tensors_[0];
            return {Matrix<T>(input_shape.rows(), input_shape.cols(), grad_output(0, 0))};
        }
    };

    // Type aliases
    using VariableF = Variable<float>;
    using VariableD = Variable<double>;

} // namespace utils