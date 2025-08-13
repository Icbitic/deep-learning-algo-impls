#pragma once

#include <functional>
#include <memory>
#include <vector>
#include "utils/autograd.hpp"
#include "utils/tensor.hpp"

/**
 * @file layers.hpp
 * @brief PyTorch-like neural network layers with automatic differentiation
 * @author Kalenitid
 * @version 1.0.0
 */

namespace dl::layers {
    using utils::Tensor;
    using utils::TensorD;
    using utils::TensorF;
    using utils::Variable;
    using utils::VariableD;
    using utils::VariableF;

    /**
     * @brief Base class for all neural network modules (PyTorch-like nn.Module)
     */
    template<typename T>
    class Module {
    public:
        virtual ~Module() = default;

        /**
         * @brief Forward pass through the module
         * @param input Input variable
         * @return Output variable
         */
        virtual std::shared_ptr<Variable<T>> forward(const std::shared_ptr<Variable<T>> &input) = 0;

        /**
         * @brief Get all parameters of this module
         * @return Vector of parameter variables
         */
        virtual std::vector<std::shared_ptr<Variable<T>>> parameters() = 0;

        /**
         * @brief Zero gradients of all parameters
         */
        virtual void zero_grad() {
            for (auto param: parameters()) {
                param->zero_grad();
            }
        }

        /**
         * @brief Set training mode
         * @param training Whether in training mode
         */
        virtual void train(bool training = true) { training_ = training; }

        /**
         * @brief Set evaluation mode
         */
        virtual void eval() { train(false); }

        /**
         * @brief Check if module is in training mode
         */
        bool is_training() const { return training_; }

    protected:
        bool training_ = true;
    };

    /**
     * @brief Linear (fully connected) layer: y = xW^T + b
     */
    template<typename T>
    class Linear : public Module<T> {
    public:
        /**
         * @brief Constructor
         * @param in_features Number of input features
         * @param out_features Number of output features
         * @param bias Whether to include bias term
         */
        Linear(size_t in_features, size_t out_features, bool bias = true);

        /**
         * @brief Forward pass: y = xW^T + b
         * @param input Input variable of shape (batch_size, in_features)
         * @return Output variable of shape (batch_size, out_features)
         */
        std::shared_ptr<Variable<T>> forward(const std::shared_ptr<Variable<T>> &input) override;

        /**
         * @brief Get parameters (weight and bias)
         */
        std::vector<std::shared_ptr<Variable<T>>> parameters() override;

        // Getters for parameters
        std::shared_ptr<Variable<T>> &weight() { return weight_; }
        std::shared_ptr<Variable<T>> &bias() { return bias_; }
        const std::shared_ptr<Variable<T>> &weight() const { return weight_; }
        const std::shared_ptr<Variable<T>> &bias() const { return bias_; }

    private:
        std::shared_ptr<Variable<T>> weight_; // Shape: (out_features, in_features)
        std::shared_ptr<Variable<T>> bias_; // Shape: (out_features,)
        bool has_bias_;
        size_t in_features_;
        size_t out_features_;

        void initialize_parameters();
    };

    /**
     * @brief ReLU activation function
     */
    template<typename T>
    class ReLU : public Module<T> {
    public:
        std::shared_ptr<Variable<T>> forward(const std::shared_ptr<Variable<T>> &input) override;

        std::vector<std::shared_ptr<Variable<T>>> parameters() override { return {}; }
    };

    /**
     * @brief Sigmoid activation function
     */
    template<typename T>
    class Sigmoid : public Module<T> {
    public:
        std::shared_ptr<Variable<T>> forward(const std::shared_ptr<Variable<T>> &input) override;

        std::vector<std::shared_ptr<Variable<T>>> parameters() override { return {}; }
    };

    /**
     * @brief Tanh activation function
     */
    template<typename T>
    class Tanh : public Module<T> {
    public:
        std::shared_ptr<Variable<T>> forward(const std::shared_ptr<Variable<T>> &input) override;

        std::vector<std::shared_ptr<Variable<T>>> parameters() override { return {}; }
    };

    /**
     * @brief Dropout layer for regularization
     */
    template<typename T>
    class Dropout : public Module<T> {
    public:
        /**
         * @brief Constructor
         * @param p Dropout probability (0.0 to 1.0)
         */
        explicit Dropout(T p = 0.5) : p_(p) {
        }

        std::shared_ptr<Variable<T>> forward(const std::shared_ptr<Variable<T>> &input) override;

        std::vector<std::shared_ptr<Variable<T>>> parameters() override { return {}; }

    private:
        T p_; // Dropout probability
    };

    /**
     * @brief Sequential container for chaining modules
     */
    template<typename T>
    class Sequential : public Module<T> {
    public:
        /**
         * @brief Add a module to the sequence
         */
        void add_module(std::shared_ptr<Module<T> > module);

        /**
         * @brief Forward pass through all modules in sequence
         */
        std::shared_ptr<Variable<T>> forward(const std::shared_ptr<Variable<T>> &input) override;

        /**
         * @brief Get all parameters from all modules
         */
        std::vector<std::shared_ptr<Variable<T>>> parameters() override;

        /**
         * @brief Zero gradients for all modules
         */
        void zero_grad() override;

        /**
         * @brief Set training mode for all modules
         */
        void train(bool training = true) override;

    private:
        std::vector<std::shared_ptr<Module<T> > > modules_;
    };

    // Type aliases for convenience
    using LinearD = Linear<double>;
    using LinearF = Linear<float>;
    using ReLUD = ReLU<double>;
    using ReLUF = ReLU<float>;
    using SigmoidD = Sigmoid<double>;
    using SigmoidF = Sigmoid<float>;
    using TanhD = Tanh<double>;
    using TanhF = Tanh<float>;
    using DropoutD = Dropout<double>;
    using DropoutF = Dropout<float>;
    using SequentialD = Sequential<double>;
    using SequentialF = Sequential<float>;
} // namespace dl::layers
