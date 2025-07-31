#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include "utils/autograd.hpp"
#include "utils/matrix.hpp"

/**
 * @file optimizers.hpp
 * @brief PyTorch-like optimizers with automatic differentiation support
 * @author Kalenitid
 * @version 1.0.0
 */

namespace dl::optimization {
    using utils::Variable;
    using utils::VariableD;
    using utils::VariableF;
    using utils::Matrix;
    using utils::MatrixD;
    using utils::MatrixF;

    /**
     * @brief Base class for autograd-compatible optimizers
     */
    template<typename T>
    class AutogradOptimizer {
    public:
        /**
         * @brief Constructor
         * @param parameters Vector of parameter variables to optimize
         */
        explicit AutogradOptimizer(std::vector<Variable<T>*> parameters)
            : parameters_(parameters) {}
        
        virtual ~AutogradOptimizer() = default;
        
        /**
         * @brief Perform one optimization step
         */
        virtual void step() = 0;
        
        /**
         * @brief Zero gradients of all parameters
         */
        virtual void zero_grad() {
            for (auto* param : parameters_) {
                param->zero_grad();
            }
        }
        
        /**
         * @brief Get learning rate
         */
        virtual T get_lr() const = 0;
        
        /**
         * @brief Set learning rate
         */
        virtual void set_lr(T lr) = 0;
        
    protected:
        std::vector<Variable<T>*> parameters_;
    };

    /**
     * @brief Stochastic Gradient Descent optimizer with autograd support
     * 
     * Updates parameters using: param = param - lr * grad
     * 
     * Supports:
     * - Basic SGD
     * - Momentum
     * - Weight decay
     * - Nesterov momentum
     */
    template<typename T>
    class SGD : public AutogradOptimizer<T> {
    public:
        /**
         * @brief Constructor
         * @param parameters Parameters to optimize
         * @param lr Learning rate
         * @param momentum Momentum factor (default: 0)
         * @param weight_decay Weight decay (L2 penalty) (default: 0)
         * @param nesterov Enable Nesterov momentum (default: false)
         */
        SGD(std::vector<Variable<T>*> parameters, 
            T lr, 
            T momentum = 0.0, 
            T weight_decay = 0.0, 
            bool nesterov = false);
        
        /**
         * @brief Perform one SGD step
         */
        void step() override;
        
        /**
         * @brief Get learning rate
         */
        T get_lr() const override { return lr_; }
        
        /**
         * @brief Set learning rate
         */
        void set_lr(T lr) override { lr_ = lr; }
        
    private:
        T lr_;
        T momentum_;
        T weight_decay_;
        bool nesterov_;
        
        // Momentum buffers for each parameter
        std::vector<Matrix<T>> momentum_buffers_;
        
        void initialize_momentum_buffers();
    };

    /**
     * @brief Adam optimizer with autograd support
     * 
     * Adaptive learning rate optimizer that computes individual learning rates
     * for different parameters from estimates of first and second moments.
     * 
     * Paper: "Adam: A Method for Stochastic Optimization" (Kingma & Ba, 2014)
     */
    template<typename T>
    class Adam : public AutogradOptimizer<T> {
    public:
        /**
         * @brief Constructor
         * @param parameters Parameters to optimize
         * @param lr Learning rate (default: 1e-3)
         * @param beta1 Coefficient for first moment estimate (default: 0.9)
         * @param beta2 Coefficient for second moment estimate (default: 0.999)
         * @param eps Term for numerical stability (default: 1e-8)
         * @param weight_decay Weight decay (L2 penalty) (default: 0)
         */
        Adam(std::vector<Variable<T>*> parameters,
             T lr = 1e-3,
             T beta1 = 0.9,
             T beta2 = 0.999,
             T eps = 1e-8,
             T weight_decay = 0.0);
        
        /**
         * @brief Perform one Adam step
         */
        void step() override;
        
        /**
         * @brief Get learning rate
         */
        T get_lr() const override { return lr_; }
        
        /**
         * @brief Set learning rate
         */
        void set_lr(T lr) override { lr_ = lr; }
        
    private:
        T lr_;
        T beta1_;
        T beta2_;
        T eps_;
        T weight_decay_;
        
        // State for each parameter
        std::vector<Matrix<T>> exp_avg_;        // First moment estimate
        std::vector<Matrix<T>> exp_avg_sq_;     // Second moment estimate
        size_t step_count_;
        
        void initialize_state();
    };

    /**
     * @brief AdamW optimizer with autograd support
     * 
     * Adam with decoupled weight decay regularization.
     * Often performs better than Adam with L2 regularization.
     * 
     * Paper: "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2017)
     */
    template<typename T>
    class AdamW : public AutogradOptimizer<T> {
    public:
        /**
         * @brief Constructor
         * @param parameters Parameters to optimize
         * @param lr Learning rate (default: 1e-3)
         * @param beta1 Coefficient for first moment estimate (default: 0.9)
         * @param beta2 Coefficient for second moment estimate (default: 0.999)
         * @param eps Term for numerical stability (default: 1e-8)
         * @param weight_decay Weight decay coefficient (default: 1e-2)
         */
        AdamW(std::vector<Variable<T>*> parameters,
              T lr = 1e-3,
              T beta1 = 0.9,
              T beta2 = 0.999,
              T eps = 1e-8,
              T weight_decay = 1e-2);
        
        /**
         * @brief Perform one AdamW step
         */
        void step() override;
        
        /**
         * @brief Get learning rate
         */
        T get_lr() const override { return lr_; }
        
        /**
         * @brief Set learning rate
         */
        void set_lr(T lr) override { lr_ = lr; }
        
    private:
        T lr_;
        T beta1_;
        T beta2_;
        T eps_;
        T weight_decay_;
        
        // State for each parameter
        std::vector<Matrix<T>> exp_avg_;        // First moment estimate
        std::vector<Matrix<T>> exp_avg_sq_;     // Second moment estimate
        size_t step_count_;
        
        void initialize_state();
    };

    /**
     * @brief RMSprop optimizer with autograd support
     * 
     * Maintains a moving average of squared gradients to normalize the gradient.
     * 
     * Paper: "Lecture 6.5-rmsprop" (Hinton, 2012)
     */
    template<typename T>
    class RMSprop : public AutogradOptimizer<T> {
    public:
        /**
         * @brief Constructor
         * @param parameters Parameters to optimize
         * @param lr Learning rate (default: 1e-2)
         * @param alpha Smoothing constant (default: 0.99)
         * @param eps Term for numerical stability (default: 1e-8)
         * @param weight_decay Weight decay (L2 penalty) (default: 0)
         * @param momentum Momentum factor (default: 0)
         */
        RMSprop(std::vector<Variable<T>*> parameters,
                T lr = 1e-2,
                T alpha = 0.99,
                T eps = 1e-8,
                T weight_decay = 0.0,
                T momentum = 0.0);
        
        /**
         * @brief Perform one RMSprop step
         */
        void step() override;
        
        /**
         * @brief Get learning rate
         */
        T get_lr() const override { return lr_; }
        
        /**
         * @brief Set learning rate
         */
        void set_lr(T lr) override { lr_ = lr; }
        
    private:
        T lr_;
        T alpha_;
        T eps_;
        T weight_decay_;
        T momentum_;
        
        // State for each parameter
        std::vector<Matrix<T>> square_avg_;     // Moving average of squared gradients
        std::vector<Matrix<T>> momentum_buffer_; // Momentum buffer (if momentum > 0)
        
        void initialize_state();
    };

    /**
     * @brief Learning rate scheduler base class
     */
    template<typename T>
    class LRScheduler {
    public:
        explicit LRScheduler(AutogradOptimizer<T>* optimizer) : optimizer_(optimizer) {}
        virtual ~LRScheduler() = default;
        
        /**
         * @brief Update learning rate
         */
        virtual void step() = 0;
        
        /**
         * @brief Get current learning rate
         */
        T get_lr() const { return optimizer_->get_lr(); }
        
    protected:
        AutogradOptimizer<T>* optimizer_;
    };

    /**
     * @brief Step learning rate scheduler
     * Decays learning rate by gamma every step_size epochs
     */
    template<typename T>
    class StepLR : public LRScheduler<T> {
    public:
        StepLR(AutogradOptimizer<T>* optimizer, size_t step_size, T gamma = 0.1)
            : LRScheduler<T>(optimizer), step_size_(step_size), gamma_(gamma), 
              last_epoch_(0), base_lr_(optimizer->get_lr()) {}
        
        void step() override;
        
    private:
        size_t step_size_;
        T gamma_;
        size_t last_epoch_;
        T base_lr_;
    };

    // Type aliases for convenience
    using SGDD = SGD<double>;
    using SGDF = SGD<float>;
    using AdamD = Adam<double>;
    using AdamF = Adam<float>;
    using AdamWD = AdamW<double>;
    using AdamWF = AdamW<float>;
    using RMSpropD = RMSprop<double>;
    using RMSpropF = RMSprop<float>;
    using StepLRD = StepLR<double>;
    using StepLRF = StepLR<float>;

} // namespace dl::optimization