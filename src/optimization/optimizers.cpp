#include "optimization/optimizers.hpp"
#include <algorithm>
#include <cmath>

namespace dl::optimization {
    // SGD Implementation
    template<typename T>
    SGD<T>::SGD(std::vector<std::shared_ptr<Variable<T>>> parameters, T lr, T momentum,
                T weight_decay, bool nesterov) : AutogradOptimizer<T>(parameters),
                                                 lr_(lr), momentum_(momentum),
                                                 weight_decay_(weight_decay),
                                                 nesterov_(nesterov) {
        initialize_momentum_buffers();
    }

    template<typename T>
    void SGD<T>::initialize_momentum_buffers() {
        momentum_buffers_.clear();
        momentum_buffers_.reserve(this->parameters_.size());

        for (const auto &param: this->parameters_) {
            // TODO: Initialize momentum buffer with same shape as parameter
            // momentum_buffers_.emplace_back(param->data().rows(), param->data().cols());
        }
    }

    template<typename T>
    void SGD<T>::step() {
        for (size_t i = 0; i < this->parameters_.size(); ++i) {
            auto param = this->parameters_[i];

            // TODO: Implement SGD update logic
            // 1. Apply weight decay if specified
            // 2. Update momentum buffer if momentum > 0
            // 3. Apply Nesterov momentum if enabled
            // 4. Update parameter: param = param - lr * effective_grad
        }
    }

    // Adam Implementation
    template<typename T>
    Adam<T>::Adam(std::vector<std::shared_ptr<Variable<T>>> parameters, T lr, T beta1, T beta2,
                  T eps, T weight_decay) : AutogradOptimizer<T>(parameters),
                                           lr_(lr), beta1_(beta1), beta2_(beta2),
                                           eps_(eps), weight_decay_(weight_decay),
                                           step_count_(0) {
        initialize_state();
    }

    template<typename T>
    void Adam<T>::initialize_state() {
        exp_avg_.clear();
        exp_avg_sq_.clear();
        exp_avg_.reserve(this->parameters_.size());
        exp_avg_sq_.reserve(this->parameters_.size());

        for (const auto &param: this->parameters_) {
            // TODO: Initialize first and second moment estimates
            // exp_avg_.emplace_back(param->data().rows(), param->data().cols());
            // exp_avg_sq_.emplace_back(param->data().rows(), param->data().cols());
        }
    }

    template<typename T>
    void Adam<T>::step() {
        step_count_++;

        for (size_t i = 0; i < this->parameters_.size(); ++i) {
            auto param = this->parameters_[i];

            // TODO: Implement Adam update logic
            // 1. Apply weight decay if specified
            // 2. Update biased first moment estimate: m_t = beta1 * m_{t-1} + (1 - beta1) * grad
            // 3. Update biased second moment estimate: v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
            // 4. Compute bias-corrected first moment: m_hat = m_t / (1 - beta1^t)
            // 5. Compute bias-corrected second moment: v_hat = v_t / (1 - beta2^t)
            // 6. Update parameter: param = param - lr * m_hat / (sqrt(v_hat) + eps)
        }
    }

    // AdamW Implementation
    template<typename T>
    AdamW<T>::AdamW(std::vector<std::shared_ptr<Variable<T>>> parameters, T lr, T beta1, T beta2,
                    T eps, T weight_decay) : AutogradOptimizer<T>(parameters),
                                             lr_(lr), beta1_(beta1), beta2_(beta2),
                                             eps_(eps), weight_decay_(weight_decay),
                                             step_count_(0) {
        initialize_state();
    }

    template<typename T>
    void AdamW<T>::initialize_state() {
        exp_avg_.clear();
        exp_avg_sq_.clear();
        exp_avg_.reserve(this->parameters_.size());
        exp_avg_sq_.reserve(this->parameters_.size());

        for (const auto param: this->parameters_) {
            // TODO: Initialize first and second moment estimates
            // exp_avg_.emplace_back(param->data().rows(), param->data().cols());
            // exp_avg_sq_.emplace_back(param->data().rows(), param->data().cols());
        }
    }

    template<typename T>
    void AdamW<T>::step() {
        step_count_++;

        for (size_t i = 0; i < this->parameters_.size(); ++i) {
            auto param = this->parameters_[i];

            // TODO: Implement AdamW update logic
            // 1. Update biased first moment estimate: m_t = beta1 * m_{t-1} + (1 - beta1) * grad
            // 2. Update biased second moment estimate: v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
            // 3. Compute bias-corrected first moment: m_hat = m_t / (1 - beta1^t)
            // 4. Compute bias-corrected second moment: v_hat = v_t / (1 - beta2^t)
            // 5. Apply decoupled weight decay: param = param * (1 - lr * weight_decay)
            // 6. Update parameter: param = param - lr * m_hat / (sqrt(v_hat) + eps)
        }
    }

    // RMSprop Implementation
    template<typename T>
    RMSprop<T>::RMSprop(std::vector<std::shared_ptr<Variable<T>>> parameters, T lr, T alpha, T eps,
                        T weight_decay,
                        T momentum) : AutogradOptimizer<T>(parameters), lr_(lr),
                                      alpha_(alpha), eps_(eps),
                                      weight_decay_(weight_decay),
                                      momentum_(momentum) {
        initialize_state();
    }

    template<typename T>
    void RMSprop<T>::initialize_state() {
        square_avg_.clear();
        momentum_buffer_.clear();
        square_avg_.reserve(this->parameters_.size());
        momentum_buffer_.reserve(this->parameters_.size());

        for (const auto param: this->parameters_) {
            // TODO: Initialize moving average of squared gradients
            // square_avg_.emplace_back(param->data().rows(), param->data().cols());
            // if (momentum_ > 0) {
            //     momentum_buffer_.emplace_back(param->data().rows(), param->data().cols());
            // }
        }
    }

    template<typename T>
    void RMSprop<T>::step() {
        for (size_t i = 0; i < this->parameters_.size(); ++i) {
            auto param = this->parameters_[i];

            // TODO: Implement RMSprop update logic
            // 1. Apply weight decay if specified
            // 2. Update moving average of squared gradients: v_t = alpha * v_{t-1} + (1 - alpha) * grad^2
            // 3. Compute update: update = grad / (sqrt(v_t) + eps)
            // 4. Apply momentum if specified: buf = momentum * buf + update
            // 5. Update parameter: param = param - lr * (momentum > 0 ? buf : update)
        }
    }

    // StepLR Implementation
    template<typename T>
    void StepLR<T>::step() {
        last_epoch_++;
        if (last_epoch_ % step_size_ == 0) {
            T new_lr = base_lr_ * std::pow(gamma_, last_epoch_ / step_size_);
            this->optimizer_->set_lr(new_lr);
        }
    }

    // Explicit template instantiations
    template class SGD<float>;
    template class SGD<double>;
    template class Adam<float>;
    template class Adam<double>;
    template class AdamW<float>;
    template class AdamW<double>;
    template class RMSprop<float>;
    template class RMSprop<double>;
    template class StepLR<float>;
    template class StepLR<double>;
} // namespace dl::optimization
