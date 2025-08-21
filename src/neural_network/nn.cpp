#include "neural_network/nn.hpp"
#include <algorithm>
#include <random>
#include <stdexcept>

namespace dl::nn {
    // ============================================================================
    // Linear Layer Implementation
    // ============================================================================

    template<typename T>
    Linear<
        T>::Linear(size_t in_features, size_t out_features,
                   bool bias) : in_features_(in_features),
                                out_features_(out_features), has_bias_(bias),
                                weight_(
                                    std::make_shared<Variable<T> >(Tensor<T>::zeros({out_features, in_features}),
                                                                   true)),
                                bias_(std::make_shared<Variable<T> >(Tensor<T>::zeros({out_features, 1}),
                                                                     true)) {
        initialize_parameters();
    }

    template<typename T>
    void Linear<T>::initialize_parameters() {
        // TODO: Implement Xavier/He initialization
        // Hint: Use normal distribution with appropriate variance
        // Xavier: std = sqrt(2.0 / (in_features + out_features))
        // He: std = sqrt(2.0 / in_features)

        // Placeholder: Initialize with small random values
        std::random_device rd;
        std::mt19937 gen(rd());
        T std_dev = std::sqrt(2.0 / (in_features_ + out_features_));
        std::normal_distribution<T> dist(0.0, std_dev);

        // TODO: Fill weight_ matrix with random values
        // TODO: Initialize bias_ to zeros if has_bias_ is true
    }

    template<typename T>
    std::shared_ptr<Variable<T> > Linear<T>::forward(const std::shared_ptr<Variable<T> > &input) {
        // TODO: Implement forward pass: y = xW^T + b
        // Steps:
        // 1. Compute input.dot(weight_.transpose())
        // 2. Add bias if has_bias_ is true
        // 3. Return result

        // Placeholder implementation
        auto output = input->dot(weight_->transpose());
        if (has_bias_) {
            output = output + bias_;
        }
        return output;
    }

    template<typename T>
    std::vector<std::shared_ptr<Variable<T> > > Linear<T>::parameters() {
        std::vector<std::shared_ptr<Variable<T> > > params;
        params.push_back(weight_);
        if (has_bias_) {
            params.push_back(bias_);
        }
        return params;
    }

    // ============================================================================
    // Activation Functions Implementation
    // ============================================================================

    template<typename T>
    std::shared_ptr<Variable<T> > ReLU<T>::forward(const std::shared_ptr<Variable<T> > &input) {
        // TODO: Implement ReLU activation
        // Hint: Use input.relu() method from autograd
        return input->relu();
    }

    template<typename T>
    std::shared_ptr<Variable<T> > Sigmoid<T>::forward(const std::shared_ptr<Variable<T> > &input) {
        // TODO: Implement Sigmoid activation
        // Hint: Use input.sigmoid() method from autograd
        return input->sigmoid();
    }

    template<typename T>
    std::shared_ptr<Variable<T> > Tanh<T>::forward(const std::shared_ptr<Variable<T> > &input) {
        // TODO: Implement Tanh activation
        // Hint: Use input.tanh() method from autograd
        return input->tanh();
    }

    // ============================================================================
    // Dropout Implementation
    // ============================================================================

    template<typename T>
    std::shared_ptr<Variable<T> > Dropout<T>::forward(const std::shared_ptr<Variable<T> > &input) {
        // TODO: Implement dropout
        // During training:
        //   - Generate random mask with probability p_
        //   - Multiply input by mask
        //   - Scale by 1/(1-p_) to maintain expected value
        // During evaluation:
        //   - Return input unchanged

        if (!this->is_training()) {
            return input;
        }

        // TODO: Implement training mode dropout
        // Placeholder: return input unchanged
        return input;
    }

    // ============================================================================
    // Sequential Container Implementation
    // ============================================================================

    template<typename T>
    void Sequential<T>::add_module(std::shared_ptr<Module<T> > module) {
        modules_.push_back(module);
    }

    template<typename T>
    std::shared_ptr<Variable<T> > Sequential<T>::forward(const std::shared_ptr<Variable<T> > &input) {
        // TODO: Implement sequential forward pass
        // Apply each module in sequence to the input

        auto output = input;
        for (auto &module: modules_) {
            output = module->forward(output);
        }
        return output;
    }

    template<typename T>
    std::vector<std::shared_ptr<Variable<T> > > Sequential<T>::parameters() {
        std::vector<std::shared_ptr<Variable<T> > > all_params;
        for (auto &module: modules_) {
            auto module_params = module->parameters();
            all_params.insert(all_params.end(), module_params.begin(),
                              module_params.end());
        }
        return all_params;
    }

    template<typename T>
    void Sequential<T>::zero_grad() {
        for (auto &module: modules_) {
            module->zero_grad();
        }
    }

    template<typename T>
    void Sequential<T>::train(bool training) {
        Module<T>::train(training);
        for (auto &module: modules_) {
            module->train(training);
        }
    }

    // ============================================================================
    // Explicit Template Instantiations
    // ============================================================================

    template class Linear<float>;
    template class Linear<double>;
    template class ReLU<float>;
    template class ReLU<double>;
    template class Sigmoid<float>;
    template class Sigmoid<double>;
    template class Tanh<float>;
    template class Tanh<double>;
    template class Dropout<float>;
    template class Dropout<double>;
    template class Sequential<float>;
    template class Sequential<double>;
} // namespace dl::layers
