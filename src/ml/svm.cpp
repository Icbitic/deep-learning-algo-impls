#include "ml/svm.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <set>
#include <stdexcept>

namespace ml {
    template<typename T>
    SVM<T>::SVM(KernelType kernel_type, T C, T gamma, int degree, T coef0, T tol,
                size_t max_iter, T learning_rate) : kernel_type_(kernel_type),
                                                    C_(C), gamma_(gamma),
                                                    degree_(degree), coef0_(coef0),
                                                    tol_(tol), max_iter_(max_iter),
                                                    learning_rate_(learning_rate),
                                                    weights_(
                                                        std::make_shared<Variable<T> >(Tensor<T>({0.0}, {1, 1}),
                                                            true)),
                                                    // Initialize with dummy data, will be resized in fit
                                                    bias_(std::make_shared<Variable<T> >(Tensor<T>({0.0}, {1, 1}),
                                                        true)),
                                                    is_fitted_(false) {
        // Initialize with dummy data

        if (C <= 0) {
            throw std::invalid_argument("C must be positive");
        }
        if (gamma <= 0) {
            throw std::invalid_argument("gamma must be positive");
        }
        if (learning_rate <= 0) {
            throw std::invalid_argument("learning_rate must be positive");
        }
    }

    template<typename T>
    void SVM<T>::fit(const Tensor<T> &X, const std::vector<int> &y) {
        if (X.rows() != y.size()) {
            throw std::invalid_argument("Number of samples in X and y must match");
        }

        // Get unique classes
        std::set<int> unique_classes(y.begin(), y.end());
        classes_.assign(unique_classes.begin(), unique_classes.end());

        if (classes_.size() != 2) {
            throw std::invalid_argument(
                "Currently only binary classification is supported");
        }

        // Convert labels to -1, +1
        std::vector<int> binary_labels(y.size());
        for (size_t i = 0; i < y.size(); ++i) {
            binary_labels[i] = (y[i] == classes_[0]) ? -1 : 1;
        }

        // Initialize parameters with autograd
        // For linear SVM, we use weight vector; for kernel SVM, we use dual variables
        if (kernel_type_ == KernelType::LINEAR) {
            // Initialize weights randomly
            Tensor<T> w_init = Tensor<T>::random({X.cols(), 1}, -0.1, 0.1);
            weights_ = std::make_shared<Variable<T> >(w_init, true); // requires_grad = true

            Tensor<T> b_init({0.0}, {1, 1});
            bias_ = std::make_shared<Variable<T> >(b_init, true);
        } else {
            // For kernel methods, initialize dual variables
            Tensor<T> alpha_init = Tensor<T>::random({X.rows(), 1}, 0.0, 0.1);
            weights_ = std::make_shared<Variable<T> >(alpha_init, true);

            Tensor<T> b_init({0.0}, {1, 1});
            bias_ = std::make_shared<Variable<T> >(b_init, true);
        }

        loss_history_.clear();

        // Training loop with gradient descent
        std::cout << "Training SVM with automatic differentiation..." << std::endl;

        for (size_t iter = 0; iter < max_iter_; ++iter) {
            T loss = gradient_step(X, binary_labels);
            loss_history_.push_back(loss);

            if (iter % 100 == 0) {
                std::cout << "Iteration " << iter << ", Loss: " << loss << std::endl;
            }

            // Check convergence
            if (loss_history_.size() > 1) {
                T loss_diff =
                        std::abs(
                            loss_history_[loss_history_.size() - 1] - loss_history_[
                                loss_history_.size() - 2]);
                if (loss_diff < tol_) {
                    std::cout << "Converged at iteration " << iter << std::endl;
                    break;
                }
            }
        }

        is_fitted_ = true;
        std::cout << "Training completed." << std::endl;
    }

    template<typename T>
    std::vector<int> SVM<T>::predict(const Tensor<T> &X) const {
        if (!is_fitted_) {
            throw std::runtime_error("Model must be fitted before prediction");
        }

        std::vector<T> decision_values = decision_function(X);
        std::vector<int> predictions(decision_values.size());

        for (size_t i = 0; i < decision_values.size(); ++i) {
            predictions[i] = (decision_values[i] >= 0) ? classes_[1] : classes_[0];
        }

        return predictions;
    }

    template<typename T>
    Tensor<T> SVM<T>::predict_proba(const Tensor<T> &X) const {
        std::vector<T> decision_values = decision_function(X);
        Tensor<T> probabilities = Tensor<T>::zeros({X.rows(), 2});

        for (size_t i = 0; i < decision_values.size(); ++i) {
            // Convert decision function to probability using sigmoid
            T prob_positive = 1.0 / (1.0 + std::exp(-decision_values[i]));
            probabilities(i, 0) = 1.0 - prob_positive;
            probabilities(i, 1) = prob_positive;
        }

        return probabilities;
    }

    template<typename T>
    std::vector<T> SVM<T>::decision_function(const Tensor<T> &X) const {
        if (!is_fitted_) {
            throw std::runtime_error("Model must be fitted before prediction");
        }

        std::vector<T> decisions(X.rows());

        for (size_t i = 0; i < X.rows(); ++i) {
            if (kernel_type_ == KernelType::LINEAR) {
                // Linear decision function: w^T * x + b
                T decision = bias_->data()(0);
                for (size_t j = 0; j < X.cols(); ++j) {
                    decision += weights_->data()(j, 0) * X(i, j);
                }
                decisions[i] = decision;
            } else {
                // Kernel-based decision function (simplified)
                decisions[i] = bias_->data()(0);
                // This would require storing support vectors and computing kernel values
                // For now, using a simplified approach
            }
        }

        return decisions;
    }

    template<typename T>
    Tensor<T> SVM<T>::support_vectors() const {
        return support_vectors_;
    }

    template<typename T>
    std::vector<size_t> SVM<T>::support() const {
        return support_indices_;
    }

    template<typename T>
    std::vector<T> SVM<T>::dual_coef() const {
        return dual_coef_;
    }

    template<typename T>
    T SVM<T>::intercept() const {
        return bias_->data()(0);
    }

    template<typename T>
    std::shared_ptr<Variable<T> > SVM<T>::kernel(const std::shared_ptr<Variable<T> > &x1,
                                                 const std::shared_ptr<Variable<T> > &x2) const {
        switch (kernel_type_) {
            case KernelType::LINEAR:
                return x1->dot(x2->transpose());

            case KernelType::RBF: {
                // RBF kernel: exp(-gamma * ||x1 - x2||^2)
                auto diff = x1 - x2;
                auto squared_norm = diff->dot(diff->transpose());
                auto gamma_var = std::make_shared<Variable<T> >(Tensor<T>({-gamma_}, {1, 1}), false);
                return (gamma_var * squared_norm)->exp();
            }

            case KernelType::POLYNOMIAL: {
                // Polynomial kernel: (gamma * x1^T * x2 + coef0)^degree
                auto dot_product = x1->dot(x2->transpose());
                auto gamma_var = std::make_shared<Variable<T> >(Tensor<T>({gamma_}, {1, 1}), false);
                auto coef0_var = std::make_shared<Variable<T> >(Tensor<T>({coef0_}, {1, 1}), false);
                auto base = gamma_var * dot_product + coef0_var;

                // Simple power implementation (for degree = 2)
                if (degree_ == 2) {
                    return base * base;
                } else {
                    return base; // Simplified for other degrees
                }
            }

            case KernelType::SIGMOID: {
                // Sigmoid kernel: tanh(gamma * x1^T * x2 + coef0)
                auto dot_product = x1->dot(x2->transpose());
                auto gamma_var = std::make_shared<Variable<T> >(Tensor<T>({gamma_}, {1, 1}), false);
                auto coef0_var = std::make_shared<Variable<T> >(Tensor<T>({coef0_}, {1, 1}), false);
                return (gamma_var * dot_product + coef0_var)->tanh();
            }

            default:
                return x1->dot(x2->transpose());
        }
    }

    template<typename T>
    std::shared_ptr<Variable<T> > SVM<T>::compute_loss(const std::shared_ptr<Variable<T> > &X,
                                                       const std::vector<int> &y) const {
        size_t n_samples = X->rows();

        if (kernel_type_ == KernelType::LINEAR) {
            // Linear SVM loss: L = (1/2) * ||w||^2 + C * sum(max(0, 1 - y_i * (w^T * x_i + b)))

            // Regularization term: (1/2) * ||w||^2
            auto w_squared = weights_->transpose()->dot(weights_);
            auto reg_term = w_squared * std::make_shared<Variable<T> >(
                                Tensor<T>({0.5}, {1, 1}), false);

            // Hinge loss term
            auto hinge_loss_sum = std::make_shared<Variable<T> >(Tensor<T>({0.0}, {1, 1}), false);

            for (size_t i = 0; i < n_samples; ++i) {
                // Extract sample x_i
                Tensor<T> x_i_data = Tensor<T>::zeros({1, X->cols()});
                for (size_t j = 0; j < X->cols(); ++j) {
                    x_i_data(0, j) = X->data()(i, j);
                }
                auto x_i = std::make_shared<Variable<T> >(x_i_data, false);

                // Compute decision function: w^T * x_i + b
                // weights_ is (n_features, 1), x_i is (1, n_features)
                // x_i.dot(weights_) should give (1, 1) result
        auto decision = x_i->dot(weights_) + bias_;

                // Compute margin: y_i * decision
                T y_i = static_cast<T>(y[i]);
                auto margin = decision * std::make_shared<Variable<T> >(
                                  Tensor<T>({static_cast<T>(y[i])}, {1, 1}), false);

                // Hinge loss: max(0, 1 - margin)
                T margin_val = margin->data()(0);
                if (margin_val < 1.0) {
                    auto hinge = std::make_shared<Variable<T> >(Tensor<T>({1.0}, {1, 1}), false) - margin;
                hinge_loss_sum = hinge_loss_sum + hinge;
                }
            }

            // Total loss: regularization + C * hinge_loss
            auto C_var = std::make_shared<Variable<T> >(Tensor<T>({C_}, {1, 1}), false);
            return reg_term + C_var * hinge_loss_sum;
        } else {
            // Kernel SVM - dual formulation
            // This is more complex and would require implementing the full dual optimization
            // For now, we'll use a simplified approach

            auto loss_sum = std::make_shared<Variable<T> >(Tensor<T>({0.0}, {1, 1}), false);

            for (size_t i = 0; i < n_samples; ++i) {
                // Extract sample x_i
                Tensor<T> x_i_data = Tensor<T>::zeros({1, X->cols()});
                for (size_t j = 0; j < X->cols(); ++j) {
                    x_i_data(0, j) = X->data()(i, j);
                }
                auto x_i = std::make_shared<Variable<T> >(x_i_data, false);

                // Simplified kernel-based decision function
                auto decision = bias_;

                // Add contribution from each training sample (simplified)
                for (size_t k = 0; k < std::min(n_samples, static_cast<size_t>(10)); ++
                     k) {
                    Tensor<T> x_k_data = Tensor<T>::zeros({1, X->cols()});
                    for (size_t j = 0; j < X->cols(); ++j) {
                        x_k_data(0, j) = X->data()(k, j);
                    }
                    auto x_k = std::make_shared<Variable<T> >(x_k_data, false);

                    auto kernel_val = kernel(x_i, x_k);

                    // Use a subset of weights as dual variables
                    if (k < weights_->rows()) {
                        Tensor<T> alpha_k({weights_->data()(k, 0)}, {1, 1});
                        auto alpha_k_var = std::make_shared<Variable<T> >(alpha_k, false);
                        decision = decision + alpha_k_var * kernel_val *
                                   std::make_shared<Variable<T> >(Tensor<T>({static_cast<T>(y[k])}, {1, 1}),
                                                                   false);
                    }
                }

                // Hinge loss
                T y_i = static_cast<T>(y[i]);
                auto margin = decision * std::make_shared<Variable<T> >(
                                  Tensor<T>({y_i}, {1, 1}), false);

                T margin_val = margin->data()(0);
                if (margin_val < 1.0) {
                    auto hinge = std::make_shared<Variable<T> >(Tensor<T>({1.0}, {1, 1}), false) - margin;
                    loss_sum = loss_sum + hinge;
                }
            }

            return loss_sum;
        }
    }

    template<typename T>
    T SVM<T>::gradient_step(const Tensor<T> &X, const std::vector<int> &y) {
        // Zero gradients
        weights_->zero_grad();
        bias_->zero_grad();

        // Convert input to Variable
        auto X_var = to_variable(X, false);

        // Compute loss
        auto loss = compute_loss(X_var, y);

        // Backward pass - compute gradients
        loss->backward();

        // Update parameters using gradient descent
        // weights_ = weights_ - learning_rate * weights_.grad()
        Tensor<T> weight_grad_scaled = Tensor<T>::zeros({weights_->grad().rows(), weights_->grad().cols()});
        for (size_t i = 0; i < weights_->grad().rows(); ++i) {
            for (size_t j = 0; j < weights_->grad().cols(); ++j) {
                weight_grad_scaled(i, j) = weights_->grad()(i, j) * (-learning_rate_);
            }
        }
        weights_ = std::make_shared<Variable<T> >(weights_->data() + weight_grad_scaled, true);

        Tensor<T> bias_grad_scaled = Tensor<T>::zeros({bias_->grad().rows(), bias_->grad().cols()});
        for (size_t i = 0; i < bias_->grad().rows(); ++i) {
            for (size_t j = 0; j < bias_->grad().cols(); ++j) {
                bias_grad_scaled(i, j) = bias_->grad()(i, j) * (-learning_rate_);
            }
        }
        bias_ = std::make_shared<Variable<T> >(bias_->data() + bias_grad_scaled, true);

        return loss->data()(0);
    }

    template<typename T>
    std::shared_ptr<Variable<T> > SVM<T>::to_variable(const Tensor<T> &matrix,
                                                      bool requires_grad) const {
        return std::make_shared<Variable<T> >(matrix, requires_grad);
    }

    // Explicit template instantiations
    template class SVM<float>;
    template class SVM<double>;
} // namespace ml
