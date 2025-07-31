#include "ml/svm.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <set>
#include <stdexcept>
#include <iostream>

namespace ml {

    template<typename T>
    SVM<T>::SVM(KernelType kernel_type, T C, T gamma, int degree, T coef0, 
                               T tol, size_t max_iter, T learning_rate) :
        kernel_type_(kernel_type), C_(C), gamma_(gamma), degree_(degree), coef0_(coef0), 
        tol_(tol), max_iter_(max_iter), learning_rate_(learning_rate), is_fitted_(false),
        weights_(Matrix<T>(1, 1, 0.0), true),  // Initialize with dummy data, will be resized in fit
        bias_(Matrix<T>(1, 1, 0.0), true) {    // Initialize with dummy data
        
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
    void SVM<T>::fit(const Matrix<T> &X, const std::vector<int> &y) {
        if (X.rows() != y.size()) {
            throw std::invalid_argument("Number of samples in X and y must match");
        }

        // Get unique classes
        std::set<int> unique_classes(y.begin(), y.end());
        classes_.assign(unique_classes.begin(), unique_classes.end());

        if (classes_.size() != 2) {
            throw std::invalid_argument("Currently only binary classification is supported");
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
            Matrix<T> w_init = Matrix<T>::random(X.cols(), 1, -0.1, 0.1);
            weights_ = Variable<T>(w_init, true); // requires_grad = true
            
            Matrix<T> b_init(1, 1, 0.0);
            bias_ = Variable<T>(b_init, true);
        } else {
            // For kernel methods, initialize dual variables
            Matrix<T> alpha_init = Matrix<T>::random(X.rows(), 1, 0.0, 0.1);
            weights_ = Variable<T>(alpha_init, true);
            
            Matrix<T> b_init(1, 1, 0.0);
            bias_ = Variable<T>(b_init, true);
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
                T loss_diff = std::abs(loss_history_[loss_history_.size()-1] - 
                                     loss_history_[loss_history_.size()-2]);
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
    T SVM<T>::gradient_step(const Matrix<T> &X, const std::vector<int> &y) {
        // Zero gradients
        weights_.zero_grad();
        bias_.zero_grad();
        
        // Convert input to Variable
        Variable<T> X_var = to_variable(X, false);
        
        // Compute loss
        Variable<T> loss = compute_loss(X_var, y);
        
        // Backward pass - compute gradients
        loss.backward();
        
        // Update parameters using gradient descent
        // weights_ = weights_ - learning_rate * weights_.grad()
        Matrix<T> weight_grad_scaled(weights_.grad().rows(), weights_.grad().cols());
        for (size_t i = 0; i < weights_.grad().rows(); ++i) {
            for (size_t j = 0; j < weights_.grad().cols(); ++j) {
                weight_grad_scaled(i, j) = weights_.grad()(i, j) * (-learning_rate_);
            }
        }
        weights_ = Variable<T>(weights_.data() + weight_grad_scaled, true);
        
        Matrix<T> bias_grad_scaled(bias_.grad().rows(), bias_.grad().cols());
        for (size_t i = 0; i < bias_.grad().rows(); ++i) {
            for (size_t j = 0; j < bias_.grad().cols(); ++j) {
                bias_grad_scaled(i, j) = bias_.grad()(i, j) * (-learning_rate_);
            }
        }
        bias_ = Variable<T>(bias_.data() + bias_grad_scaled, true);
        
        return loss.data()(0, 0);
    }

    template<typename T>
    Variable<T> SVM<T>::compute_loss(const Variable<T> &X, const std::vector<int> &y) const {
        size_t n_samples = X.rows();
        
        if (kernel_type_ == KernelType::LINEAR) {
            // Linear SVM loss: L = (1/2) * ||w||^2 + C * sum(max(0, 1 - y_i * (w^T * x_i + b)))
            
            // Regularization term: (1/2) * ||w||^2
            Variable<T> w_squared = weights_.transpose().dot(weights_);
            Variable<T> reg_term = w_squared * Variable<T>(Matrix<T>(1, 1, 0.5), false);
            
            // Hinge loss term
            Variable<T> hinge_loss_sum = Variable<T>(Matrix<T>(1, 1, 0.0), false);
            
            for (size_t i = 0; i < n_samples; ++i) {
                // Extract sample x_i
                Matrix<T> x_i_data(1, X.cols());
                for (size_t j = 0; j < X.cols(); ++j) {
                    x_i_data(0, j) = X.data()(i, j);
                }
                Variable<T> x_i = Variable<T>(x_i_data, false);
                
                // Compute decision function: w^T * x_i + b
                // weights_ is (n_features, 1), x_i is (1, n_features)
                // x_i.dot(weights_) should give (1, 1) result
                Variable<T> decision = x_i.dot(weights_) + bias_;
                
                // Compute margin: y_i * decision
                T y_i = static_cast<T>(y[i]);
                Variable<T> margin = decision * Variable<T>(Matrix<T>(1, 1, y_i), false);
                
                // Hinge loss: max(0, 1 - margin)
                T margin_val = margin.data()(0, 0);
                if (margin_val < 1.0) {
                    Variable<T> hinge = Variable<T>(Matrix<T>(1, 1, 1.0), false) - margin;
                    hinge_loss_sum = hinge_loss_sum + hinge;
                }
            }
            
            // Total loss: regularization + C * hinge_loss
            Variable<T> C_var = Variable<T>(Matrix<T>(1, 1, C_), false);
            return reg_term + C_var * hinge_loss_sum;
            
        } else {
            // Kernel SVM - dual formulation
            // This is more complex and would require implementing the full dual optimization
            // For now, we'll use a simplified approach
            
            Variable<T> loss_sum = Variable<T>(Matrix<T>(1, 1, 0.0), false);
            
            for (size_t i = 0; i < n_samples; ++i) {
                // Extract sample x_i
                Matrix<T> x_i_data(1, X.cols());
                for (size_t j = 0; j < X.cols(); ++j) {
                    x_i_data(0, j) = X(i, j);
                }
                Variable<T> x_i = Variable<T>(x_i_data, false);
                
                // Simplified kernel-based decision function
                Variable<T> decision = bias_;
                
                // Add contribution from each training sample (simplified)
                for (size_t k = 0; k < std::min(n_samples, static_cast<size_t>(10)); ++k) {
                    Matrix<T> x_k_data(1, X.cols());
                    for (size_t j = 0; j < X.cols(); ++j) {
                        x_k_data(0, j) = X.data()(k, j);
                    }
                    Variable<T> x_k = Variable<T>(x_k_data, false);
                    
                    Variable<T> kernel_val = kernel(x_i, x_k);
                    
                    // Use a subset of weights as dual variables
                    if (k < weights_.rows()) {
                        Matrix<T> alpha_k(1, 1, weights_.data()(k, 0));
                        Variable<T> alpha_k_var = Variable<T>(alpha_k, false);
                        decision = decision + alpha_k_var * kernel_val * Variable<T>(Matrix<T>(1, 1, static_cast<T>(y[k])), false);
                    }
                }
                
                // Hinge loss
                T y_i = static_cast<T>(y[i]);
                Variable<T> margin = decision * Variable<T>(Matrix<T>(1, 1, y_i), false);
                
                T margin_val = margin.data()(0, 0);
                if (margin_val < 1.0) {
                    Variable<T> hinge = Variable<T>(Matrix<T>(1, 1, 1.0), false) - margin;
                    loss_sum = loss_sum + hinge;
                }
            }
            
            return loss_sum;
        }
    }

    template<typename T>
    Variable<T> SVM<T>::kernel(const Variable<T> &x1, const Variable<T> &x2) const {
        switch (kernel_type_) {
            case KernelType::LINEAR:
                return x1.dot(x2.transpose());
                
            case KernelType::RBF: {
                // RBF kernel: exp(-gamma * ||x1 - x2||^2)
                Variable<T> diff = x1 - x2;
                Variable<T> squared_norm = diff.dot(diff.transpose());
                Variable<T> gamma_var = Variable<T>(Matrix<T>(1, 1, -gamma_), false);
                return (gamma_var * squared_norm).exp();
            }
            
            case KernelType::POLYNOMIAL: {
                // Polynomial kernel: (gamma * x1^T * x2 + coef0)^degree
                Variable<T> dot_product = x1.dot(x2.transpose());
                Variable<T> gamma_var = Variable<T>(Matrix<T>(1, 1, gamma_), false);
                Variable<T> coef0_var = Variable<T>(Matrix<T>(1, 1, coef0_), false);
                Variable<T> base = gamma_var * dot_product + coef0_var;
                
                // Simple power implementation (for degree = 2)
                if (degree_ == 2) {
                    return base * base;
                } else {
                    return base; // Simplified for other degrees
                }
            }
            
            case KernelType::SIGMOID: {
                // Sigmoid kernel: tanh(gamma * x1^T * x2 + coef0)
                Variable<T> dot_product = x1.dot(x2.transpose());
                Variable<T> gamma_var = Variable<T>(Matrix<T>(1, 1, gamma_), false);
                Variable<T> coef0_var = Variable<T>(Matrix<T>(1, 1, coef0_), false);
                return (gamma_var * dot_product + coef0_var).tanh();
            }
            
            default:
                return x1.dot(x2.transpose());
        }
    }

    template<typename T>
    Variable<T> SVM<T>::to_variable(const Matrix<T> &matrix, bool requires_grad) const {
        return Variable<T>(matrix, requires_grad);
    }

    template<typename T>
    std::vector<int> SVM<T>::predict(const Matrix<T> &X) const {
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
    std::vector<T> SVM<T>::decision_function(const Matrix<T> &X) const {
        if (!is_fitted_) {
            throw std::runtime_error("Model must be fitted before prediction");
        }
        
        std::vector<T> decisions(X.rows());
        
        for (size_t i = 0; i < X.rows(); ++i) {
            if (kernel_type_ == KernelType::LINEAR) {
                // Linear decision function: w^T * x + b
                T decision = bias_.data()(0, 0);
                for (size_t j = 0; j < X.cols(); ++j) {
                    decision += weights_.data()(j, 0) * X(i, j);
                }
                decisions[i] = decision;
            } else {
                // Kernel-based decision function (simplified)
                decisions[i] = bias_.data()(0, 0);
                // This would require storing support vectors and computing kernel values
                // For now, using a simplified approach
            }
        }
        
        return decisions;
    }

    template<typename T>
    Matrix<T> SVM<T>::predict_proba(const Matrix<T> &X) const {
        std::vector<T> decision_values = decision_function(X);
        Matrix<T> probabilities(X.rows(), 2);
        
        for (size_t i = 0; i < decision_values.size(); ++i) {
            // Convert decision function to probability using sigmoid
            T prob_positive = 1.0 / (1.0 + std::exp(-decision_values[i]));
            probabilities(i, 0) = 1.0 - prob_positive;
            probabilities(i, 1) = prob_positive;
        }
        
        return probabilities;
    }

    template<typename T>
    Matrix<T> SVM<T>::support_vectors() const {
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
        return bias_.data()(0, 0);
    }

    // Explicit template instantiations
    template class SVM<float>;
    template class SVM<double>;

} // namespace ml