#include "ml/svm.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <set>
#include <stdexcept>

namespace ml {

    template<typename T>
    SVM<T>::SVM(KernelType kernel_type, T C, T gamma, int degree, T coef0, T tol, size_t max_iter) :
        kernel_type_(kernel_type), C_(C), gamma_(gamma), degree_(degree), coef0_(coef0), tol_(tol), max_iter_(max_iter),
        intercept_(0), is_fitted_(false) {
        if (C <= 0) {
            throw std::invalid_argument("C must be positive");
        }
        if (gamma <= 0) {
            throw std::invalid_argument("gamma must be positive");
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

        // Run SMO algorithm
        smo(X, binary_labels);

        is_fitted_ = true;
    }

    template<typename T>
    std::vector<int> SVM<T>::predict(const Matrix<T> &X) const {
        if (!is_fitted_) {
            throw std::runtime_error("Model must be fitted before prediction");
        }

        std::vector<T> decision_values = decision_function(X);
        std::vector<int> predictions(X.rows());

        for (size_t i = 0; i < X.rows(); ++i) {
            predictions[i] = (decision_values[i] >= 0) ? classes_[1] : classes_[0];
        }

        return predictions;
    }

    template<typename T>
    Matrix<T> SVM<T>::predict_proba(const Matrix<T> &X) const {
        if (!is_fitted_) {
            throw std::runtime_error("Model must be fitted before prediction");
        }

        std::vector<T> decision_values = decision_function(X);
        Matrix<T> probabilities(X.rows(), 2);

        for (size_t i = 0; i < X.rows(); ++i) {
            // Use sigmoid function to convert decision values to probabilities
            T sigmoid_val = 1.0 / (1.0 + std::exp(-decision_values[i]));
            probabilities(i, 0) = 1.0 - sigmoid_val; // Probability of class 0
            probabilities(i, 1) = sigmoid_val; // Probability of class 1
        }

        return probabilities;
    }

    template<typename T>
    std::vector<T> SVM<T>::decision_function(const Matrix<T> &X) const {
        if (!is_fitted_) {
            throw std::runtime_error("Model must be fitted before prediction");
        }

        std::vector<T> decision_values(X.rows(), 0);

        for (size_t i = 0; i < X.rows(); ++i) {
            T sum = 0;
            for (size_t j = 0; j < support_vectors_.rows(); ++j) {
                std::vector<T> x_i(X.cols()), sv_j(support_vectors_.cols());
                for (size_t k = 0; k < X.cols(); ++k) {
                    x_i[k] = X(i, k);
                }
                for (size_t k = 0; k < support_vectors_.cols(); ++k) {
                    sv_j[k] = support_vectors_(j, k);
                }
                sum += dual_coef_[j] * kernel(x_i, sv_j);
            }
            decision_values[i] = sum + intercept_;
        }

        return decision_values;
    }

    template<typename T>
    Matrix<T> SVM<T>::support_vectors() const {
        if (!is_fitted_) {
            throw std::runtime_error("Model must be fitted before accessing support vectors");
        }
        return support_vectors_;
    }

    template<typename T>
    std::vector<size_t> SVM<T>::support() const {
        if (!is_fitted_) {
            throw std::runtime_error("Model must be fitted before accessing support indices");
        }
        return support_indices_;
    }

    template<typename T>
    std::vector<T> SVM<T>::dual_coef() const {
        if (!is_fitted_) {
            throw std::runtime_error("Model must be fitted before accessing dual coefficients");
        }
        return dual_coef_;
    }

    template<typename T>
    T SVM<T>::intercept() const {
        if (!is_fitted_) {
            throw std::runtime_error("Model must be fitted before accessing intercept");
        }
        return intercept_;
    }

    template<typename T>
    T SVM<T>::kernel(const std::vector<T> &x1, const std::vector<T> &x2) const {
        switch (kernel_type_) {
            case KernelType::LINEAR: {
                T dot_product = 0;
                for (size_t i = 0; i < x1.size(); ++i) {
                    dot_product += x1[i] * x2[i];
                }
                return dot_product;
            }
            case KernelType::POLYNOMIAL: {
                T dot_product = 0;
                for (size_t i = 0; i < x1.size(); ++i) {
                    dot_product += x1[i] * x2[i];
                }
                return std::pow(gamma_ * dot_product + coef0_, degree_);
            }
            case KernelType::RBF: {
                T squared_distance = 0;
                for (size_t i = 0; i < x1.size(); ++i) {
                    T diff = x1[i] - x2[i];
                    squared_distance += diff * diff;
                }
                return std::exp(-gamma_ * squared_distance);
            }
            case KernelType::SIGMOID: {
                T dot_product = 0;
                for (size_t i = 0; i < x1.size(); ++i) {
                    dot_product += x1[i] * x2[i];
                }
                return std::tanh(gamma_ * dot_product + coef0_);
            }
            default:
                throw std::invalid_argument("Unknown kernel type");
        }
    }

    template<typename T>
    Matrix<T> SVM<T>::kernel_matrix(const Matrix<T> &X1, const Matrix<T> &X2) const {
        Matrix<T> K(X1.rows(), X2.rows());
        for (size_t i = 0; i < X1.rows(); ++i) {
            for (size_t j = 0; j < X2.rows(); ++j) {
                std::vector<T> x1(X1.cols()), x2(X2.cols());
                for (size_t k = 0; k < X1.cols(); ++k) {
                    x1[k] = X1(i, k);
                }
                for (size_t k = 0; k < X2.cols(); ++k) {
                    x2[k] = X2(j, k);
                }
                K(i, j) = kernel(x1, x2);
            }
        }
        return K;
    }

    template<typename T>
    void SVM<T>::smo(const Matrix<T> &X, const std::vector<int> &y) {
        size_t n_samples = X.rows();
        std::vector<T> alphas(n_samples, 0);
        T b = 0;

        // Compute kernel matrix
        Matrix<T> K = kernel_matrix(X, X);

        std::vector<T> errors(n_samples);
        for (size_t i = 0; i < n_samples; ++i) {
            T sum = 0;
            for (size_t j = 0; j < n_samples; ++j) {
                sum += alphas[j] * y[j] * K(i, j);
            }
            errors[i] = sum + b - y[i];
        }

        size_t num_changed = 0;
        bool examine_all = true;
        size_t iter = 0;

        while ((num_changed > 0 || examine_all) && iter < max_iter_) {
            num_changed = 0;
            if (examine_all) {
                for (size_t i = 0; i < n_samples; ++i) {
                    // Simplified SMO - this is a basic implementation
                    // In practice, you'd want a more sophisticated version
                    if ((y[i] * errors[i] < -tol_ && alphas[i] < C_) || (y[i] * errors[i] > tol_ && alphas[i] > 0)) {
                        size_t j = select_j(i, errors[i], alphas, errors);
                        if (i != j) {
                            // This is a simplified version - full SMO is more complex
                            num_changed++;
                        }
                    }
                }
            } else {
                // Examine non-bound examples
                for (size_t i = 0; i < n_samples; ++i) {
                    if (alphas[i] > 0 && alphas[i] < C_) {
                        if ((y[i] * errors[i] < -tol_ && alphas[i] < C_) ||
                            (y[i] * errors[i] > tol_ && alphas[i] > 0)) {
                            size_t j = select_j(i, errors[i], alphas, errors);
                            if (i != j) {
                                num_changed++;
                            }
                        }
                    }
                }
            }

            if (examine_all) {
                examine_all = false;
            } else if (num_changed == 0) {
                examine_all = true;
            }
            iter++;
        }

        // Extract support vectors
        std::vector<size_t> support_idx;
        std::vector<T> support_alphas;
        for (size_t i = 0; i < n_samples; ++i) {
            if (alphas[i] > tol_) {
                support_idx.push_back(i);
                support_alphas.push_back(alphas[i] * y[i]);
            }
        }

        support_indices_ = support_idx;
        dual_coef_ = support_alphas;

        // Extract support vectors
        support_vectors_ = Matrix<T>(support_idx.size(), X.cols());
        for (size_t i = 0; i < support_idx.size(); ++i) {
            for (size_t j = 0; j < X.cols(); ++j) {
                support_vectors_(i, j) = X(support_idx[i], j);
            }
        }

        intercept_ = b;
    }

    template<typename T>
    size_t SVM<T>::select_j(size_t i, T E1, const std::vector<T> &alphas, const std::vector<T> &errors) const {
        // Simple heuristic: choose j that maximizes |E1 - E2|
        size_t best_j = 0;
        T max_diff = 0;
        for (size_t j = 0; j < alphas.size(); ++j) {
            if (j != i) {
                T diff = std::abs(E1 - errors[j]);
                if (diff > max_diff) {
                    max_diff = diff;
                    best_j = j;
                }
            }
        }
        return best_j;
    }

    template<typename T>
    T SVM<T>::clip_alpha(T alpha, T L, T H) const {
        if (alpha > H)
            return H;
        if (alpha < L)
            return L;
        return alpha;
    }

    // Explicit template instantiation
    template class SVM<float>;
    template class SVM<double>;

} // namespace ml
