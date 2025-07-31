#pragma once

#include <functional>
#include <vector>
#include "../utils/matrix.hpp"

/**
 * @file svm.hpp
 * @brief Support Vector Machine implementation
 * @author Kalenitid
 * @version 1.0.0
 */

namespace ml {
    using namespace utils;

    /**
     * @brief Kernel function types for SVM
     */
    enum class KernelType {
        LINEAR, ///< Linear kernel: K(x, y) = x^T * y
        POLYNOMIAL, ///< Polynomial kernel: K(x, y) = (gamma * x^T * y + coef0)^degree
        RBF, ///< Radial Basis Function kernel: K(x, y) = exp(-gamma * ||x - y||^2)
        SIGMOID ///< Sigmoid kernel: K(x, y) = tanh(gamma * x^T * y + coef0)
    };

    /**
     * @brief A class implementing Support Vector Machine for classification
     *
     * This class provides functionality for binary and multi-class classification
     * using Support Vector Machines with various kernel functions.
     *
     * @tparam T The data type for matrix elements (typically float or double)
     *
     * @example
     * ```cpp
     * // Create an SVM with RBF kernel
     * SVM<double> svm(KernelType::RBF, 1.0, 0.1);
     *
     * // Fit SVM to training data
     * Matrix<double> X_train = training_features;
     * std::vector<int> y_train = training_labels;
     * svm.fit(X_train, y_train);
     *
     * // Predict on test data
     * Matrix<double> X_test = test_features;
     * std::vector<int> predictions = svm.predict(X_test);
     * ```
     */
    template<typename T>
    class SVM {
    public:
        /**
         * @brief Constructor for SVM
         *
         * @param kernel_type Type of kernel function to use
         * @param C Regularization parameter
         * @param gamma Kernel coefficient (for RBF, polynomial, sigmoid)
         * @param degree Degree of polynomial kernel
         * @param coef0 Independent term in polynomial/sigmoid kernel
         * @param tol Tolerance for stopping criterion
         * @param max_iter Maximum number of iterations
         */
        SVM(KernelType kernel_type = KernelType::RBF, T C = 1.0, T gamma = 1.0, int degree = 3, T coef0 = 0.0,
            T tol = 1e-3, size_t max_iter = 1000);

        /**
         * @brief Fit the SVM model to training data
         *
         * @param X Training feature matrix (samples x features)
         * @param y Training labels vector
         */
        void fit(const Matrix<T> &X, const std::vector<int> &y);

        /**
         * @brief Predict class labels for samples
         *
         * @param X Feature matrix to predict
         * @return Vector of predicted class labels
         */
        std::vector<int> predict(const Matrix<T> &X) const;

        /**
         * @brief Predict class probabilities for samples
         *
         * @param X Feature matrix to predict
         * @return Matrix of class probabilities
         */
        Matrix<T> predict_proba(const Matrix<T> &X) const;

        /**
         * @brief Get decision function values
         *
         * @param X Feature matrix
         * @return Decision function values
         */
        std::vector<T> decision_function(const Matrix<T> &X) const;

        /**
         * @brief Get support vectors
         *
         * @return Matrix containing support vectors
         */
        Matrix<T> support_vectors() const;

        /**
         * @brief Get support vector indices
         *
         * @return Vector of support vector indices
         */
        std::vector<size_t> support() const;

        /**
         * @brief Get dual coefficients
         *
         * @return Vector of dual coefficients
         */
        std::vector<T> dual_coef() const;

        /**
         * @brief Get intercept (bias) term
         *
         * @return Intercept value
         */
        T intercept() const;

    private:
        KernelType kernel_type_; ///< Kernel function type
        T C_; ///< Regularization parameter
        T gamma_; ///< Kernel coefficient
        int degree_; ///< Polynomial degree
        T coef0_; ///< Independent term
        T tol_; ///< Tolerance
        size_t max_iter_; ///< Maximum iterations

        Matrix<T> support_vectors_; ///< Support vectors
        std::vector<size_t> support_indices_; ///< Support vector indices
        std::vector<T> dual_coef_; ///< Dual coefficients
        T intercept_; ///< Intercept term
        std::vector<int> classes_; ///< Unique class labels
        bool is_fitted_; ///< Whether model is fitted

        /**
         * @brief Compute kernel function between two vectors
         *
         * @param x1 First vector
         * @param x2 Second vector
         * @return Kernel value
         */
        T kernel(const std::vector<T> &x1, const std::vector<T> &x2) const;

        /**
         * @brief Compute kernel matrix
         *
         * @param X1 First matrix
         * @param X2 Second matrix
         * @return Kernel matrix
         */
        Matrix<T> kernel_matrix(const Matrix<T> &X1, const Matrix<T> &X2) const;

        /**
         * @brief Sequential Minimal Optimization (SMO) algorithm
         *
         * @param X Training features
         * @param y Training labels
         */
        void smo(const Matrix<T> &X, const std::vector<int> &y);

        /**
         * @brief Select two alphas to optimize
         *
         * @param i First alpha index
         * @param E1 Error for first alpha
         * @param alphas Current alpha values
         * @param errors Current error values
         * @return Second alpha index
         */
        size_t select_j(size_t i, T E1, const std::vector<T> &alphas, const std::vector<T> &errors) const;

        /**
         * @brief Clip alpha value to bounds
         *
         * @param alpha Alpha value to clip
         * @param L Lower bound
         * @param H Upper bound
         * @return Clipped alpha value
         */
        T clip_alpha(T alpha, T L, T H) const;
    };

    /// Type aliases for common use cases
    using SVMF = SVM<float>;
    using SVMD = SVM<double>;

} // namespace ml
