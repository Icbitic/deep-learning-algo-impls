#pragma once

#include <functional>
#include <vector>
#include "../utils/autograd.hpp"
#include "../utils/tensor.hpp"

/**
 * @file svm.hpp
 * @brief Support Vector Machine implementation with automatic differentiation
 * @author Kalenitid
 * @version 1.0.0
 */

namespace ml {
    using namespace dl;

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
     * @brief A class implementing Support Vector Machine with automatic differentiation
     *
     * This class provides functionality for binary classification using Support Vector Machines
     * with automatic gradient computation similar to PyTorch's autograd engine.
     *
     * @tparam T The data type for matrix elements (typically float or double)
     *
     * @example
     * ```cpp
     * // Create an SVM with RBF kernel and autograd
     * SVM<double> svm(KernelType::RBF, 1.0, 0.1);
     *
     * // Fit SVM to training data with automatic gradient computation
     * Tensor<double> X_train = training_features;
     * std::vector<int> y_train = training_labels;
     * svm.fit(X_train, y_train);
     *
     * // Predict on test data
     * Tensor<double> X_test = test_features;
     * std::vector<int> predictions = svm.predict(X_test);
     * ```
     */
    template<typename T>
    class SVM {
    public:
        /**
         * @brief Constructor
         * @param kernel_type Type of kernel function to use
         * @param C Regularization parameter
         * @param gamma Kernel coefficient for RBF, polynomial and sigmoid kernels
         * @param degree Degree for polynomial kernel
         * @param coef0 Independent term for polynomial and sigmoid kernels
         * @param tol Tolerance for stopping criterion
         * @param max_iter Maximum number of iterations
         * @param learning_rate Learning rate for gradient descent
         */
        SVM(KernelType kernel_type = KernelType::RBF, T C = 1.0, T gamma = 1.0,
            int degree = 3, T coef0 = 0.0,
            T tol = 1e-3, size_t max_iter = 1000, T learning_rate = 0.01);

        /**
         * @brief Fit the SVM model to training data using automatic differentiation
         * @param X Training features matrix (n_samples x n_features)
         * @param y Training labels vector
         */
        void fit(const Tensor<T> &X, const std::vector<int> &y);

        /**
         * @brief Predict class labels for samples
         * @param X Input samples matrix (n_samples x n_features)
         * @return Vector of predicted class labels
         */
        std::vector<int> predict(const Tensor<T> &X) const;

        /**
         * @brief Predict class probabilities for samples
         * @param X Input samples matrix (n_samples x n_features)
         * @return Matrix of class probabilities (n_samples x n_classes)
         */
        Tensor<T> predict_proba(const Tensor<T> &X) const;

        /**
         * @brief Compute the decision function for samples
         * @param X Input samples matrix (n_samples x n_features)
         * @return Vector of decision function values
         */
        std::vector<T> decision_function(const Tensor<T> &X) const;

        /**
         * @brief Get support vectors
         * @return Matrix of support vectors
         */
        Tensor<T> support_vectors() const;

        /**
         * @brief Get support vector indices
         * @return Vector of support vector indices
         */
        std::vector<size_t> support() const;

        /**
         * @brief Get dual coefficients
         * @return Vector of dual coefficients
         */
        std::vector<T> dual_coef() const;

        /**
         * @brief Get intercept term
         * @return Intercept value
         */
        T intercept() const;

        /**
         * @brief Get training loss history
         * @return Vector of loss values during training
         */
        std::vector<T> loss_history() const { return loss_history_; }

    private:
        KernelType kernel_type_; ///< Kernel function type
        T C_; ///< Regularization parameter
        T gamma_; ///< Kernel coefficient
        int degree_; ///< Polynomial degree
        T coef0_; ///< Independent term
        T tol_; ///< Tolerance
        size_t max_iter_; ///< Maximum iterations
        T learning_rate_; ///< Learning rate for gradient descent

        std::shared_ptr<Variable<T> > weights_; ///< Weight parameters (with autograd)
        std::shared_ptr<Variable<T> > bias_; ///< Bias parameter (with autograd)
        Tensor<T> support_vectors_; ///< Support vectors
        std::vector<size_t> support_indices_; ///< Support vector indices
        std::vector<T> dual_coef_; ///< Dual coefficients
        std::vector<int> classes_; ///< Unique class labels
        bool is_fitted_; ///< Whether model is fitted
        std::vector<T> loss_history_; ///< Training loss history

        /**
         * @brief Compute kernel function between two vectors using autograd
         * @param x1 First vector as Variable
         * @param x2 Second vector as Variable
         * @return Kernel value as Variable
         */
        std::shared_ptr<Variable<T> > kernel(const std::shared_ptr<Variable<T> > &x1,
                                             const std::shared_ptr<Variable<T> > &x2) const;

        /**
         * @brief Compute kernel matrix using autograd
         * @param X1 First matrix as Variable
         * @param X2 Second matrix as Variable
         * @return Kernel matrix as Variable
         */
        std::shared_ptr<Variable<T> > kernel_matrix(const std::shared_ptr<Variable<T> > &X1,
                                                    const std::shared_ptr<Variable<T> > &X2) const;

        /**
         * @brief Compute SVM loss function using autograd
         * @param X Training features as Variable
         * @param y Training labels
         * @return Loss value as Variable
         */
        std::shared_ptr<Variable<T> > compute_loss(const std::shared_ptr<Variable<T> > &X,
                                                   const std::vector<int> &y) const;

        /**
         * @brief Perform gradient descent step
         * @param X Training features
         * @param y Training labels
         * @return Current loss value
         */
        T gradient_step(const Tensor<T> &X, const std::vector<int> &y);

        /**
         * @brief Convert matrix to Variable
         * @param matrix Input matrix
         * @param requires_grad Whether to track gradients
         * @return Variable wrapping the matrix
         */
        std::shared_ptr<Variable<T> > to_variable(const Tensor<T> &matrix,
                                                  bool requires_grad = false) const;
    };

    // Type aliases
    using SVMF = SVM<float>;
    using SVMD = SVM<double>;
} // namespace ml
