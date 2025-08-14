#pragma once

#include <cmath>
#include <vector>
#include "utils/tensor.hpp"

/**
 * @file functions.hpp
 * @brief Activation functions for neural networks
 * @author Kalenitid
 * @version 1.0.0
 */

namespace dl::activation {
    using dl::MatrixD;

    /**
     * @brief Abstract base class for activation functions
     *
     * This class defines the common interface that all activation functions
     * must implement. Activation functions are essential components in neural
     * networks that introduce non-linearity to the model.
     *
     * @note All derived classes must implement forward and backward methods
     */
    class ActivationFunction {
    public:
        /**
         * @brief Virtual destructor for proper cleanup
         */
        virtual ~ActivationFunction() = default;

        /**
         * @todo Add pure virtual forward and backward methods
         * @brief Forward pass computation (to be implemented by derived classes)
         * @brief Backward pass computation (to be implemented by derived classes)
         */
    };

    /**
     * @brief Rectified Linear Unit (ReLU) activation function
     *
     * ReLU is one of the most commonly used activation functions in deep learning.
     * It outputs the input directly if positive, otherwise outputs zero.
     *
     * Mathematical definition:
     * - Forward: f(x) = max(0, x)
     * - Derivative: f'(x) = 1 if x > 0, else 0
     *
     * @note ReLU helps mitigate the vanishing gradient problem
     */
    class ReLU : public ActivationFunction {
    public:
        /**
         * @brief Compute ReLU forward pass
         * @param x Input value
         * @return max(0, x)
         */
        double forward(double x);

        /**
         * @brief Compute ReLU derivative
         * @param x Input value
         * @return 1 if x > 0, else 0
         */
        double backward(double x);
    };

    /**
     * @brief Sigmoid activation function
     *
     * The sigmoid function maps any real number to a value between 0 and 1,
     * making it useful for binary classification problems.
     *
     * Mathematical definition:
     * - Forward: f(x) = 1 / (1 + exp(-x))
     * - Derivative: f'(x) = f(x) * (1 - f(x))
     *
     * @warning Can suffer from vanishing gradient problem for large |x|
     */
    class Sigmoid : public ActivationFunction {
    public:
        /**
         * @brief Compute sigmoid forward pass
         * @param x Input value
         * @return 1 / (1 + exp(-x))
         */
        double forward(double x);

        /**
         * @brief Compute sigmoid derivative
         * @param x Input value
         * @return sigmoid(x) * (1 - sigmoid(x))
         */
        double backward(double x);
    };

    /**
     * @brief Hyperbolic tangent (Tanh) activation function
     *
     * Tanh maps input values to the range (-1, 1), making it zero-centered
     * which can help with gradient flow during training.
     *
     * Mathematical definition:
     * - Forward: f(x) = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
     * - Derivative: f'(x) = 1 - tanh²(x)
     *
     * @note Zero-centered output can improve convergence
     */
    class Tanh : public ActivationFunction {
    public:
        /**
         * @brief Compute tanh forward pass
         * @param x Input value
         * @return tanh(x)
         */
        double forward(double x);

        /**
         * @brief Compute tanh derivative
         * @param x Input value
         * @return 1 - tanh²(x)
         */
        double backward(double x);
    };

    /**
     * @brief Softmax activation function
     *
     * Softmax is commonly used in the output layer of multi-class classification
     * networks. It converts a vector of real numbers into a probability distribution.
     *
     * Mathematical definition:
     * - Forward: f(x_i) = exp(x_i) / Σ(exp(x_j)) for j=1 to n
     * - Backward: Jacobian matrix with elements ∂f_i/∂x_j
     *
     * @note Output values sum to 1, making them interpretable as probabilities
     * @warning Numerically unstable for large input values without proper scaling
     */
    class Softmax : public ActivationFunction {
    public:
        /**
         * @brief Compute softmax forward pass
         * @param x Input matrix/vector
         * @return Probability distribution matrix
         */
        MatrixD forward(const MatrixD &x);

        /**
         * @brief Compute softmax Jacobian matrix
         * @param x Input matrix/vector
         * @return Jacobian matrix for backpropagation
         */
        MatrixD backward(const MatrixD &x);
    };

    /**
     * @brief Leaky Rectified Linear Unit (Leaky ReLU) activation function
     *
     * Leaky ReLU addresses the "dying ReLU" problem by allowing a small
     * gradient when the input is negative, preventing neurons from becoming
     * completely inactive.
     *
     * Mathematical definition:
     * - Forward: f(x) = max(αx, x) where α is a small positive constant
     * - Derivative: f'(x) = 1 if x > 0, else α
     *
     * @note Helps prevent dead neurons compared to standard ReLU
     */
    class LeakyReLU : public ActivationFunction {
    public:
        /**
         * @brief Constructor with configurable leak parameter
         * @param alpha Leak coefficient for negative inputs (default: 0.01)
         */
        LeakyReLU(double alpha = 0.01);

        /**
         * @brief Compute Leaky ReLU forward pass
         * @param x Input value
         * @return max(alpha * x, x)
         */
        double forward(double x);

        /**
         * @brief Compute Leaky ReLU derivative
         * @param x Input value
         * @return 1 if x > 0, else alpha
         */
        double backward(double x);

    private:
        /**
         * @brief Leak coefficient for negative inputs
         */
        double alpha_;
    };
} // namespace dl::activation
