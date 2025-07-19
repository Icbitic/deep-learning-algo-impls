#pragma once

#include <vector>
#include <cmath>
#include "utils/matrix.hpp"

namespace dl::activation {
    using utils::MatrixD;
    /**
     * Base Activation Function Interface
     * TODO: Define common interface for activation functions
     */
    class ActivationFunction {
    public:
        virtual ~ActivationFunction() = default;

        // TODO: Add pure virtual forward and backward methods
    };

    /**
     * ReLU Activation Function
     * TODO: Implement ReLU with:
     * - Forward pass: max(0, x)
     * - Backward pass: derivative
     */
    class ReLU : public ActivationFunction {
    public:
        double forward(double x);

        double backward(double x);
    };

    /**
     * Sigmoid Activation Function
     * TODO: Implement Sigmoid with:
     * - Forward pass: 1 / (1 + exp(-x))
     * - Backward pass: sigmoid(x) * (1 - sigmoid(x))
     */
    class Sigmoid : public ActivationFunction {
    public:
        double forward(double x);

        double backward(double x);
    };

    /**
     * Tanh Activation Function
     * TODO: Implement Tanh with:
     * - Forward pass: tanh(x)
     * - Backward pass: 1 - tanh²(x)
     */
    class Tanh : public ActivationFunction {
    public:
        double forward(double x);

        double backward(double x);
    };

    /**
     * Softmax Activation Function
     * TODO: Implement Softmax with:
     * - Forward pass: exp(xi) / sum(exp(xj))
     * - Backward pass: Jacobian matrix
     */
    class Softmax : public ActivationFunction {
    public:
        MatrixD forward(const MatrixD &x);

        MatrixD backward(const MatrixD &x);
    };

    /**
     * Leaky ReLU Activation Function
     * TODO: Implement Leaky ReLU with:
     * - Forward pass: max(alpha * x, x)
     * - Configurable alpha parameter
     */
    class LeakyReLU : public ActivationFunction {
    public:
        LeakyReLU(double alpha = 0.01);

        double forward(double x);

        double backward(double x);

    private:
        double alpha_;
    };
} // namespace dl::activation
