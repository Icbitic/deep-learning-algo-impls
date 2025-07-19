#include "activation/functions.hpp"
#include <algorithm>
#include <cmath>

namespace dl::activation {
    // ReLU Implementation
    double ReLU::forward(double x) {
        // TODO: Implement ReLU forward pass
        // Formula: f(x) = max(0, x)
        return std::max(0.0, x);
    }

    double ReLU::backward(double x) {
        // TODO: Implement ReLU backward pass
        // Formula: f'(x) = 1 if x > 0, else 0
        return x > 0.0 ? 1.0 : 0.0;
    }

    // Sigmoid Implementation
    double Sigmoid::forward(double x) {
        // TODO: Implement Sigmoid forward pass
        // Formula: f(x) = 1 / (1 + exp(-x))
        return 1.0 / (1.0 + std::exp(-x));
    }

    double Sigmoid::backward(double x) {
        // TODO: Implement Sigmoid backward pass
        // Formula: f'(x) = f(x) * (1 - f(x))
        double fx = forward(x);
        return fx * (1.0 - fx);
    }

    // Tanh Implementation
    double Tanh::forward(double x) {
        // TODO: Implement Tanh forward pass
        // Formula: f(x) = tanh(x)
        return std::tanh(x);
    }

    double Tanh::backward(double x) {
        // TODO: Implement Tanh backward pass
        // Formula: f'(x) = 1 - tanh^2(x)
        double fx = forward(x);
        return 1.0 - fx * fx;
    }

    // Softmax Implementation
    MatrixD Softmax::forward(const MatrixD &x) {
        // TODO: Implement Softmax forward pass
        // Formula: f(x_i) = exp(x_i) / sum(exp(x_j)) for all j
        // Placeholder return
        return MatrixD(x.rows(), x.cols());
    }

    MatrixD Softmax::backward(const MatrixD &x) {
        // TODO: Implement Softmax backward pass
        // Formula: Jacobian matrix computation
        // Placeholder return
        return MatrixD(x.rows(), x.cols());
    }

    // LeakyReLU Implementation
    LeakyReLU::LeakyReLU(double alpha) : alpha_(alpha) {
        // Constructor implementation
    }

    double LeakyReLU::forward(double x) {
        // TODO: Implement LeakyReLU forward pass
        // Formula: f(x) = x if x > 0, else alpha * x
        return x > 0.0 ? x : alpha_ * x;
    }

    double LeakyReLU::backward(double x) {
        // TODO: Implement LeakyReLU backward pass
        // Formula: f'(x) = 1 if x > 0, else alpha
        return x > 0.0 ? 1.0 : alpha_;
    }
} // namespace dl::activation
