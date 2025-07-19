#pragma once

#include <memory>
#include <vector>
#include "utils/matrix.hpp"

namespace dl::loss {
    using utils::MatrixD;
    /**
     * Base Loss Function Interface
     * TODO: Define common interface for loss functions
     */
    class LossFunction {
    public:
        virtual ~LossFunction() = default;

        // TODO: Add pure virtual forward and backward methods
        // TODO: Add method to compute loss value
    };

    /**
     * Mean Squared Error Loss
     * TODO: Implement MSE with:
     * - Forward pass: (1/n) * sum((y_pred - y_true)²)
     * - Backward pass: (2/n) * (y_pred - y_true)
     */
    class MeanSquaredError : public LossFunction {
    public:
        double forward(const MatrixD &predictions, const MatrixD &targets);

        MatrixD backward(const MatrixD &predictions, const MatrixD &targets);
    };

    /**
     * Cross Entropy Loss
     * TODO: Implement Cross Entropy with:
     * - Forward pass: -sum(y_true * log(y_pred))
     * - Backward pass: -y_true / y_pred
     * - Numerical stability considerations
     */
    class CrossEntropyLoss : public LossFunction {
    public:
        double forward(const MatrixD &predictions, const MatrixD &targets);

        MatrixD backward(const MatrixD &predictions, const MatrixD &targets);
    };

    /**
     * Binary Cross Entropy Loss
     * TODO: Implement Binary Cross Entropy with:
     * - Forward pass: -(y*log(p) + (1-y)*log(1-p))
     * - Backward pass: derivative computation
     */
    class BinaryCrossEntropyLoss : public LossFunction {
    public:
        double forward(const MatrixD &predictions, const MatrixD &targets);

        MatrixD backward(const MatrixD &predictions, const MatrixD &targets);
    };

    /**
     * Hinge Loss (for SVM)
     * TODO: Implement Hinge Loss with:
     * - Forward pass: max(0, 1 - y_true * y_pred)
     * - Backward pass: conditional gradient
     */
    class HingeLoss : public LossFunction {
    public:
        double forward(const MatrixD &predictions, const MatrixD &targets);

        MatrixD backward(const MatrixD &predictions, const MatrixD &targets);
    };

    /**
     * Huber Loss
     * TODO: Implement Huber Loss with:
     * - Combines MSE and MAE
     * - Robust to outliers
     * - Configurable delta parameter
     */
    class HuberLoss : public LossFunction {
    public:
        HuberLoss(double delta = 1.0);

        double forward(const MatrixD &predictions, const MatrixD &targets);

        MatrixD backward(const MatrixD &predictions, const MatrixD &targets);

    private:
        double delta_;
    };
} // namespace dl::loss
