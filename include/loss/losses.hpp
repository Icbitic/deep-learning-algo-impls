#pragma once

#include <memory>
#include <vector>
#include "utils/autograd.hpp"
#include "utils/tensor.hpp"

/**
 * @file losses.hpp
 * @brief PyTorch-like loss functions with automatic differentiation
 * @author Kalenitid
 * @version 1.0.0
 */

namespace dl::loss {
    using dl::Tensor;
    using dl::TensorD;
    using dl::TensorF;
    using dl::Variable;
    using dl::VariableD;
    using dl::VariableF;

    /**
     * @brief Base class for autograd-compatible loss functions
     */
    template<typename T>
    class AutogradLoss {
    public:
        virtual ~AutogradLoss() = default;

        /**
         * @brief Compute loss between predictions and targets
         * @param predictions Model predictions
         * @param targets Ground truth targets
         * @return Loss value as a Variable (scalar)
         */
        virtual std::shared_ptr<Variable<T> > forward(const std::shared_ptr<Variable<T> > &predictions,
                                                      const std::shared_ptr<Variable<T> > &targets) = 0;

        /**
         * @brief Convenience operator for computing loss
         */
        std::shared_ptr<Variable<T> > operator()(const std::shared_ptr<Variable<T> > &predictions,
                                                 const std::shared_ptr<Variable<T> > &targets) {
            return forward(predictions, targets);
        }
    };

    /**
     * @brief Mean Squared Error Loss with autograd support
     *
     * MSE(y_pred, y_true) = (1/n) * sum((y_pred - y_true)²)
     *
     * Commonly used for regression tasks.
     */
    template<typename T>
    class MSELoss : public AutogradLoss<T> {
    public:
        /**
         * @brief Constructor
         * @param reduction Type of reduction ('mean', 'sum', 'none')
         */
        explicit MSELoss(const std::string &reduction = "mean") : reduction_(
            reduction) {
        }

        /**
         * @brief Forward pass: compute MSE loss
         * @param predictions Predicted values
         * @param targets Target values
         * @return MSE loss
         */
        std::shared_ptr<Variable<T> > forward(const std::shared_ptr<Variable<T> > &predictions,
                                              const std::shared_ptr<Variable<T> > &targets) override;

    private:
        std::string reduction_;
    };

    /**
     * @brief Cross Entropy Loss with autograd support
     *
     * CrossEntropy(y_pred, y_true) = -sum(y_true * log(softmax(y_pred)))
     *
     * Commonly used for multi-class classification.
     */
    template<typename T>
    class CrossEntropyLoss : public AutogradLoss<T> {
    public:
        /**
         * @brief Constructor
         * @param reduction Type of reduction ('mean', 'sum', 'none')
         */
        explicit CrossEntropyLoss(const std::string &reduction = "mean") : reduction_(
            reduction) {
        }

        /**
         * @brief Forward pass: compute cross entropy loss
         * @param predictions Raw logits (before softmax)
         * @param targets Target class indices or one-hot vectors
         * @return Cross entropy loss
         */
        std::shared_ptr<Variable<T> > forward(const std::shared_ptr<Variable<T> > &predictions,
                                              const std::shared_ptr<Variable<T> > &targets) override;

    private:
        std::string reduction_;

        /**
         * @brief Apply softmax to logits
         */
        std::shared_ptr<Variable<T> > softmax(const std::shared_ptr<Variable<T> > &logits);

        /**
         * @brief Apply log softmax (numerically stable)
         */
        std::shared_ptr<Variable<T> > log_softmax(const std::shared_ptr<Variable<T> > &logits);
    };

    /**
     * @brief Binary Cross Entropy Loss with autograd support
     *
     * BCE(y_pred, y_true) = -[y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)]
     *
     * Commonly used for binary classification.
     */
    template<typename T>
    class BCELoss : public AutogradLoss<T> {
    public:
        /**
         * @brief Constructor
         * @param reduction Type of reduction ('mean', 'sum', 'none')
         */
        explicit BCELoss(const std::string &reduction = "mean") : reduction_(
            reduction) {
        }

        /**
         * @brief Forward pass: compute binary cross entropy loss
         * @param predictions Predicted probabilities (after sigmoid)
         * @param targets Binary target values (0 or 1)
         * @return BCE loss
         */
        std::shared_ptr<Variable<T> > forward(const std::shared_ptr<Variable<T> > &predictions,
                                              const std::shared_ptr<Variable<T> > &targets) override;

    private:
        std::string reduction_;
    };

    /**
     * @brief Binary Cross Entropy with Logits Loss
     *
     * Combines sigmoid and BCE for numerical stability.
     * More stable than applying sigmoid then BCE separately.
     */
    template<typename T>
    class BCEWithLogitsLoss : public AutogradLoss<T> {
    public:
        /**
         * @brief Constructor
         * @param reduction Type of reduction ('mean', 'sum', 'none')
         */
        explicit
        BCEWithLogitsLoss(const std::string &reduction = "mean") : reduction_(
            reduction) {
        }

        /**
         * @brief Forward pass: compute BCE loss from logits
         * @param predictions Raw logits (before sigmoid)
         * @param targets Binary target values (0 or 1)
         * @return BCE loss
         */
        std::shared_ptr<Variable<T> > forward(const std::shared_ptr<Variable<T> > &predictions,
                                              const std::shared_ptr<Variable<T> > &targets) override;

    private:
        std::string reduction_;
    };

    /**
     * @brief Hinge Loss with autograd support
     *
     * Hinge(y_pred, y_true) = max(0, 1 - y_true * y_pred)
     *
     * Commonly used for SVM and margin-based classification.
     */
    template<typename T>
    class HingeLoss : public AutogradLoss<T> {
    public:
        /**
         * @brief Constructor
         * @param reduction Type of reduction ('mean', 'sum', 'none')
         */
        explicit HingeLoss(const std::string &reduction = "mean") : reduction_(
            reduction) {
        }

        /**
         * @brief Forward pass: compute hinge loss
         * @param predictions Predicted values
         * @param targets Target values (-1 or +1)
         * @return Hinge loss
         */
        std::shared_ptr<Variable<T> > forward(const std::shared_ptr<Variable<T> > &predictions,
                                              const std::shared_ptr<Variable<T> > &targets) override;

    private:
        std::string reduction_;
    };

    /**
     * @brief Huber Loss with autograd support
     *
     * Combines MSE and MAE for robustness to outliers.
     *
     * Huber(y_pred, y_true) = {
     *   0.5 * (y_pred - y_true)²     if |y_pred - y_true| <= delta
     *   delta * |y_pred - y_true| - 0.5 * delta²   otherwise
     * }
     */
    template<typename T>
    class HuberLoss : public AutogradLoss<T> {
    public:
        /**
         * @brief Constructor
         * @param delta Threshold for switching between MSE and MAE
         * @param reduction Type of reduction ('mean', 'sum', 'none')
         */
        explicit
        HuberLoss(T delta = 1.0,
                  const std::string &reduction = "mean") : delta_(delta),
                                                           reduction_(reduction) {
        }

        /**
         * @brief Forward pass: compute Huber loss
         * @param predictions Predicted values
         * @param targets Target values
         * @return Huber loss
         */
        std::shared_ptr<Variable<T> > forward(const std::shared_ptr<Variable<T> > &predictions,
                                              const std::shared_ptr<Variable<T> > &targets) override;

    private:
        T delta_;
        std::string reduction_;
    };

    // Type aliases for convenience
    using MSELossD = MSELoss<double>;
    using MSELossF = MSELoss<float>;
    using CrossEntropyLossD = CrossEntropyLoss<double>;
    using CrossEntropyLossF = CrossEntropyLoss<float>;
    using BCELossD = BCELoss<double>;
    using BCELossF = BCELoss<float>;
    using BCEWithLogitsLossD = BCEWithLogitsLoss<double>;
    using BCEWithLogitsLossF = BCEWithLogitsLoss<float>;
    using HingeLossD = HingeLoss<double>;
    using HingeLossF = HingeLoss<float>;
    using HuberLossD = HuberLoss<double>;
    using HuberLossF = HuberLoss<float>;
} // namespace dl::loss
