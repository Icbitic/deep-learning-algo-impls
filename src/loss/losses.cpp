#include "loss/losses.hpp"
#include <algorithm>
#include <cmath>

namespace dl::loss {
    // ============================================================================
    // MSE Loss Implementation
    // ============================================================================

    template<typename T>
    std::shared_ptr<Variable<T> > MSELoss<T>::forward(const std::shared_ptr<Variable<T> > &predictions,
                                                      const std::shared_ptr<Variable<T> > &targets) {
        // TODO: Implement MSE loss computation
        // Steps:
        // 1. Compute difference: diff = predictions - targets
        // 2. Square the difference: squared_diff = diff * diff
        // 3. Apply reduction (mean, sum, or none)

        auto diff = predictions - targets;
        auto squared_diff = diff * diff;

        if (reduction_ == "mean") {
            return squared_diff->mean();
        } else if (reduction_ == "sum") {
            return squared_diff->sum();
        } else {
            return squared_diff;
        }
    }

    // ============================================================================
    // Cross Entropy Loss Implementation
    // ============================================================================

    template<typename T>
    std::shared_ptr<Variable<T> > CrossEntropyLoss<T>::softmax(const std::shared_ptr<Variable<T> > &logits) {
        // TODO: Implement numerically stable softmax
        // Steps:
        // 1. Subtract max for numerical stability: shifted = logits - max(logits)
        // 2. Compute exp: exp_shifted = exp(shifted)
        // 3. Normalize: softmax = exp_shifted / sum(exp_shifted)

        // Placeholder implementation
        auto exp_logits = logits->exp();
        auto sum_exp = exp_logits->sum();
        // TODO: Implement proper broadcasting division
        return exp_logits; // Placeholder
    }

    template<typename T>
    std::shared_ptr<Variable<T> > CrossEntropyLoss<T>::log_softmax(const std::shared_ptr<Variable<T> > &logits) {
        // TODO: Implement numerically stable log_softmax
        // log_softmax(x) = x - log(sum(exp(x)))
        // More stable than log(softmax(x))

        // Placeholder implementation
        return logits->log(); // Placeholder
    }

    template<typename T>
    std::shared_ptr<Variable<T> > CrossEntropyLoss<T>::forward(const std::shared_ptr<Variable<T> > &predictions,
                                                               const std::shared_ptr<Variable<T> > &targets) {
        // TODO: Implement cross entropy loss
        // Steps:
        // 1. Apply log_softmax to predictions
        // 2. Compute negative log likelihood: -targets * log_softmax_pred
        // 3. Apply reduction

        auto log_probs = log_softmax(predictions);
        auto nll = targets * log_probs;
        auto neg_one = std::make_shared<Variable<T> >(Tensor<T>({-1.0}, {1, 1}), false);
        auto loss = nll * neg_one;
        // Negate

        if (reduction_ == "mean") {
            return loss->mean();
        } else if (reduction_ == "sum") {
            return loss->sum();
        } else {
            return loss;
        }
    }

    // ============================================================================
    // Binary Cross Entropy Loss Implementation
    // ============================================================================

    template<typename T>
    std::shared_ptr<Variable<T> > BCELoss<T>::forward(const std::shared_ptr<Variable<T> > &predictions,
                                                      const std::shared_ptr<Variable<T> > &targets) {
        // TODO: Implement BCE loss
        // BCE = -[y * log(p) + (1-y) * log(1-p)]
        // Steps:
        // 1. Compute log(predictions) and log(1 - predictions)
        // 2. Apply BCE formula
        // 3. Apply reduction

        // Add small epsilon for numerical stability
        T eps = 1e-7;
        auto eps_var = std::make_shared<Variable<T> >(Tensor<T>({eps}, {1, 1}), false);
        auto one_var = std::make_shared<Variable<T> >(Tensor<T>({1.0}, {1, 1}), false);

        // Clamp predictions to avoid log(0)
        // TODO: Implement proper clamping
        auto log_pred = predictions->log();
        auto one_minus_pred = one_var - predictions;
        auto log_one_minus_pred = one_minus_pred->log();

        auto loss = targets * log_pred + (one_var - targets) * log_one_minus_pred;
        auto neg_one = std::make_shared<Variable<T> >(Tensor<T>({-1.0}, {1, 1}), false);
        loss = loss * neg_one; // Negate

        if (reduction_ == "mean") {
            return loss->mean();
        } else if (reduction_ == "sum") {
            return loss->sum();
        } else {
            return loss;
        }
    }

    // ============================================================================
    // BCE with Logits Loss Implementation
    // ============================================================================

    template<typename T>
    std::shared_ptr<Variable<T> > BCEWithLogitsLoss<T>::forward(const std::shared_ptr<Variable<T> > &predictions,
                                                                const std::shared_ptr<Variable<T> > &targets) {
        // TODO: Implement BCE with logits (more numerically stable)
        // Use the identity: BCE_with_logits(x, y) = max(x, 0) - x*y + log(1 + exp(-|x|))
        // This avoids computing sigmoid explicitly

        // Placeholder: Apply sigmoid then BCE
        auto sigmoid_pred = predictions->sigmoid();
        BCELoss<T> bce_loss(reduction_);
        return bce_loss.forward(sigmoid_pred, targets);
    }

    // ============================================================================
    // Hinge Loss Implementation
    // ============================================================================

    template<typename T>
    std::shared_ptr<Variable<T> > HingeLoss<T>::forward(const std::shared_ptr<Variable<T> > &predictions,
                                                        const std::shared_ptr<Variable<T> > &targets) {
        // TODO: Implement hinge loss
        // Hinge(y_pred, y_true) = max(0, 1 - y_true * y_pred)
        // Steps:
        // 1. Compute margin: 1 - targets * predictions
        // 2. Apply max(0, margin) - this requires implementing max operation
        // 3. Apply reduction

        auto one_var = std::make_shared<Variable<T> >(Tensor<T>({1.0}, {1, 1}), false);
        auto margin = one_var - targets * predictions;

        // TODO: Implement max(0, margin) operation
        // For now, placeholder implementation
        auto loss = margin; // Placeholder

        if (reduction_ == "mean") {
            return loss->mean();
        } else if (reduction_ == "sum") {
            return loss->sum();
        } else {
            return loss;
        }
    }

    // ============================================================================
    // Huber Loss Implementation
    // ============================================================================

    template<typename T>
    std::shared_ptr<Variable<T> > HuberLoss<T>::forward(const std::shared_ptr<Variable<T> > &predictions,
                                                        const std::shared_ptr<Variable<T> > &targets) {
        // TODO: Implement Huber loss
        // Huber(y_pred, y_true) = {
        //   0.5 * (y_pred - y_true)²     if |y_pred - y_true| <= delta
        //   delta * |y_pred - y_true| - 0.5 * delta²   otherwise
        // }
        // Steps:
        // 1. Compute absolute difference: |predictions - targets|
        // 2. Create mask for |diff| <= delta
        // 3. Apply conditional logic
        // 4. Apply reduction

        auto diff = predictions - targets;
        auto abs_diff = diff; // TODO: Implement abs() operation

        // Placeholder: Use MSE for now
        auto squared_diff = diff * diff;
        auto half_var = std::make_shared<Variable<T> >(Tensor<T>({0.5}, {1, 1}), false);
        auto loss = half_var * squared_diff;

        if (reduction_ == "mean") {
            return loss->mean();
        } else if (reduction_ == "sum") {
            return loss->sum();
        } else {
            return loss;
        }
    }

    // ============================================================================
    // Explicit Template Instantiations
    // ============================================================================

    template class MSELoss<float>;
    template class MSELoss<double>;
    template class CrossEntropyLoss<float>;
    template class CrossEntropyLoss<double>;
    template class BCELoss<float>;
    template class BCELoss<double>;
    template class BCEWithLogitsLoss<float>;
    template class BCEWithLogitsLoss<double>;
    template class HingeLoss<float>;
    template class HingeLoss<double>;
    template class HuberLoss<float>;
    template class HuberLoss<double>;
} // namespace dl::loss
