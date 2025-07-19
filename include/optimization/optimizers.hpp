#pragma once

#include <memory>
#include <vector>
#include "utils/matrix.hpp"

using dl::utils::MatrixD;

namespace dl::optimization {
    /**
     * Base Optimizer Interface
     * TODO: Define common interface for all optimizers
     */
    class Optimizer {
    public:
        virtual ~Optimizer() = default;

        virtual void update(MatrixD &weights, const MatrixD &gradients) = 0;

    protected:
        // TODO: Add common optimizer parameters
    };

    /**
     * Stochastic Gradient Descent
     * TODO: Implement SGD with:
     * - Basic gradient descent
     * - Momentum support
     * - Learning rate scheduling
     */
    class SGD : public Optimizer {
    public:
        SGD(double learning_rate);

        void update(MatrixD &weights, const MatrixD &gradients) override;

    private:
        double learning_rate_;
        // TODO: Add momentum, velocity
    };

    /**
     * Adam Optimizer
     * TODO: Implement Adam with:
     * - Adaptive learning rates
     * - Bias correction
     * - First and second moment estimates
     */
    class Adam : public Optimizer {
    public:
        Adam(double learning_rate, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8);

        void update(MatrixD &weights, const MatrixD &gradients) override;

    private:
        double learning_rate_;
        double beta1_;
        double beta2_;
        double epsilon_;
        // TODO: Add moment estimates
    };

    /**
     * RMSprop Optimizer
     * TODO: Implement RMSprop with:
     * - Exponential moving average of squared gradients
     * - Adaptive learning rates
     */
    class RMSprop : public Optimizer {
    public:
        RMSprop(double learning_rate, double decay_rate = 0.9, double epsilon = 1e-8);

        void update(MatrixD &weights, const MatrixD &gradients) override;

    private:
        double learning_rate_;
        double decay_rate_;
        double epsilon_;
        // TODO: Add squared gradient average
    };
} // namespace dl::optimization
