#pragma once

#include <vector>
#include <memory>
#include "utils/matrix.hpp"
#include "utils/data_loader.hpp"

using dl::utils::MatrixD;

namespace dl::neural_networks {
    /**
     * Feedforward Neural Network
     * TODO: Implement basic feedforward neural network with:
     * - Multiple layers
     * - Configurable activation functions
     * - Forward propagation
     * - Backpropagation
     * - Weight initialization strategies
     */
    class FeedforwardNetwork {
    public:
        FeedforwardNetwork(const std::vector<size_t> &layer_sizes);

        MatrixD forward(const MatrixD &input);

        void backward(const MatrixD &target);

        void train(const utils::Dataset<double> &dataset, int epochs, double learning_rate);

        MatrixD predict(const MatrixD &input);

    private:
        // TODO: Add member variables for layers, weights, biases
    };
} // namespace dl::neural_networks
