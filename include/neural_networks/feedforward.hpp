#pragma once

#include <memory>
#include <vector>
#include "utils/data_loader.hpp"
#include "utils/matrix.hpp"

/**
 * @file feedforward.hpp
 * @brief Feedforward neural network implementation
 * @author Deep Learning Algorithm Implementations
 * @version 1.0.0
 */

using dl::utils::MatrixD;

namespace dl::neural_networks {
    /**
     * @brief Feedforward Neural Network implementation
     * 
     * A feedforward neural network (also known as a multilayer perceptron) is
     * a fundamental type of artificial neural network where information flows
     * in one direction from input to output without cycles.
     * 
     * Key features:
     * - Multiple fully connected layers
     * - Configurable activation functions per layer
     * - Forward propagation for inference
     * - Backpropagation for training
     * - Various weight initialization strategies
     * 
     * @example
     * ```cpp
     * // Create a network with 784 inputs, 128 hidden, 10 outputs
     * std::vector<size_t> layers = {784, 128, 64, 10};
     * FeedforwardNetwork network(layers);
     * 
     * // Train the network
     * network.train(dataset, epochs=100, learning_rate=0.01);
     * 
     * // Make predictions
     * auto predictions = network.predict(test_input);
     * ```
     * 
     * @note This implementation supports arbitrary network architectures
     */
    class FeedforwardNetwork {
    public:
        /**
         * @brief Constructor to create a feedforward network
         * @param layer_sizes Vector specifying the number of neurons in each layer
         *                   (including input and output layers)
         * 
         * @example
         * ```cpp
         * // Create network: 784 inputs -> 128 hidden -> 64 hidden -> 10 outputs
         * std::vector<size_t> architecture = {784, 128, 64, 10};
         * FeedforwardNetwork network(architecture);
         * ```
         */
        FeedforwardNetwork(const std::vector<size_t> &layer_sizes);

        /**
         * @brief Perform forward propagation through the network
         * @param input Input data matrix (samples x features)
         * @return Output activations from the final layer
         * 
         * @note This method computes the network's response to input data
         *       by propagating activations through all layers
         */
        MatrixD forward(const MatrixD &input);

        /**
         * @brief Perform backward propagation to compute gradients
         * @param target Target output values for training
         * 
         * @note This method computes gradients using the chain rule
         *       and updates internal gradient storage for weight updates
         */
        void backward(const MatrixD &target);

        /**
         * @brief Train the network using the provided dataset
         * @param dataset Training dataset containing input-output pairs
         * @param epochs Number of training epochs
         * @param learning_rate Learning rate for gradient descent
         * 
         * @note Training uses mini-batch gradient descent with backpropagation
         */
        void train(const utils::Dataset<double> &dataset, int epochs, double learning_rate);

        /**
         * @brief Make predictions on new input data
         * @param input Input data matrix for prediction
         * @return Predicted output values
         * 
         * @note This is equivalent to forward() but emphasizes inference usage
         */
        MatrixD predict(const MatrixD &input);

    private:
        /**
         * @todo Add member variables for network architecture
         * @brief Network layers, weights, biases, and activation functions
         * 
         * Planned member variables:
         * - std::vector<MatrixD> weights_: Weight matrices for each layer
         * - std::vector<MatrixD> biases_: Bias vectors for each layer
         * - std::vector<std::unique_ptr<ActivationFunction>> activations_: Activation functions
         * - std::vector<size_t> layer_sizes_: Architecture specification
         */
    };
} // namespace dl::neural_networks
