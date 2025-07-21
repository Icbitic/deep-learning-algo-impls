#pragma once

#include <memory>
#include <string>
#include <vector>
#include "utils/data_loader.hpp"
#include "utils/matrix.hpp"

using dl::utils::MatrixD;

namespace dl::neural_networks {
    // Forward declaration of Layer class
    class Layer;

    /**
     * @brief Generic Neural Network Model implementation
     *
     * This class implements a versatile neural network model that can be composed
     * of any combination of layers. It provides a PyTorch-like interface for
     * building, training, and using neural network models.
     */
    class Model {
    public:
        /**
         * @brief Default constructor for an empty model
         */
        Model();

        /**
         * @brief Add a layer to the model
         * @param layer Shared pointer to a Layer object
         * @return Reference to this model (for method chaining)
         */
        Model &add(std::shared_ptr<Layer> layer);

        /**
         * @brief Forward pass through the entire model
         * @param input Input data matrix
         * @return Output from the final layer
         */
        MatrixD forward(const MatrixD &input);

        /**
         * @brief Backward pass through the entire model
         * @param target Target output values for training
         */
        void backward(const MatrixD &target);

        /**
         * @brief Update all layer parameters using gradients
         * @param learning_rate Learning rate for parameter updates
         */
        void update_parameters(double learning_rate);

        /**
         * @brief Train the model using the provided dataset
         * @param dataset Training dataset containing input-output pairs
         * @param epochs Number of training epochs
         * @param learning_rate Learning rate for gradient descent
         * @param batch_size Size of mini-batches for training (default: 32)
         */
        void train(const utils::Dataset<double> &dataset, int epochs, double learning_rate, size_t batch_size = 32);

        /**
         * @brief Make predictions on new input data
         * @param input Input data matrix for prediction
         * @return Predicted output values
         */
        MatrixD predict(const MatrixD &input);

        /**
         * @brief Get the number of layers in the model
         * @return Number of layers
         */
        size_t num_layers() const;

        /**
         * @brief Get a string representation of the model architecture
         * @return String describing the model architecture
         */
        std::string summary() const;

    private:
        std::vector<std::shared_ptr<Layer>> layers_;
        MatrixD last_output_;
    };
} // namespace dl::neural_networks
