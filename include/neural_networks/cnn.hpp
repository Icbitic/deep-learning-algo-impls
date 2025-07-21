#pragma once

#include <memory>
#include <string>
#include <vector>
#include "activation/functions.hpp"
#include "utils/data_loader.hpp"
#include "utils/matrix.hpp"

using dl::utils::MatrixD;

namespace dl::neural_networks {
    enum class PoolingType { MAX, AVERAGE };

    /**
     * @brief Base Layer class for neural network layers
     *
     * This abstract class defines the interface that all neural network layers
     * must implement. It provides the foundation for building neural networks
     * with a consistent layer-based architecture similar to PyTorch.
     */
    class Layer {
    public:
        /**
         * @brief Virtual destructor for proper cleanup
         */
        virtual ~Layer() = default;

        /**
         * @brief Forward pass through the layer
         * @param input Input data matrix
         * @return Output activations from the layer
         */
        virtual MatrixD forward(const MatrixD &input) = 0;

        /**
         * @brief Backward pass through the layer
         * @param gradient Gradient from the next layer
         * @return Gradient with respect to the input
         */
        virtual MatrixD backward(const MatrixD &gradient) = 0;

        /**
         * @brief Update layer parameters using gradients
         * @param learning_rate Learning rate for parameter updates
         */
        virtual void update_parameters(double learning_rate) = 0;

        /**
         * @brief Get the name of the layer
         * @return Layer name as string
         */
        virtual std::string name() const = 0;
    };

    /**
     * @brief Convolutional Layer implementation
     */
    class ConvolutionLayer : public Layer {
    public:
        /**
         * @brief Constructor for convolutional layer
         * @param input_channels Number of input channels
         * @param output_channels Number of output channels (filters)
         * @param kernel_size Size of the convolution kernel (square)
         * @param stride Stride for the convolution operation
         * @param padding Padding size for the input
         * @param activation Activation function to use (optional)
         */
        ConvolutionLayer(size_t input_channels, size_t output_channels, size_t kernel_size, size_t stride = 1,
                         size_t padding = 0, std::shared_ptr<activation::ActivationFunction> activation = nullptr);

        /**
         * @brief Forward pass through the convolutional layer
         * @param input Input data matrix
         * @return Output feature maps
         */
        MatrixD forward(const MatrixD &input) override;

        /**
         * @brief Backward pass through the convolutional layer
         * @param gradient Gradient from the next layer
         * @return Gradient with respect to the input
         */
        MatrixD backward(const MatrixD &gradient) override;

        /**
         * @brief Update layer parameters using gradients
         * @param learning_rate Learning rate for parameter updates
         */
        void update_parameters(double learning_rate) override;

        /**
         * @brief Get the name of the layer
         * @return Layer name as string
         */
        std::string name() const override { return "ConvolutionLayer"; }

    private:
        size_t input_channels_;
        size_t output_channels_;
        size_t kernel_size_;
        size_t stride_;
        size_t padding_;
        std::vector<MatrixD> kernels_;
        std::vector<double> biases_;
        std::shared_ptr<activation::ActivationFunction> activation_;

        // Cache for backward pass
        MatrixD input_cache_;
        std::vector<MatrixD> gradient_kernels_;
        std::vector<double> gradient_biases_;
    };

    /**
     * @brief Pooling Layer implementation
     */
    class PoolingLayer : public Layer {
    public:
        /**
         * @brief Constructor for pooling layer
         * @param pool_size Size of the pooling window (square)
         * @param stride Stride for the pooling operation
         * @param type Type of pooling (MAX or AVERAGE)
         */
        PoolingLayer(size_t pool_size, size_t stride = 1, PoolingType type = PoolingType::MAX);

        /**
         * @brief Forward pass through the pooling layer
         * @param input Input data matrix
         * @return Pooled output
         */
        MatrixD forward(const MatrixD &input) override;

        /**
         * @brief Backward pass through the pooling layer
         * @param gradient Gradient from the next layer
         * @return Gradient with respect to the input
         */
        MatrixD backward(const MatrixD &gradient) override;

        /**
         * @brief Update layer parameters (no parameters in pooling layer)
         * @param learning_rate Learning rate (unused)
         */
        void update_parameters(double learning_rate) override {};

        /**
         * @brief Get the name of the layer
         * @return Layer name as string
         */
        std::string name() const override { return "PoolingLayer"; }

    private:
        size_t pool_size_;
        size_t stride_;
        PoolingType type_;

        // Cache for backward pass
        MatrixD input_cache_;
        MatrixD max_indices_; // For max pooling backward pass
    };

    /**
     * @brief Flatten Layer for converting 2D feature maps to 1D vectors
     */
    class FlattenLayer : public Layer {
    public:
        /**
         * @brief Constructor for flatten layer
         */
        FlattenLayer();

        /**
         * @brief Forward pass through the flatten layer
         * @param input Input data matrix (feature maps)
         * @return Flattened 1D vector
         */
        MatrixD forward(const MatrixD &input) override;

        /**
         * @brief Backward pass through the flatten layer
         * @param gradient Gradient from the next layer
         * @return Gradient with respect to the input
         */
        MatrixD backward(const MatrixD &gradient) override;

        /**
         * @brief Update layer parameters (no parameters in flatten layer)
         * @param learning_rate Learning rate (unused)
         */
        void update_parameters(double learning_rate) override {};

        /**
         * @brief Get the name of the layer
         * @return Layer name as string
         */
        std::string name() const override { return "FlattenLayer"; }

    private:
        // Cache for backward pass
        size_t input_rows_;
        size_t input_cols_;
    };

    /**
     * @brief Fully Connected (Dense) Layer implementation
     */
    class FullyConnectedLayer : public Layer {
    public:
        /**
         * @brief Constructor for fully connected layer
         * @param input_size Number of input neurons
         * @param output_size Number of output neurons
         * @param activation Activation function to use (optional)
         */
        FullyConnectedLayer(size_t input_size, size_t output_size,
                            std::shared_ptr<activation::ActivationFunction> activation = nullptr);

        /**
         * @brief Forward pass through the fully connected layer
         * @param input Input data matrix
         * @return Output activations
         */
        MatrixD forward(const MatrixD &input) override;

        /**
         * @brief Backward pass through the fully connected layer
         * @param gradient Gradient from the next layer
         * @return Gradient with respect to the input
         */
        MatrixD backward(const MatrixD &gradient) override;

        /**
         * @brief Update layer parameters using gradients
         * @param learning_rate Learning rate for parameter updates
         */
        void update_parameters(double learning_rate) override;

        /**
         * @brief Get the name of the layer
         * @return Layer name as string
         */
        std::string name() const override { return "FullyConnectedLayer"; }

    private:
        size_t input_size_;
        size_t output_size_;
        MatrixD weights_;
        std::vector<double> biases_;
        std::shared_ptr<activation::ActivationFunction> activation_;

        // Cache for backward pass
        MatrixD input_cache_;
        MatrixD gradient_weights_;
        std::vector<double> gradient_biases_;
    };

    /**
     * @brief Convolutional Neural Network implementation
     *
     * This class implements a convolutional neural network using a layer-based
     * architecture similar to PyTorch. Layers can be added sequentially to build
     * the network architecture.
     */
    class ConvolutionalNetwork {
    public:
        /**
         * @brief Default constructor for an empty network
         */
        ConvolutionalNetwork();

        /**
         * @brief Add a layer to the network
         * @param layer Shared pointer to a Layer object
         * @return Reference to this network (for method chaining)
         */
        ConvolutionalNetwork &add(std::shared_ptr<Layer> layer);

        /**
         * @brief Forward pass through the entire network
         * @param input Input data matrix
         * @return Output from the final layer
         */
        MatrixD forward(const MatrixD &input);

        /**
         * @brief Backward pass through the entire network
         * @param target Target output values for training
         */
        void backward(const MatrixD &target);

        /**
         * @brief Update all layer parameters using gradients
         * @param learning_rate Learning rate for parameter updates
         */
        void update_parameters(double learning_rate);

        /**
         * @brief Train the network using the provided dataset
         * @param dataset Training dataset containing input-output pairs
         * @param epochs Number of training epochs
         * @param learning_rate Learning rate for gradient descent
         */
        void train(const utils::Dataset<double> &dataset, int epochs, double learning_rate);

        /**
         * @brief Make predictions on new input data
         * @param input Input data matrix for prediction
         * @return Predicted output values
         */
        MatrixD predict(const MatrixD &input);

    private:
        std::vector<std::shared_ptr<Layer>> layers_;
        MatrixD last_output_;
    };
} // namespace dl::neural_networks
