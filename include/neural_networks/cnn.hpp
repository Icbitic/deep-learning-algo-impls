#pragma once

#include <memory>
#include <vector>
#include "utils/data_loader.hpp"
#include "utils/matrix.hpp"

using dl::utils::MatrixD;

namespace dl::neural_networks {
    enum class PoolingType { MAX, AVERAGE };

    /**
     * Convolutional Neural Network
     * TODO: Implement CNN with:
     * - Convolutional layers
     * - Pooling layers
     * - Activation functions
     * - Batch normalization
     * - Dropout
     */
    class ConvolutionalNetwork {
    public:
        ConvolutionalNetwork();

        MatrixD forward(const MatrixD &input);

        void backward(const MatrixD &target);

        void train(const utils::Dataset<double> &dataset, int epochs, double learning_rate);

        MatrixD predict(const MatrixD &input);

    private:
        // TODO: Add member variables for layers, kernels, feature maps
    };

    /**
     * Convolution Layer
     * TODO: Implement individual convolution layer
     */
    class ConvolutionLayer {
    public:
        ConvolutionLayer(size_t input_channels, size_t output_channels, size_t kernel_size, size_t stride,
                         size_t padding);

        MatrixD forward(const MatrixD &input);

        void backward(const MatrixD &gradient);

    private:
        size_t input_channels_;
        size_t output_channels_;
        size_t kernel_size_;
        size_t stride_;
        size_t padding_;
        // TODO: Add kernels, biases, activation function
    };

    /**
     * Pooling Layer
     * TODO: Implement pooling operations
     */
    class PoolingLayer {
    public:
        PoolingLayer(size_t pool_size, size_t stride, PoolingType type);

        MatrixD forward(const MatrixD &input);

        void backward(const MatrixD &gradient);

    private:
        size_t pool_size_;
        size_t stride_;
        PoolingType type_;
        // TODO: Add pooling parameters
    };
} // namespace dl::neural_networks
