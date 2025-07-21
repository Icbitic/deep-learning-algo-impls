#include "neural_networks/cnn.hpp"

#include <algorithm>
#include <iostream>
#include <random>
#include <ranges>
#include "utils/data_loader.hpp"

namespace dl::neural_networks {
    // ConvolutionalNetwork implementation
    ConvolutionalNetwork::ConvolutionalNetwork() {
        // Initialize an empty network
    }

    ConvolutionalNetwork &ConvolutionalNetwork::add(std::shared_ptr<Layer> layer) {
        layers_.push_back(layer);
        return *this;
    }

    MatrixD ConvolutionalNetwork::forward(const MatrixD &input) {
        if (layers_.empty()) {
            return input; // No layers, return input as is
        }

        // Forward pass through all layers
        MatrixD output = input;
        for (auto &layer: layers_) {
            output = layer->forward(output);
        }

        last_output_ = output; // Cache for backward pass
        return output;
    }

    void ConvolutionalNetwork::backward(const MatrixD &target) {
        if (layers_.empty()) {
            return;
        }

        // Compute output layer gradient (assuming MSE loss for now)
        // For cross-entropy loss or other loss functions, this would be different
        MatrixD gradient = last_output_ - target;

        // Backpropagate through all layers in reverse order
        for (const auto &layer: std::ranges::reverse_view(layers_)) {
            gradient = layer->backward(gradient);
        }
    }

    void ConvolutionalNetwork::update_parameters(double learning_rate) {
        // Update parameters in all layers
        for (auto &layer: layers_) {
            layer->update_parameters(learning_rate);
        }
    }

    void ConvolutionalNetwork::train(const utils::Dataset<double> &dataset, int epochs, double learning_rate) {
        // Training loop
        for (int epoch = 0; epoch < epochs; ++epoch) {
            double total_loss = 0.0;
            size_t batch_count = 0;

            // Manually iterate through batches since Dataset doesn't implement range-based for loop
            for (size_t i = 0; i < dataset.size(); i += 32) { // Using a default batch size of 32
                // Get batch of data
                auto batch = dataset.get_batch(i, 32);
                const auto &inputs = batch.first;
                const auto &targets = batch.second;

                // Forward pass
                MatrixD outputs = forward(inputs);

                // Compute loss (MSE)
                MatrixD error = outputs - targets;
                // Calculate squared error manually since element_wise_multiply doesn't exist
                double batch_loss = 0.0;
                for (size_t r = 0; r < error.rows(); ++r) {
                    for (size_t c = 0; c < error.cols(); ++c) {
                        batch_loss += 0.5 * error(r, c) * error(r, c);
                    }
                }
                batch_loss /= inputs.rows();
                total_loss += batch_loss;

                // Backward pass
                backward(targets);

                // Update parameters
                update_parameters(learning_rate);

                ++batch_count;
            }

            // Print epoch statistics
            if (epoch % 10 == 0 || epoch == epochs - 1) {
                std::cout << "Epoch " << epoch + 1 << "/" << epochs << ", Loss: " << (total_loss / batch_count)
                          << std::endl;
            }
        }
    }

    MatrixD ConvolutionalNetwork::predict(const MatrixD &input) { return forward(input); }

    // ConvolutionLayer implementation
    ConvolutionLayer::ConvolutionLayer(size_t input_channels, size_t output_channels, size_t kernel_size, size_t stride,
                                       size_t padding, std::shared_ptr<activation::ActivationFunction> activation) :
        input_channels_(input_channels), output_channels_(output_channels), kernel_size_(kernel_size), stride_(stride),
        padding_(padding), activation_(activation) {

        // Initialize kernels with random values (Xavier/Glorot initialization)
        std::random_device rd;
        std::mt19937 gen(rd());
        double scale = std::sqrt(2.0 / (input_channels * kernel_size * kernel_size + output_channels));
        std::normal_distribution<double> dist(0.0, scale);

        // Create kernels for each input-output channel combination
        kernels_.resize(output_channels_ * input_channels_);
        for (auto &kernel: kernels_) {
            kernel = MatrixD(kernel_size_, kernel_size_);
            for (size_t i = 0; i < kernel_size_; ++i) {
                for (size_t j = 0; j < kernel_size_; ++j) {
                    kernel(i, j) = dist(gen);
                }
            }
        }

        // Initialize biases to zero
        biases_.resize(output_channels_, 0.0);

        // Initialize gradient storage
        gradient_kernels_.resize(kernels_.size());
        gradient_biases_.resize(biases_.size(), 0.0);
    }

    MatrixD ConvolutionLayer::forward(const MatrixD &input) {
        // Cache input for backward pass
        input_cache_ = input;

        // TODO: Implement actual convolution operation
        // This is a placeholder implementation
        // In a real implementation, we would:
        // 1. Apply padding to input if specified
        // 2. Slide kernel across input with specified stride
        // 3. Compute dot product at each position
        // 4. Add bias and apply activation function

        // For now, just return a dummy output
        size_t output_height = (input.rows() + 2 * padding_ - kernel_size_) / stride_ + 1;
        size_t output_width = (input.cols() + 2 * padding_ - kernel_size_) / stride_ + 1;

        MatrixD output(output_height, output_width);

        // Apply activation if provided
        if (activation_) {
            // Apply activation function element-wise
            // This would be implemented based on the activation function interface
        }

        return output;
    }

    MatrixD ConvolutionLayer::backward(const MatrixD &gradient) {
        // TODO: Implement actual convolution backward pass
        // This is a placeholder implementation
        // In a real implementation, we would:
        // 1. Compute gradients with respect to kernels
        // 2. Compute gradients with respect to biases
        // 3. Compute gradients with respect to input

        // For now, just return a dummy gradient
        return MatrixD(input_cache_.rows(), input_cache_.cols());
    }

    void ConvolutionLayer::update_parameters(double learning_rate) {
        // Update kernels
        for (size_t i = 0; i < kernels_.size(); ++i) {
            // Manually apply the learning rate to each element of the gradient
            MatrixD scaled_gradient(gradient_kernels_[i].rows(), gradient_kernels_[i].cols());
            for (size_t r = 0; r < gradient_kernels_[i].rows(); ++r) {
                for (size_t c = 0; c < gradient_kernels_[i].cols(); ++c) {
                    scaled_gradient(r, c) = gradient_kernels_[i](r, c) * learning_rate;
                }
            }
            kernels_[i] = kernels_[i] - scaled_gradient;
        }

        // Update biases
        for (size_t i = 0; i < biases_.size(); ++i) {
            biases_[i] -= gradient_biases_[i] * learning_rate;
        }

        // Reset gradients
        std::fill(gradient_biases_.begin(), gradient_biases_.end(), 0.0);
    }

    // PoolingLayer implementation
    PoolingLayer::PoolingLayer(size_t pool_size, size_t stride, PoolingType type) :
        pool_size_(pool_size), stride_(stride), type_(type) {
        // No parameters to initialize for pooling layer
    }

    MatrixD PoolingLayer::forward(const MatrixD &input) {
        // Cache input for backward pass
        input_cache_ = input;

        // TODO: Implement actual pooling operation
        // This is a placeholder implementation
        // In a real implementation, we would:
        // 1. Slide pooling window across input
        // 2. Apply pooling operation (max or average)
        // 3. For max pooling, store indices for backward pass

        // For now, just return a dummy output
        size_t output_height = (input.rows() - pool_size_) / stride_ + 1;
        size_t output_width = (input.cols() - pool_size_) / stride_ + 1;

        return MatrixD(output_height, output_width);
    }

    MatrixD PoolingLayer::backward(const MatrixD &gradient) {
        // TODO: Implement actual pooling backward pass
        // This is a placeholder implementation
        // In a real implementation, we would:
        // 1. For max pooling, route gradient only through max elements
        // 2. For average pooling, distribute gradient evenly

        // For now, just return a dummy gradient
        return MatrixD(input_cache_.rows(), input_cache_.cols());
    }

    // FlattenLayer implementation
    FlattenLayer::FlattenLayer() {
        // No parameters to initialize
    }

    MatrixD FlattenLayer::forward(const MatrixD &input) {
        // Cache input dimensions for backward pass
        input_rows_ = input.rows();
        input_cols_ = input.cols();

        // Flatten 2D matrix to 1D vector (as a matrix with 1 row)
        MatrixD output(1, input.rows() * input.cols());

        // Copy elements from input to output in row-major order
        for (size_t i = 0; i < input.rows(); ++i) {
            for (size_t j = 0; j < input.cols(); ++j) {
                output(0, i * input.cols() + j) = input(i, j);
            }
        }

        return output;
    }

    MatrixD FlattenLayer::backward(const MatrixD &gradient) {
        // Reshape 1D gradient back to 2D
        MatrixD output(input_rows_, input_cols_);

        // Copy elements from gradient to output in row-major order
        for (size_t i = 0; i < input_rows_; ++i) {
            for (size_t j = 0; j < input_cols_; ++j) {
                output(i, j) = gradient(0, i * input_cols_ + j);
            }
        }

        return output;
    }

    // FullyConnectedLayer implementation
    FullyConnectedLayer::FullyConnectedLayer(size_t input_size, size_t output_size,
                                             std::shared_ptr<activation::ActivationFunction> activation) :
        input_size_(input_size), output_size_(output_size), activation_(activation) {

        // Initialize weights with random values (Xavier/Glorot initialization)
        std::random_device rd;
        std::mt19937 gen(rd());
        double scale = std::sqrt(2.0 / (input_size + output_size));
        std::normal_distribution<double> dist(0.0, scale);

        weights_ = MatrixD(output_size_, input_size_);
        for (size_t i = 0; i < output_size_; ++i) {
            for (size_t j = 0; j < input_size_; ++j) {
                weights_(i, j) = dist(gen);
            }
        }

        // Initialize biases to zero
        biases_.resize(output_size_, 0.0);

        // Initialize gradient storage
        gradient_weights_ = MatrixD(output_size_, input_size_);
        gradient_biases_.resize(output_size_, 0.0);
    }

    MatrixD FullyConnectedLayer::forward(const MatrixD &input) {
        // Cache input for backward pass
        input_cache_ = input;

        // Compute output = weights * input + biases
        MatrixD output = weights_ * input.transpose();

        // Add biases
        for (size_t i = 0; i < output_size_; ++i) {
            for (size_t j = 0; j < output.cols(); ++j) {
                output(i, j) += biases_[i];
            }
        }

        // Apply activation if provided
        if (activation_) {
            // Apply activation function element-wise
            // This would be implemented based on the activation function interface
        }

        return output.transpose();
    }

    MatrixD FullyConnectedLayer::backward(const MatrixD &gradient) {
        // Compute gradient with respect to weights
        gradient_weights_ = gradient_weights_ + gradient.transpose() * input_cache_;

        // Compute gradient with respect to biases
        for (size_t i = 0; i < output_size_; ++i) {
            for (size_t j = 0; j < gradient.rows(); ++j) {
                gradient_biases_[i] += gradient(j, i);
            }
        }

        // Compute gradient with respect to input
        return gradient * weights_;
    }

    void FullyConnectedLayer::update_parameters(double learning_rate) {
        // Update weights
        // Manually apply the learning rate to each element of the gradient
        MatrixD scaled_gradient(gradient_weights_.rows(), gradient_weights_.cols());
        for (size_t r = 0; r < gradient_weights_.rows(); ++r) {
            for (size_t c = 0; c < gradient_weights_.cols(); ++c) {
                scaled_gradient(r, c) = gradient_weights_(r, c) * learning_rate;
            }
        }
        weights_ = weights_ - scaled_gradient;

        // Update biases
        for (size_t i = 0; i < biases_.size(); ++i) {
            biases_[i] -= gradient_biases_[i] * learning_rate;
        }

        // Reset gradients
        gradient_weights_ = MatrixD(output_size_, input_size_);
        std::fill(gradient_biases_.begin(), gradient_biases_.end(), 0.0);
    }
} // namespace dl::neural_networks
