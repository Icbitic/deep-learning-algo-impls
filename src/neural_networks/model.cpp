#include "neural_networks/model.hpp"
#include "neural_networks/cnn.hpp" // For Layer class definition

#include <algorithm>
#include <iostream>
#include <sstream>

namespace dl::neural_networks {
    Model::Model() {
        // Initialize an empty model
    }

    Model &Model::add(std::shared_ptr<Layer> layer) {
        layers_.push_back(layer);
        return *this;
    }

    MatrixD Model::forward(const MatrixD &input) {
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

    void Model::backward(const MatrixD &target) {
        if (layers_.empty()) {
            return;
        }

        // Compute output layer gradient (assuming MSE loss for now)
        // For cross-entropy loss or other loss functions, this would be different
        MatrixD gradient = last_output_ - target;

        // Backpropagate through all layers in reverse order
        for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
            gradient = (*it)->backward(gradient);
        }
    }

    void Model::update_parameters(double learning_rate) {
        // Update parameters in all layers
        for (auto &layer: layers_) {
            layer->update_parameters(learning_rate);
        }
    }

    void Model::train(const utils::Dataset<double> &dataset, int epochs, double learning_rate, size_t batch_size) {
        // Training loop
        for (int epoch = 0; epoch < epochs; ++epoch) {
            double total_loss = 0.0;
            size_t batch_count = 0;

            // Manually iterate through batches since Dataset doesn't implement range-based for loop
            for (size_t i = 0; i < dataset.size(); i += batch_size) {
                // Get batch of data
                auto batch = dataset.get_batch(i, batch_size);
                const auto &inputs = batch.first;
                const auto &targets = batch.second;

                // Forward pass
                MatrixD outputs = forward(inputs);

                // Compute loss (MSE)
                MatrixD error = outputs - targets;
                // Calculate squared error manually
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

    MatrixD Model::predict(const MatrixD &input) { return forward(input); }

    size_t Model::num_layers() const { return layers_.size(); }

    std::string Model::summary() const {
        std::stringstream ss;
        ss << "Model Summary:\n";
        ss << "-------------\n";
        ss << "Total layers: " << layers_.size() << "\n\n";

        for (size_t i = 0; i < layers_.size(); ++i) {
            ss << "Layer " << i + 1 << ": " << layers_[i]->name() << "\n";
        }

        return ss.str();
    }
} // namespace dl::neural_networks
