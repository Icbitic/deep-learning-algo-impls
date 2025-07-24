#include <iostream>
#include <memory>
#include <vector>

// Include the necessary headers
#include "activation/functions.hpp"
#include "neural_networks/cnn.hpp"
#include "neural_networks/feedforward.hpp"
#include "neural_networks/model.hpp"
#include "utils/data_loader.hpp"
#include "utils/matrix.hpp"
#include "utils/pca.hpp"

int main() {
    std::cout << "=== Deep Learning Algorithm Implementations ===" << std::endl;
    std::cout << "\nThis project provides a framework for implementing deep learning algorithms from scratch."
              << std::endl;
    std::cout << "\nProject Structure:" << std::endl;
    std::cout << "├── Neural Networks: Feedforward, CNN, RNN/LSTM/GRU" << std::endl;
    std::cout << "├── Optimization: SGD, Adam, RMSprop" << std::endl;
    std::cout << "├── Activation Functions: ReLU, Sigmoid, Tanh, Softmax" << std::endl;
    std::cout << "├── Loss Functions: MSE, Cross-entropy, Hinge loss" << std::endl;
    std::cout << "└── Utilities: Matrix operations, Data loading" << std::endl;

    // Example of creating a neural network with the new generic Model class
    std::cout << "\n=== Neural Network Example with Generic Model ===" << std::endl;

    // Create activation functions
    auto relu = std::make_shared<dl::activation::ReLU>();
    auto sigmoid = std::make_shared<dl::activation::Sigmoid>();

    // Create a generic model
    dl::neural_networks::Model model;

    // Example 1: Create a CNN architecture
    std::cout << "\nExample 1: CNN Architecture" << std::endl;
    dl::neural_networks::Model cnn_model;

    // Add layers to create a CNN (similar to PyTorch)
    cnn_model.add(std::make_shared<dl::neural_networks::ConvolutionLayer>(1, 16, 3, 1, 1, relu))
            .add(std::make_shared<dl::neural_networks::PoolingLayer>(2, 2))
            .add(std::make_shared<dl::neural_networks::ConvolutionLayer>(16, 32, 3, 1, 1, relu))
            .add(std::make_shared<dl::neural_networks::PoolingLayer>(2, 2))
            .add(std::make_shared<dl::neural_networks::FlattenLayer>())
            .add(std::make_shared<dl::neural_networks::FullyConnectedLayer>(32 * 7 * 7, 128, relu))
            .add(std::make_shared<dl::neural_networks::FullyConnectedLayer>(128, 10, sigmoid));

    std::cout << cnn_model.summary() << std::endl;

    // Example 2: Create a simple feedforward network
    std::cout << "\nExample 2: Feedforward Network Architecture" << std::endl;
    dl::neural_networks::Model ff_model;

    // Add layers to create a feedforward network
    ff_model.add(std::make_shared<dl::neural_networks::FullyConnectedLayer>(784, 128, relu))
            .add(std::make_shared<dl::neural_networks::FullyConnectedLayer>(128, 64, relu))
            .add(std::make_shared<dl::neural_networks::FullyConnectedLayer>(64, 10, sigmoid));

    std::cout << ff_model.summary() << std::endl;

    // Example 3: Principal Component Analysis (PCA)
    std::cout << "\n=== Example 3: Principal Component Analysis (PCA) ===" << std::endl;

    // Create a sample dataset
    dl::utils::MatrixD data({{2.5, 2.4},
                             {0.5, 0.7},
                             {2.2, 2.9},
                             {1.9, 2.2},
                             {3.1, 3.0},
                             {2.3, 2.7},
                             {2.0, 1.6},
                             {1.0, 1.1},
                             {1.5, 1.6},
                             {1.1, 0.9}});

    // Create a PCA object
    dl::utils::PCAD pca;

    // Fit the PCA model to the data
    pca.fit(data);

    // Get the explained variance ratio
    auto variance_ratio = pca.explained_variance_ratio();
    std::cout << "Explained variance ratio: [" << variance_ratio[0] << ", " << variance_ratio[1] << "]" << std::endl;

    // Transform the data to the principal component space (reduce to 1 dimension)
    auto transformed = pca.transform(data, 1);
    std::cout << "\nOriginal data (first 3 samples):" << std::endl;
    for (size_t i = 0; i < 3; ++i) {
        std::cout << "[" << data(i, 0) << ", " << data(i, 1) << "]" << std::endl;
    }

    std::cout << "\nTransformed data (first 3 samples, reduced to 1 dimension):" << std::endl;
    for (size_t i = 0; i < 3; ++i) {
        std::cout << "[" << transformed(i, 0) << "]" << std::endl;
    }

    // Get the principal components
    auto components = pca.components();
    std::cout << "\nPrincipal components:" << std::endl;
    std::cout << "[" << components(0, 0) << ", " << components(1, 0) << "]" << std::endl;
    std::cout << "[" << components(0, 1) << ", " << components(1, 1) << "]" << std::endl;

    std::cout << "\n=== Example Usage (Generic Model) ===" << std::endl;
    std::cout << "/*" << std::endl;
    std::cout << "// Create a generic model" << std::endl;
    std::cout << "dl::neural_networks::Model model;" << std::endl;
    std::cout << "\n// Create activation functions" << std::endl;
    std::cout << "auto relu = std::make_shared<dl::activation::ReLU>();" << std::endl;
    std::cout << "auto sigmoid = std::make_shared<dl::activation::Sigmoid>();" << std::endl;
    std::cout << "\n// Example 1: Create a CNN architecture" << std::endl;
    std::cout << "model.add(std::make_shared<dl::neural_networks::ConvolutionLayer>(1, 16, 3, 1, 1, relu))"
              << std::endl;
    std::cout << "     .add(std::make_shared<dl::neural_networks::PoolingLayer>(2, 2))" << std::endl;
    std::cout << "     .add(std::make_shared<dl::neural_networks::ConvolutionLayer>(16, 32, 3, 1, 1, relu))"
              << std::endl;
    std::cout << "     .add(std::make_shared<dl::neural_networks::PoolingLayer>(2, 2))" << std::endl;
    std::cout << "     .add(std::make_shared<dl::neural_networks::FlattenLayer>())" << std::endl;
    std::cout << "     .add(std::make_shared<dl::neural_networks::FullyConnectedLayer>(32 * 7 * 7, 128, relu))"
              << std::endl;
    std::cout << "     .add(std::make_shared<dl::neural_networks::FullyConnectedLayer>(128, 10, sigmoid));"
              << std::endl;
    std::cout << "\n// Example 2: Create a simple feedforward network" << std::endl;
    std::cout << "dl::neural_networks::Model ff_model;" << std::endl;
    std::cout << "ff_model.add(std::make_shared<dl::neural_networks::FullyConnectedLayer>(784, 128, relu))"
              << std::endl;
    std::cout << "        .add(std::make_shared<dl::neural_networks::FullyConnectedLayer>(128, 64, relu))" << std::endl;
    std::cout << "        .add(std::make_shared<dl::neural_networks::FullyConnectedLayer>(64, 10, sigmoid));"
              << std::endl;
    std::cout << "\n// Train the model" << std::endl;
    std::cout << "model.train(mnist_dataset, 10, 0.01);" << std::endl;
    std::cout << "\n// Make predictions" << std::endl;
    std::cout << "MatrixD test_image = load_image(\"test_digit.png\");" << std::endl;
    std::cout << "MatrixD prediction = model.predict(test_image);" << std::endl;
    std::cout << "\n// Get model summary" << std::endl;
    std::cout << "std::cout << model.summary() << std::endl;" << std::endl;
    std::cout << "\n// Example 3: Using PCA for dimensionality reduction" << std::endl;
    std::cout << "// Create a PCA object" << std::endl;
    std::cout << "dl::utils::PCAD pca;" << std::endl;
    std::cout << "\n// Fit the PCA model to your data" << std::endl;
    std::cout << "MatrixD data = load_dataset(\"high_dimensional_data.csv\");" << std::endl;
    std::cout << "pca.fit(data);" << std::endl;
    std::cout << "\n// Get the explained variance ratio" << std::endl;
    std::cout << "auto variance_ratio = pca.explained_variance_ratio();" << std::endl;
    std::cout << "for (size_t i = 0; i < variance_ratio.size(); ++i) {" << std::endl;
    std::cout << R"(    std::cout << "Component " << i << ": " << variance_ratio[i] << std::endl;)" << std::endl;
    std::cout << "}" << std::endl;
    std::cout << "\n// Reduce dimensions (e.g., to 2D for visualization)" << std::endl;
    std::cout << "MatrixD reduced_data = pca.transform(data, 2);" << std::endl;
    std::cout << "*/" << std::endl;

    std::cout << "\nHappy coding! 🚀" << std::endl;

    return 0;
}
