# Deep Learning Algorithm Implementations

A comprehensive C++ library for implementing and learning deep learning algorithms from scratch, featuring modern C++ design patterns, extensive documentation, and automated CI/CD.

[![CI](https://github.com/your-username/deep-learning-algo-impls/workflows/CI/badge.svg)](https://github.com/your-username/deep-learning-algo-impls/actions)
[![Documentation](https://img.shields.io/badge/docs-doxygen-blue.svg)](https://your-username.github.io/deep-learning-algo-impls/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

## 🎯 Project Goals

This project provides a structured framework for implementing fundamental deep learning algorithms in C++. It's designed
for educational purposes and hands-on learning of:

- **Neural network architectures** (Feedforward, CNN, RNN, LSTM, GRU)
- **Optimization algorithms** (SGD, Adam, RMSprop)
- **Activation functions** (ReLU, Sigmoid, Tanh, Softmax, LeakyReLU)
- **Loss functions** (MSE, Cross-entropy, Hinge loss)
- **Mathematical utilities** and high-performance matrix operations
- **Data processing** with comprehensive loading and preprocessing utilities

## 🚀 Key Features

- 📚 **Comprehensive Documentation**: Full Doxygen documentation with examples and mathematical descriptions
- 🔧 **Modern C++**: Uses C++23 features and best practices
- 🧪 **Tested**: Comprehensive test suite with Google Test
- 🔄 **CI/CD**: Automated testing, static analysis, and documentation deployment
- 📊 **Performance**: Optimized matrix operations and memory management
- 🎓 **Educational**: Detailed comments and learning-focused design

## 📁 Project Structure

```
deep-learning-algo-impls/
├── include/                    # Header files
│   ├── neural_networks/        # Neural network architectures
│   │   ├── feedforward.hpp     # Feedforward neural networks
│   │   ├── cnn.hpp            # Convolutional neural networks
│   │   └── rnn.hpp            # Recurrent neural networks (RNN/LSTM/GRU)
│   ├── optimization/           # Optimization algorithms
│   │   └── optimizers.hpp     # SGD, Adam, RMSprop optimizers
│   ├── activation/             # Activation functions
│   │   └── functions.hpp      # ReLU, Sigmoid, Tanh, Softmax
│   ├── loss/                   # Loss functions
│   │   └── functions.hpp      # MSE, Cross-entropy, Hinge loss
│   └── utils/                  # Utility classes
│       ├── matrix.hpp         # Matrix operations
│       └── data_loader.hpp    # Data loading and preprocessing
├── src/                        # Implementation files
│   ├── neural_networks/        # Neural network implementations
│   └── optimization/           # Optimizer implementations
├── tests/                      # Unit tests
│   ├── test_feedforward.cpp    # Neural network tests
│   ├── test_matrix.cpp        # Matrix operation tests
│   └── test_optimizers.cpp    # Optimizer tests
├── .github/workflows/          # CI/CD pipelines
│   └── ci.yml                 # Automated testing workflow
├── CMakeLists.txt             # Build configuration
├── Doxyfile                   # Documentation configuration
└── main.cpp                   # Example usage
```

## 📖 Documentation

Full API documentation is automatically generated using Doxygen and deployed to GitHub Pages:

🔗 **[View Documentation](https://your-username.github.io/deep-learning-algo-impls/)**

The documentation includes:
- Complete API reference with examples
- Mathematical descriptions of algorithms
- Usage patterns and best practices
- Implementation guides and tutorials

## 🚀 Quick Start

### Matrix Operations

```cpp
#include "utils/matrix.hpp"
using namespace dl::utils;

// Create matrices
Matrix<double> a(3, 3, 1.0);  // 3x3 matrix filled with 1.0
Matrix<double> b = Matrix<double>::random(3, 3);  // Random 3x3 matrix

// Matrix operations
auto c = a * b;  // Matrix multiplication
auto d = a + b;  // Element-wise addition
auto e = a.transpose();  // Transpose
```

### Neural Network Training

```cpp
#include "neural_networks/feedforward.hpp"
#include "utils/data_loader.hpp"
using namespace dl;

// Define network architecture
std::vector<size_t> layers = {784, 128, 64, 10};  // MNIST-like network
neural_networks::FeedforwardNetwork network(layers);

// Load and preprocess data
auto [features, labels] = utils::CSVLoader::load_features_labels(
    "data.csv", {0, 1, 2, 3}, {4});
utils::Dataset<double> dataset(features, labels);

// Train the network
network.train(dataset, epochs=100, learning_rate=0.01);

// Make predictions
auto predictions = network.predict(test_features);
```

### Data Loading and Preprocessing

```cpp
#include "utils/data_loader.hpp"
using namespace dl::utils;

// Load CSV data
auto data = CSVLoader::load_csv("dataset.csv");

// Preprocess data
auto normalized = Preprocessor::normalize(data, 0.0, 1.0);
auto standardized = Preprocessor::standardize(data);

// Split dataset
auto [train, val, test] = Preprocessor::train_val_test_split(
    dataset, 0.7, 0.15);

// Create data loader for batch processing
DataLoader<double> loader(train, batch_size=32, shuffle=true);
while (loader.has_next()) {
    auto [batch_features, batch_labels] = loader.next_batch();
    // Process batch...
}
```

## 🛠️ Prerequisites

- **C++23** compatible compiler (GCC 11+, Clang 14+, or MSVC 2022+)
- **CMake** 3.31 or higher
- **Google Test** for unit testing
- **Git** for version control

### Installing Dependencies

#### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install -y cmake ninja-build libgtest-dev
sudo apt-get install -y gcc-11 g++-11  # or clang-14
```

#### macOS

```bash
brew install cmake ninja googletest
```

#### Windows (vcpkg)

```bash
vcpkg install gtest
```

## 🚀 Building the Project

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd deep-learning-algo-impls
   ```

2. **Configure and build**
   ```bash
   cmake -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build
   ```

3. **Run tests**
   ```bash
   cd build
   ctest --output-on-failure
   ```

4. **Run the main executable**
   ```bash
   ./build/deep_learning_algo_impls
   ```

## 📚 Implementation Guide

This project provides header files with comprehensive TODO comments and example structures. Each algorithm should be
implemented following these guidelines:

### 1. Neural Networks

- **Feedforward Networks**: Implement basic multilayer perceptrons with configurable architectures
- **CNNs**: Add convolution, pooling, and feature extraction layers
- **RNNs**: Implement sequence processing with LSTM and GRU variants

### 2. Optimization

- **SGD**: Basic gradient descent with momentum support
- **Adam**: Adaptive learning rates with bias correction
- **RMSprop**: Root mean square propagation

### 3. Mathematical Utilities

- **Matrix Class**: Efficient matrix operations for linear algebra
- **Activation Functions**: Differentiable activation functions
- **Loss Functions**: Various loss functions for different tasks

### 4. Data Processing

- **Data Loaders**: CSV and image data loading utilities
- **Preprocessing**: Normalization, standardization, and augmentation

## 🧪 Testing Strategy

The project includes comprehensive unit tests for:

- Matrix operations and mathematical correctness
- Neural network forward/backward propagation
- Optimizer convergence and update rules
- Activation and loss function derivatives

### Running Specific Tests

```bash
# Run all tests
ctest

# Run specific test suite
./build/run_tests --gtest_filter="MatrixTest.*"

# Run with verbose output
./build/run_tests --gtest_filter="*" --gtest_output="verbose"
```

## 🔄 Continuous Integration

The project includes GitHub Actions workflows that automatically:

- Build and test on multiple platforms (Ubuntu, macOS)
- Test with different compilers (GCC, Clang)
- Run static analysis and code formatting checks
- Generate documentation (when implemented)
- Perform memory leak detection

## 📖 Learning Path

Recommended implementation order for learning:

1. **Start with Matrix utilities** - Foundation for all operations
2. **Implement activation functions** - Simple mathematical functions
3. **Build feedforward networks** - Core neural network concepts
4. **Add optimization algorithms** - Learning and convergence
5. **Implement loss functions** - Training objectives
6. **Extend to CNNs** - Spatial data processing
7. **Add RNNs/LSTMs** - Sequential data processing

## 🤝 Contributing

This is a learning-focused project. Feel free to:

- Implement the TODO items in the headers
- Add comprehensive tests for your implementations
- Improve documentation and examples
- Optimize performance and memory usage
- Add new algorithms and techniques

## 📄 License

Apache License 2.0

## 🔗 Resources

- [Deep Learning Book](http://www.deeplearningbook.org/) by Ian Goodfellow

---

**Happy Learning! 🚀**