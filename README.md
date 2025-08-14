# Deep Learning & Machine Learning Algorithms in C++

A C++ library for implementing deep learning and machine learning algorithms from scratch.

[![CI](https://github.com/Icbitic/deep-learning-algo-impls/workflows/CI/badge.svg)](https://github.com/your-username/deep-learning-algo-impls/actions)
[![Documentation](https://img.shields.io/badge/docs-doxygen-blue.svg)](https://your-username.github.io/deep-learning-algo-impls/)

## Features

- **Neural Networks**: Feedforward, CNN, RNN, LSTM
- **Optimization**: SGD, Adam, RMSprop
- **ML Algorithms**: PCA, K-Means, SVM
- **Utilities**: Matrix operations, data loading, autograd
- **Modern C++23** with comprehensive tests

## Quick Start

```bash
# Clone and build
git clone <repository-url>
cd deep-learning-algo-impls
cmake -B build
cmake --build build

# Run tests
cd build && ctest
```

## Usage Example

```cpp
#include "utils/tensor.hpp"
#include "ml/pca.hpp"

// Matrix operations
Tensor<double> a(3, 3, 1.0);
Tensor<double> b = Tensor<double>::random(3, 3);
auto c = a * b;

// PCA example
PCA pca;
pca.fit(data);
auto reduced = pca.transform(data, 2);
```

## Requirements

- C++23 compatible compiler
- CMake 3.31+
- Google Test

## Documentation

[API Documentation](https://icbitic.github.io/deep-learning-algo-impls/)

## License

Apache License 2.0
