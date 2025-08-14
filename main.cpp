#include <iostream>
#include <memory>
#include <vector>

// Include the necessary headers
#include "include/utils/autograd.hpp"
#include "include/utils/tensor.hpp"

using TensorD = dl::Tensor<double>;
using VariableD = dl::Variable<double>;


int main() {
    try {
        auto x = dl::make_variable_scalar(1.0, true);
        auto y = dl::make_variable_scalar(1.0, true);
        auto z = *x + *y;
        z->backward();


    } catch (const std::exception &e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
