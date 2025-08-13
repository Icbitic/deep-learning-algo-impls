#include <iostream>
#include <memory>
#include <vector>

// Include the necessary headers
#include "utils/autograd.hpp"
#include "utils/tensor.hpp"

using TensorD = utils::Tensor<double>;
using VariableD = utils::Variable<double>;


int main() {
    try {
        auto x = utils::make_variable_scalar(1.0, true);
        auto y = utils::make_variable_scalar(1.0, true);
        auto z = *x + *y;
        z->backward();


    } catch (const std::exception &e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
