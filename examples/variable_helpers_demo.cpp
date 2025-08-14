/**
 * @file variable_helpers_demo.cpp
 * @brief Demonstration of shorthand helper functions for creating Variables on heap
 * @author Kalenitid
 * 
 * This example shows how to use the new helper functions to create Variables
 * on the heap in a more convenient way, similar to std::make_shared.
 */

#include <iostream>
#include "../include/utils/autograd.hpp"
#include "../include/utils/tensor.hpp"

using namespace dl;

int main() {
    auto a = make_variable_ones<double>({1, 2}, true);
    auto b = make_variable(dl::TensorD::from_array({{1, 2}, {2, 3}}), true);

    a->zero_grad();
    b->zero_grad();

    // Perform some computation
    auto c = a->matmul(b);
    auto d = c->sum();
    auto e = dl::make_variable_scalar(1.0, true);
    auto two = dl::make_variable_scalar(2.0, true);
    auto f = d - e * two;
    auto mean = c->mean();

    mean->backward();

    std::cout << "   a = " << a->grad() << std::endl;
    std::cout << "   b = " << b->grad() << std::endl;
    std::cout << "   e = " << e->grad() << std::endl;
    return 0;
}
