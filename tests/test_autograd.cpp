#include <gtest/gtest.h>
#include "utils/autograd.hpp"
#include "utils/tensor.hpp"
#include <memory>
#include <vector>
#include <cmath>

using namespace dl;

class AutogradTest : public ::testing::Test {
protected:
    void SetUp() override {
        tensor1 = Tensor<float>({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
        tensor2 = Tensor<float>({5.0f, 6.0f, 7.0f, 8.0f}, {2, 2});
        tensor3 = Tensor<float>({0.1f, 0.2f, 0.3f, 0.4f}, {2, 2});

        // Create variables
        var1 = make_variable(tensor1, true);
        var2 = make_variable(tensor2, true);
        var3 = make_variable(tensor3, false);
    }

    Tensor<float> tensor1, tensor2, tensor3;
    std::shared_ptr<Variable<float> > var1, var2, var3;
};

// Test Variable construction and basic properties
TEST_F(AutogradTest, VariableConstruction) {
    // Test construction with requires_grad=true
    Variable<float> v1(tensor1, true);
    EXPECT_TRUE(v1.requires_grad());
    EXPECT_EQ(v1.data().shape(), tensor1.shape());

    // Test construction with requires_grad=false
    Variable<float> v2(tensor2, false);
    EXPECT_FALSE(v2.requires_grad());

    // Test gradient function is null initially
    EXPECT_EQ(v1.get_grad_fn(), nullptr);
}

// Test Variable data access and modification
TEST_F(AutogradTest, VariableDataAccess) {
    Variable<float> v(tensor1, true);

    // Test data access
    EXPECT_EQ(v.data().shape(), tensor1.shape());
    EXPECT_EQ(v.size(), tensor1.size());

    // Test shape access
    EXPECT_EQ(v.rows(), 2u);
    EXPECT_EQ(v.cols(), 2u);

    // Test requires_grad modification
    v.set_requires_grad(false);
    EXPECT_FALSE(v.requires_grad());

    v.set_requires_grad(true);
    EXPECT_TRUE(v.requires_grad());
}

// Test Variable cloning and detaching
TEST_F(AutogradTest, VariableCloneDetach) {
    Variable<float> v(tensor1, true);

    // Test clone
    auto cloned = v.clone(true);
    EXPECT_TRUE(cloned->requires_grad());
    EXPECT_EQ(cloned->data().shape(), v.data().shape());

    auto cloned_no_grad = v.clone(false);
    EXPECT_FALSE(cloned_no_grad->requires_grad());

    // Test detach
    auto detached = v.detach();
    EXPECT_FALSE(detached->requires_grad());
    EXPECT_EQ(detached->get_grad_fn(), nullptr);
}

// Test gradient zeroing
TEST_F(AutogradTest, GradientZeroing) {
    // Initially gradient should be empty
    EXPECT_EQ(static_cast<int>(var1->grad().size()), 0);

    // After backward pass, gradient should exist
    auto result = dl::operator*(var1, var1);
    auto sum_result = result->sum();
    sum_result->backward();

    EXPECT_GT(static_cast<int>(var1->grad().size()), 0);

    // Zero gradient
    var1->zero_grad();
    EXPECT_EQ(static_cast<int>(var1->grad().size()), 0);
}

// Test basic arithmetic operations
TEST_F(AutogradTest, ArithmeticOperations) {
    auto v1 = make_variable(tensor1, true);
    auto v2 = make_variable(tensor2, true);

    // Test addition
    auto add_result = v1 + v2;
    EXPECT_TRUE(add_result->requires_grad());
    EXPECT_NE(add_result->get_grad_fn(), nullptr);

    // Test subtraction
    auto sub_result = v1 - v2;
    EXPECT_TRUE(sub_result->requires_grad());
    EXPECT_NE(sub_result->get_grad_fn(), nullptr);

    // Test multiplication
    auto mul_result = dl::operator*(v1, v2);
    EXPECT_TRUE(mul_result->requires_grad());
    EXPECT_NE(mul_result->get_grad_fn(), nullptr);

    // Test division
    auto div_result = v1 / v2;
    EXPECT_TRUE(div_result->requires_grad());
    EXPECT_NE(div_result->get_grad_fn(), nullptr);
}

// Test scalar operations
TEST_F(AutogradTest, ScalarOperations) {
    auto v = make_variable(tensor1, true);
    float scalar = 2.5f;

    // Test scalar addition
    auto add_result = v + scalar;
    EXPECT_TRUE(add_result->requires_grad());

    // Test scalar multiplication
    auto mul_result = dl::operator*(v, scalar);
    EXPECT_TRUE(mul_result->requires_grad());

    // Test scalar subtraction
    auto sub_result = v - scalar;
    EXPECT_TRUE(sub_result->requires_grad());

    // Test scalar division
    auto div_result = v / scalar;
    EXPECT_TRUE(div_result->requires_grad());
}

// Test matrix operations
TEST_F(AutogradTest, MatrixOperations) {
    auto v1 = make_variable(tensor1, true);
    auto v2 = make_variable(tensor2, true);

    // Test dot product
    auto dot_result = v1->dot(v2);
    EXPECT_TRUE(dot_result->requires_grad());
    EXPECT_NE(dot_result->get_grad_fn(), nullptr);

    // Test matrix multiplication
    auto matmul_result = v1->matmul(v2);
    EXPECT_TRUE(matmul_result->requires_grad());
    EXPECT_NE(matmul_result->get_grad_fn(), nullptr);

    // Test transpose
    auto transpose_result = v1->transpose();
    EXPECT_TRUE(transpose_result->requires_grad());
    EXPECT_NE(transpose_result->get_grad_fn(), nullptr);
}

// Test activation functions
TEST_F(AutogradTest, ActivationFunctions) {
    auto v = make_variable(tensor1, true);

    // Test sigmoid
    auto sigmoid_result = v->sigmoid();
    EXPECT_TRUE(sigmoid_result->requires_grad());
    EXPECT_NE(sigmoid_result->get_grad_fn(), nullptr);

    // Test tanh
    auto tanh_result = v->tanh();
    EXPECT_TRUE(tanh_result->requires_grad());
    EXPECT_NE(tanh_result->get_grad_fn(), nullptr);

    // Test ReLU
    auto relu_result = v->relu();
    EXPECT_TRUE(relu_result->requires_grad());
    EXPECT_NE(relu_result->get_grad_fn(), nullptr);

    // Test exp
    auto exp_result = v->exp();
    EXPECT_TRUE(exp_result->requires_grad());
    EXPECT_NE(exp_result->get_grad_fn(), nullptr);

    // Test log (use positive values)
    Tensor<float> pos_tensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    auto pos_v = make_variable(pos_tensor, true);

    auto log_result = pos_v->log();
    EXPECT_TRUE(log_result->requires_grad());
    EXPECT_NE(log_result->get_grad_fn(), nullptr);
}

// Test reduction operations
TEST_F(AutogradTest, ReductionOperations) {
    auto v = make_variable(tensor1, true);

    // Test sum
    auto sum_result = v->sum();
    EXPECT_TRUE(sum_result->requires_grad());
    EXPECT_NE(sum_result->get_grad_fn(), nullptr);

    // Test mean
    auto mean_result = v->mean();
    EXPECT_TRUE(mean_result->requires_grad());
    EXPECT_NE(mean_result->get_grad_fn(), nullptr);
}

// Test backward pass with simple operations
TEST_F(AutogradTest, SimpleBackwardPass) {
    // Simple computation: z = (v1 + v2) * v1
    auto temp = var1 + var2;
    auto result = dl::operator*(temp, var1);
    auto loss = result->sum();

    // Backward pass
    loss->backward();

    // Check that gradients exist
    EXPECT_GT(static_cast<int>(var1->grad().size()), 0);
    EXPECT_GT(static_cast<int>(var2->grad().size()), 0);

    // Check gradient shapes
    EXPECT_EQ(var1->grad().shape(), var1->data().shape());
    EXPECT_EQ(var2->grad().shape(), var2->data().shape());
}

// Test backward pass with activation functions
TEST_F(AutogradTest, ActivationBackwardPass) {
    // Test sigmoid backward pass
    std::cout << "Debug: var1 address = " << var1.get() << std::endl;
    auto sigmoid_result = var1->sigmoid();
    auto loss = sigmoid_result->sum();

    // Backward pass
    loss->backward();

    // Check that gradients exist
    std::cout << "Debug: var1->grad().size() = " << var1->grad().size() << std::endl;
    EXPECT_GT(static_cast<int>(var1->grad().size()), 0);

    // Test ReLU backward pass
    var1->zero_grad(); // Clear previous gradients
    auto relu_result = var1->relu();
    auto loss2 = relu_result->sum();

    // Backward pass
    loss2->backward();

    // Check that gradients exist
    EXPECT_GT(static_cast<int>(var1->grad().size()), 0);
}

// Test gradient accumulation
TEST_F(AutogradTest, GradientAccumulation) {
    // First backward pass
    auto result1 = dl::operator*(var1, var1);
    auto sum1 = result1->sum();
    sum1->backward();

    Tensor<float> first_grad = var1->grad();

    // Second backward pass (should accumulate)
    auto result2 = var1 + var1;
    auto sum2 = result2->sum();
    sum2->backward();

    // Gradient should be accumulated
    EXPECT_NE(var1->grad().data(), first_grad.data());
}

// Test no gradient computation when requires_grad=false
TEST_F(AutogradTest, NoGradientComputation) {
    auto v1 = make_variable(tensor1, false); // requires_grad=false
    auto v2 = make_variable(tensor2, true);

    auto result = v1 + v2;
    auto sum_result = result->sum();
    sum_result->backward();

    // v1 should not have gradients
    EXPECT_EQ(static_cast<int>(v1->grad().size()), 0);

    // v2 should have gradients
    EXPECT_GT(static_cast<int>(v2->grad().size()), 0);
}

// Test Function classes directly
TEST_F(AutogradTest, FunctionClasses) {
    std::vector<Variable<float> > inputs = {*var1, *var2};

    // Test AddFunction
    AddFunction<float> add_fn;
    Tensor<float> add_result = add_fn.forward(inputs);
    EXPECT_EQ(add_fn.num_inputs(), 2);

    auto add_grads = add_fn.backward(Tensor<float>::ones({2, 2}));
    EXPECT_EQ(add_grads.size(), 2);

    // Test MulFunction
    MulFunction<float> mul_fn;
    Tensor<float> mul_result = mul_fn.forward(inputs);
    EXPECT_EQ(mul_fn.num_inputs(), 2);

    auto mul_grads = mul_fn.backward(Tensor<float>::ones({2, 2}));
    EXPECT_EQ(mul_grads.size(), 2);

    // Test SigmoidFunction
    SigmoidFunction<float> sigmoid_fn;
    std::vector<Variable<float> > single_input = {*var1};
    Tensor<float> sigmoid_result = sigmoid_fn.forward(single_input);
    EXPECT_EQ(sigmoid_fn.num_inputs(), 1);

    auto sigmoid_grads = sigmoid_fn.backward(Tensor<float>::ones({2, 2}));
    EXPECT_EQ(sigmoid_grads.size(), 1);
}

// Test error handling
TEST_F(AutogradTest, ErrorHandling) {
    // Test backward on non-gradient variable
    Variable<float> v(tensor1, false);

    // This should not throw but should do nothing
    EXPECT_NO_THROW(v.backward());
    EXPECT_EQ(v.grad().size(), 0);
}

// Test complex computational graph
TEST_F(AutogradTest, ComplexComputationalGraph) {
    auto x = make_variable(tensor1, true);
    auto w1 = make_variable(tensor2, true);
    auto w2 = make_variable(tensor3, true);

    // Complex computational graph: h1 = sigmoid(x * w1), h2 = tanh(h1 * w2), loss = sum(h2)
    auto h1 = dl::operator*(x, w1)->sigmoid();
    auto h2 = dl::operator*(h1, w2)->tanh();
    auto loss = h2->sum();

    // Backward pass
    loss->backward();

    // All variables should have gradients
    EXPECT_GT(static_cast<int>(x->grad().size()), 0);
    EXPECT_GT(static_cast<int>(w1->grad().size()), 0);
    EXPECT_GT(static_cast<int>(w2->grad().size()), 0);

    // Check gradient shapes
    EXPECT_EQ(x->grad().shape(), x->data().shape());
    EXPECT_EQ(w1->grad().shape(), w1->data().shape());
    EXPECT_EQ(w2->grad().shape(), w2->data().shape());
}

// Test numerical gradient checking
TEST_F(AutogradTest, NumericalGradientCheck) {
    const float eps = 1e-4f;
    const float tolerance = 1e-2f; // Increased tolerance for numerical precision

    // Simple function: f(x) = sum(x^2)
    std::vector<float> data = {2.0f, 3.0f};
    Tensor<float> x_tensor({2.0f, 3.0f}, {2, 1});
    auto x = make_variable(x_tensor, true);

    // Analytical gradient
    auto x_squared = dl::operator*(x, x);
    auto y_temp = x_squared->sum();
    y_temp->backward();
    Tensor<float> analytical_grad = x->grad();

    // Numerical gradient
    std::vector<float> numerical_grad_data;
    for (size_t i = 0; i < data.size(); ++i) {
        // f(x + eps)
        std::vector<float> x_plus = data;
        x_plus[i] += eps;
        Tensor<float> x_plus_tensor({x_plus[0], x_plus[1]}, {2, 1});
        auto x_plus_var = make_variable(x_plus_tensor, false);
        auto x_plus_squared = dl::operator*(x_plus_var, x_plus_var);
        auto y_plus = x_plus_squared->sum();

        // f(x - eps)
        std::vector<float> x_minus = data;
        x_minus[i] -= eps;
        Tensor<float> x_minus_tensor({x_minus[0], x_minus[1]}, {2, 1});
        auto x_minus_var = make_variable(x_minus_tensor, false);
        auto x_minus_squared = dl::operator*(x_minus_var, x_minus_var);
        auto y_minus = x_minus_squared->sum();

        // Numerical gradient: (f(x+eps) - f(x-eps)) / (2*eps)
        float num_grad = (y_plus->data().data()[0] - y_minus->data().data()[0]) / (2 * eps);
        numerical_grad_data.push_back(num_grad);
    }

    // Compare analytical and numerical gradients
    for (size_t i = 0; i < data.size(); ++i) {
        float analytical = analytical_grad(i, 0); // Use proper 2D indexing instead of flat indexing
        float numerical = numerical_grad_data[i];
        EXPECT_NEAR(analytical, numerical, tolerance)
            << "Gradient mismatch at index " << i
            << ": analytical=" << analytical
            << ", numerical=" << numerical;
    }
}

// Test memory management and shared pointers
TEST_F(AutogradTest, MemoryManagement) {
    auto v1 = make_variable(tensor1, true);
    auto v2 = make_variable(tensor2, true);

    // Create computation that should maintain references
    auto result = v1 + v2;

    // The result should have a gradient function
    EXPECT_NE(result->get_grad_fn(), nullptr);

    // The gradient function should maintain references to inputs
    const auto &inputs = result->get_grad_fn()->get_inputs();
    EXPECT_EQ(inputs.size(), 2);

    // Test that we can still access the original variables through the graph
    EXPECT_NE(inputs[0], nullptr);
    EXPECT_NE(inputs[1], nullptr);
}

// Test class specifically for backward pass unit tests
class BackwardPassTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test tensors with different dimensions for comprehensive testing
        tensor_scalar = Tensor<float>({5.0f}, {1});
        tensor_1d = Tensor<float>({1.0f, 2.0f, 3.0f}, {3});
        tensor_2x2 = Tensor<float>({1.5f, 2.5f, 3.5f, 4.5f}, {2, 2});
        tensor_3x1 = Tensor<float>({2.0f, 3.0f, 4.0f}, {3, 1});
        tensor_2x3 = Tensor<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
        tensor_3d = Tensor<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, {2, 2, 2});

        // Create variables with gradient tracking
        var_scalar = make_variable(tensor_scalar, true);
        var_1d = make_variable(tensor_1d, true);
        var_2x2 = make_variable(tensor_2x2, true);
        var_3x1 = make_variable(tensor_3x1, true);
        var_2x3 = make_variable(tensor_2x3, true);
        var_3d = make_variable(tensor_3d, true);
    }

    // Helper function for numerical gradient checking
    float numerical_gradient(std::function<float(float)> func, float x, float h = 1e-5f) {
        return (func(x + h) - func(x - h)) / (2.0f * h);
    }

    // Helper function to check gradient correctness with tolerance
    void check_gradient(const Tensor<float> &analytical, const Tensor<float> &numerical, float tolerance = 1e-4f) {
        ASSERT_EQ(analytical.shape(), numerical.shape());
        for (size_t i = 0; i < analytical.size(); ++i) {
            EXPECT_NEAR(analytical.data()[i], numerical.data()[i], tolerance)
                << "Gradient mismatch at index " << i;
        }
    }

    Tensor<float> tensor_scalar, tensor_1d, tensor_2x2, tensor_3x1, tensor_2x3, tensor_3d;
    std::shared_ptr<Variable<float> > var_scalar, var_1d, var_2x2, var_3x1, var_2x3, var_3d;
};

// Test AddFunction backward pass
TEST_F(BackwardPassTest, AddFunctionBackward) {
    auto var1 = make_variable(Tensor<float>({1.0f, 2.0f}, {2, 1}), true);
    auto var2 = make_variable(Tensor<float>({3.0f, 4.0f}, {2, 1}), true);

    auto result = var1 + var2;
    auto loss = result->sum();

    loss->backward();

    // For addition: d/dx(x + y) = 1, d/dy(x + y) = 1
    Tensor<float> expected_grad({1.0f, 1.0f}, {2, 1});
    check_gradient(var1->grad(), expected_grad);
    check_gradient(var2->grad(), expected_grad);
}

// Test SubFunction backward pass
TEST_F(BackwardPassTest, SubFunctionBackward) {
    auto var1 = make_variable(Tensor<float>({1.0f, 2.0f}, {2, 1}), true);
    auto var2 = make_variable(Tensor<float>({3.0f, 4.0f}, {2, 1}), true);

    auto result = var1 - var2;
    auto loss = result->sum();

    loss->backward();

    // For subtraction: d/dx(x - y) = 1, d/dy(x - y) = -1
    Tensor<float> expected_grad1({1.0f, 1.0f}, {2, 1});
    Tensor<float> expected_grad2({-1.0f, -1.0f}, {2, 1});
    check_gradient(var1->grad(), expected_grad1);
    check_gradient(var2->grad(), expected_grad2);
}

// Test MulFunction backward pass
TEST_F(BackwardPassTest, MulFunctionBackward) {
    auto var1 = make_variable(Tensor<float>({2.0f, 3.0f}, {2, 1}), true);
    auto var2 = make_variable(Tensor<float>({4.0f, 5.0f}, {2, 1}), true);

    auto result = var1 * var2;
    auto loss = result->sum();

    loss->backward();

    // For multiplication: d/dx(x * y) = y, d/dy(x * y) = x
    check_gradient(var1->grad(), var2->data());
    check_gradient(var2->grad(), var1->data());
}

// Test DivFunction backward pass
TEST_F(BackwardPassTest, DivFunctionBackward) {
    auto var1 = make_variable(Tensor<float>({6.0f, 8.0f}, {2, 1}), true);
    auto var2 = make_variable(Tensor<float>({2.0f, 4.0f}, {2, 1}), true);

    auto result = var1 / var2;
    auto loss = result->sum();

    loss->backward();

    // For division: d/dx(x / y) = 1/y, d/dy(x / y) = -x/(y^2)
    Tensor<float> expected_grad1({1.0f / 2.0f, 1.0f / 4.0f}, {2, 1});
    Tensor<float> expected_grad2({-6.0f / (2.0f * 2.0f), -8.0f / (4.0f * 4.0f)}, {2, 1});
    check_gradient(var1->grad(), expected_grad1);
    check_gradient(var2->grad(), expected_grad2);
}

// Test ScalarAddFunction backward pass
TEST_F(BackwardPassTest, ScalarAddFunctionBackward) {
    auto var = make_variable(Tensor<float>({1.0f, 2.0f, 3.0f}, {3, 1}), true);
    float scalar = 5.0f;

    auto result = var + scalar;
    auto loss = result->sum();

    loss->backward();

    // For scalar addition: d/dx(x + c) = 1
    Tensor<float> expected_grad({1.0f, 1.0f, 1.0f}, {3, 1});
    check_gradient(var->grad(), expected_grad);
}

// Test ScalarMulFunction backward pass
TEST_F(BackwardPassTest, ScalarMulFunctionBackward) {
    auto var = make_variable(Tensor<float>({1.0f, 2.0f, 3.0f}, {3, 1}), true);
    float scalar = 3.0f;

    auto result = var * scalar;
    auto loss = result->sum();

    loss->backward();

    // For scalar multiplication: d/dx(x * c) = c
    Tensor<float> expected_grad({3.0f, 3.0f, 3.0f}, {3, 1});
    check_gradient(var->grad(), expected_grad);
}

// Test ScalarSubFunction backward pass
TEST_F(BackwardPassTest, ScalarSubFunctionBackward) {
    auto var = make_variable(Tensor<float>({5.0f, 6.0f, 7.0f}, {3, 1}), true);
    float scalar = 2.0f;

    auto result = var - scalar;
    auto loss = result->sum();

    loss->backward();

    // For scalar subtraction: d/dx(x - c) = 1
    Tensor<float> expected_grad({1.0f, 1.0f, 1.0f}, {3, 1});
    check_gradient(var->grad(), expected_grad);
}

// Test ScalarDivFunction backward pass
TEST_F(BackwardPassTest, ScalarDivFunctionBackward) {
    auto var = make_variable(Tensor<float>({6.0f, 8.0f, 10.0f}, {3, 1}), true);
    float scalar = 2.0f;

    auto result = var / scalar;
    auto loss = result->sum();

    loss->backward();

    // For scalar division: d/dx(x / c) = 1/c
    Tensor<float> expected_grad({1.0f / 2.0f, 1.0f / 2.0f, 1.0f / 2.0f}, {3, 1});
    check_gradient(var->grad(), expected_grad);
}

// Test ReverseScalarSubFunction backward pass
TEST_F(BackwardPassTest, ReverseScalarSubFunctionBackward) {
    auto var = make_variable(Tensor<float>({1.0f, 2.0f, 3.0f}, {3, 1}), true);
    float scalar = 10.0f;

    auto result = scalar - var;
    auto loss = result->sum();

    loss->backward();

    // For reverse scalar subtraction: d/dx(c - x) = -1
    Tensor<float> expected_grad({-1.0f, -1.0f, -1.0f}, {3, 1});
    check_gradient(var->grad(), expected_grad);
}

// Test ReverseScalarDivFunction backward pass
TEST_F(BackwardPassTest, ReverseScalarDivFunctionBackward) {
    auto var = make_variable(Tensor<float>({2.0f, 4.0f, 5.0f}, {3, 1}), true);
    float scalar = 20.0f;

    auto result = scalar / var;
    auto loss = result->sum();

    loss->backward();

    // For reverse scalar division: d/dx(c / x) = -c/(x^2)
    Tensor<float> expected_grad({-20.0f / (2.0f * 2.0f), -20.0f / (4.0f * 4.0f), -20.0f / (5.0f * 5.0f)}, {3, 1});
    check_gradient(var->grad(), expected_grad);
}

// Test SigmoidFunction backward pass
TEST_F(BackwardPassTest, SigmoidFunctionBackward) {
    auto var = make_variable(Tensor<float>({0.0f, 1.0f, -1.0f}, {3, 1}), true);

    auto result = var->sigmoid();
    auto loss = result->sum();

    loss->backward();

    // For sigmoid: d/dx(sigmoid(x)) = sigmoid(x) * (1 - sigmoid(x))
    // Calculate expected gradients manually
    auto sigmoid_vals = result->data();
    Tensor<float> expected_grad({
                                    sigmoid_vals.data()[0] * (1.0f - sigmoid_vals.data()[0]),
                                    sigmoid_vals.data()[1] * (1.0f - sigmoid_vals.data()[1]),
                                    sigmoid_vals.data()[2] * (1.0f - sigmoid_vals.data()[2])
                                }, {3, 1});

    check_gradient(var->grad(), expected_grad);
}

// Test ReLUFunction backward pass
TEST_F(BackwardPassTest, ReLUFunctionBackward) {
    auto var = make_variable(Tensor<float>({-1.0f, 0.0f, 2.0f, -3.0f}, {4, 1}), true);

    auto result = var->relu();
    auto loss = result->sum();

    loss->backward();

    // For ReLU: d/dx(relu(x)) = 1 if x > 0, else 0
    Tensor<float> expected_grad({0.0f, 0.0f, 1.0f, 0.0f}, {4, 1});
    check_gradient(var->grad(), expected_grad);
}

// Test TanhFunction backward pass
TEST_F(BackwardPassTest, TanhFunctionBackward) {
    auto var = make_variable(Tensor<float>({0.0f, 1.0f, -0.5f}, {3, 1}), true);

    auto result = var->tanh();
    auto loss = result->sum();

    loss->backward();

    // For tanh: d/dx(tanh(x)) = 1 - tanh^2(x)
    auto tanh_vals = result->data();
    Tensor<float> expected_grad({
                                    1.0f - tanh_vals.data()[0] * tanh_vals.data()[0],
                                    1.0f - tanh_vals.data()[1] * tanh_vals.data()[1],
                                    1.0f - tanh_vals.data()[2] * tanh_vals.data()[2]
                                }, {3, 1});

    check_gradient(var->grad(), expected_grad);
}

// Test ExpFunction backward pass
TEST_F(BackwardPassTest, ExpFunctionBackward) {
    auto var = make_variable(Tensor<float>({0.0f, 1.0f, 2.0f}, {3, 1}), true);

    auto result = var->exp();
    auto loss = result->sum();

    loss->backward();

    // For exp: d/dx(exp(x)) = exp(x)
    check_gradient(var->grad(), result->data());
}

// Test LogFunction backward pass
TEST_F(BackwardPassTest, LogFunctionBackward) {
    auto var = make_variable(Tensor<float>({1.0f, 2.0f, 3.0f}, {3, 1}), true);

    auto result = var->log();
    auto loss = result->sum();

    loss->backward();

    // For log: d/dx(log(x)) = 1/x
    Tensor<float> expected_grad({1.0f / 1.0f, 1.0f / 2.0f, 1.0f / 3.0f}, {3, 1});
    check_gradient(var->grad(), expected_grad);
}

// Test SumFunction backward pass
TEST_F(BackwardPassTest, SumFunctionBackward) {
    auto var = make_variable(Tensor<float>({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}), true);

    auto result = var->sum();
    result->backward();

    // For sum: gradient is ones tensor with same shape as input
    Tensor<float> expected_grad({1.0f, 1.0f, 1.0f, 1.0f}, {2, 2});
    check_gradient(var->grad(), expected_grad);
}

// Test MeanFunction backward pass
TEST_F(BackwardPassTest, MeanFunctionBackward) {
    auto var = make_variable(Tensor<float>({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}), true);

    auto result = var->mean();
    result->backward();

    // For mean: gradient is 1/n where n is the number of elements
    float grad_val = 1.0f / 4.0f;
    Tensor<float> expected_grad({grad_val, grad_val, grad_val, grad_val}, {2, 2});
    check_gradient(var->grad(), expected_grad);
}

// Test MatMulFunction backward pass
TEST_F(BackwardPassTest, MatMulFunctionBackward) {
    auto var1 = make_variable(Tensor<float>({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}), true);
    auto var2 = make_variable(Tensor<float>({5.0f, 6.0f, 7.0f, 8.0f}, {2, 2}), true);

    auto result = var1->matmul(var2);
    auto loss = result->sum();

    loss->backward();

    // For matrix multiplication: d/dA(A @ B) = grad_output @ B^T
    //                           d/dB(A @ B) = A^T @ grad_output
    // Since grad_output is ones matrix after sum(), we can verify shapes
    EXPECT_EQ(var1->grad().shape(), var1->data().shape());
    EXPECT_EQ(var2->grad().shape(), var2->data().shape());

    // Verify gradients are non-zero
    EXPECT_GT(var1->grad().data()[0], 0.0f);
    EXPECT_GT(var2->grad().data()[0], 0.0f);
}

// Test TransposeFunction backward pass
TEST_F(BackwardPassTest, TransposeFunctionBackward) {
    auto var = make_variable(Tensor<float>({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}), true);

    auto result = var->transpose();
    auto loss = result->sum();

    loss->backward();

    // For transpose: gradient should be transposed back to original shape
    Tensor<float> expected_grad({1.0f, 1.0f, 1.0f, 1.0f}, {2, 2});
    check_gradient(var->grad(), expected_grad);
}

// Test ViewFunction backward pass
TEST_F(BackwardPassTest, ViewFunctionBackward) {
    auto var = make_variable(Tensor<float>({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}), true);

    auto result = var->view({4, 1});
    auto loss = result->sum();

    loss->backward();

    // For view/reshape: gradient should be reshaped back to original shape
    Tensor<float> expected_grad({1.0f, 1.0f, 1.0f, 1.0f}, {2, 2});
    check_gradient(var->grad(), expected_grad);
}

// ============ COMPREHENSIVE DIMENSIONAL TESTS ============

// Test scalar operations with different dimensions
TEST_F(BackwardPassTest, ScalarOperationsMultiDimensional) {
    // Test scalar multiplication with 1D tensor (element-wise)
    auto scalar_var = make_variable(Tensor<float>({2.0f}, {1}), true);
    auto tensor_1d_var = make_variable(Tensor<float>({1.0f, 2.0f, 3.0f}, {3}), true);

    // Create scalar tensors for each element to multiply
    auto scalar1 = make_variable(Tensor<float>({2.0f}, {1}), true);
    auto scalar2 = make_variable(Tensor<float>({2.0f}, {1}), true);
    auto scalar3 = make_variable(Tensor<float>({2.0f}, {1}), true);

    auto elem1 = make_variable(Tensor<float>({1.0f}, {1}), true);
    auto elem2 = make_variable(Tensor<float>({2.0f}, {1}), true);
    auto elem3 = make_variable(Tensor<float>({3.0f}, {1}), true);

    auto result1 = scalar1 * elem1;
    auto result2 = scalar2 * elem2;
    auto result3 = scalar3 * elem3;

    auto total_loss = result1 + result2 + result3;
    auto loss = total_loss->sum();
    loss->backward();

    // Each scalar should have gradient equal to its corresponding element
    EXPECT_NEAR(scalar1->grad().data()[0], 1.0f, 1e-4f);
    EXPECT_NEAR(scalar2->grad().data()[0], 2.0f, 1e-4f);
    EXPECT_NEAR(scalar3->grad().data()[0], 3.0f, 1e-4f);
}

// Test 1D tensor operations
TEST_F(BackwardPassTest, OneDimensionalTensorOperations) {
    auto var1_1d = make_variable(Tensor<float>({2.0f, 3.0f, 4.0f}, {3}), true);
    auto var2_1d = make_variable(Tensor<float>({1.0f, 2.0f, 3.0f}, {3}), true);

    // Element-wise multiplication
    auto result = var1_1d * var2_1d;
    auto loss = result->sum();
    loss->backward();

    // Check gradients: d/dx(x*y) = y, d/dy(x*y) = x
    Tensor<float> expected_grad1({1.0f, 2.0f, 3.0f}, {3});
    Tensor<float> expected_grad2({2.0f, 3.0f, 4.0f}, {3});
    check_gradient(var1_1d->grad(), expected_grad1);
    check_gradient(var2_1d->grad(), expected_grad2);
}

// Test 2D tensor operations
TEST_F(BackwardPassTest, TwoDimensionalTensorOperations) {
    auto var1_2d = make_variable(Tensor<float>({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}), true);
    auto var2_2d = make_variable(Tensor<float>({2.0f, 1.0f, 4.0f, 3.0f}, {2, 2}), true);

    // Element-wise addition and then sum
    auto result = var1_2d + var2_2d;
    auto loss = result->sum();
    loss->backward();

    // For addition, gradients should be all ones
    Tensor<float> expected_grad({1.0f, 1.0f, 1.0f, 1.0f}, {2, 2});
    check_gradient(var1_2d->grad(), expected_grad);
    check_gradient(var2_2d->grad(), expected_grad);
}

// Test 3D tensor operations
TEST_F(BackwardPassTest, ThreeDimensionalTensorOperations) {
    auto var1_3d = make_variable(tensor_3d, true);
    auto var2_3d = make_variable(Tensor<float>({2.0f, 1.0f, 3.0f, 2.0f, 1.0f, 3.0f, 2.0f, 1.0f}, {2, 2, 2}), true);

    // Element-wise subtraction
    auto result = var1_3d - var2_3d;
    auto loss = result->sum();
    loss->backward();

    // For subtraction: d/dx(x-y) = 1, d/dy(x-y) = -1
    Tensor<float> expected_grad1({1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}, {2, 2, 2});
    Tensor<float> expected_grad2({-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f}, {2, 2, 2});
    check_gradient(var1_3d->grad(), expected_grad1);
    check_gradient(var2_3d->grad(), expected_grad2);
}

// ============ ADVANCED MATHEMATICAL OPERATIONS ============

// Test exponential function backward pass
TEST_F(BackwardPassTest, ExpFunctionAdvanced) {
    // Test with different tensor dimensions
    auto var_small = make_variable(Tensor<float>({0.5f, 1.0f, 1.5f}, {3}), true);

    auto result = var_small->exp();
    auto loss = result->sum();
    loss->backward();

    // d/dx(exp(x)) = exp(x)
    Tensor<float> expected_grad({std::exp(0.5f), std::exp(1.0f), std::exp(1.5f)}, {3});
    check_gradient(var_small->grad(), expected_grad);
}

// Test logarithm function backward pass
TEST_F(BackwardPassTest, LogFunctionAdvanced) {
    // Test with positive values to avoid log(0) or log(negative)
    auto var_pos = make_variable(Tensor<float>({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}), true);

    auto result = var_pos->log();
    auto loss = result->sum();
    loss->backward();

    // d/dx(log(x)) = 1/x
    Tensor<float> expected_grad({1.0f / 1.0f, 1.0f / 2.0f, 1.0f / 3.0f, 1.0f / 4.0f}, {2, 2});
    check_gradient(var_pos->grad(), expected_grad);
}

// Test power operations (if available)
TEST_F(BackwardPassTest, PowerOperations) {
    auto var_base = make_variable(Tensor<float>({2.0f, 3.0f}, {2}), true);

    // Test x^2 using multiplication
    auto result = var_base * var_base;
    auto loss = result->sum();
    loss->backward();

    // d/dx(x^2) = 2x
    Tensor<float> expected_grad({4.0f, 6.0f}, {2});
    check_gradient(var_base->grad(), expected_grad);
}

// Test complex chain rule with multiple operations
TEST_F(BackwardPassTest, ComplexChainRule) {
    auto var_x = make_variable(Tensor<float>({1.0f, 2.0f}, {2}), true);

    // Compute f(x) = exp(x^2 + 1)
    auto x_squared = var_x * var_x;
    auto x_squared_plus_one = x_squared + make_variable(Tensor<float>({1.0f, 1.0f}, {2}), false);
    auto result = x_squared_plus_one->exp();
    auto loss = result->sum();

    loss->backward();

    // d/dx[exp(x^2 + 1)] = exp(x^2 + 1) * 2x
    float grad1 = std::exp(1.0f * 1.0f + 1.0f) * 2.0f * 1.0f;
    float grad2 = std::exp(2.0f * 2.0f + 1.0f) * 2.0f * 2.0f;
    Tensor<float> expected_grad({grad1, grad2}, {2});
    check_gradient(var_x->grad(), expected_grad, 1e-3f); // Slightly larger tolerance for complex operations
}

// Test element-wise operations with same dimensions
TEST_F(BackwardPassTest, ElementWiseOperations) {
    // Test element-wise operations with same shaped tensors
    auto var1 = make_variable(Tensor<float>({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}), true);
    auto var2 = make_variable(Tensor<float>({2.0f, 1.0f, 4.0f, 3.0f}, {2, 2}), true);

    // Element-wise multiplication
    auto result = var1 * var2;
    auto loss = result->sum();
    loss->backward();

    // For multiplication: d/dx(x*y) = y, d/dy(x*y) = x
    Tensor<float> expected_grad1({2.0f, 1.0f, 4.0f, 3.0f}, {2, 2});
    Tensor<float> expected_grad2({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    check_gradient(var1->grad(), expected_grad1);
    check_gradient(var2->grad(), expected_grad2);
}

// Test reduction operations with different dimensions
TEST_F(BackwardPassTest, ReductionOperationsAdvanced) {
    // Test sum reduction on 3D tensor
    auto var_3d_test = make_variable(tensor_3d, true);
    auto sum_result = var_3d_test->sum();
    sum_result->backward();

    // Sum gradient should be all ones
    Tensor<float> expected_grad({1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}, {2, 2, 2});
    check_gradient(var_3d_test->grad(), expected_grad);

    // Reset and test mean
    var_3d_test->zero_grad();
    auto mean_result = var_3d_test->mean();
    mean_result->backward();

    // Mean gradient should be 1/n for each element
    float mean_grad = 1.0f / 8.0f; // 8 elements in 2x2x2 tensor
    Tensor<float> expected_mean_grad({
                                         mean_grad, mean_grad, mean_grad, mean_grad,
                                         mean_grad, mean_grad, mean_grad, mean_grad
                                     }, {2, 2, 2});
    check_gradient(var_3d_test->grad(), expected_mean_grad);
}

// Test mixed operations with simpler computation
TEST_F(BackwardPassTest, MixedOperations) {
    // Test combination of different operations with simpler chain
    auto var_a = make_variable(Tensor<float>({1.0f, 2.0f}, {2}), true);
    auto var_b = make_variable(Tensor<float>({3.0f, 4.0f}, {2}), true);

    // f(a,b) = (a * b) + exp(a)
    auto mult = var_a * var_b;
    auto exp_a = var_a->exp();
    auto result = mult + exp_a;
    auto loss = result->sum();

    loss->backward();

    // Verify gradients are computed and non-zero
    // For var_a: gradient = b + exp(a)
    // For var_b: gradient = a
    EXPECT_GT(std::abs(var_a->grad().data()[0]), 0.0f);
    EXPECT_GT(std::abs(var_a->grad().data()[1]), 0.0f);
    EXPECT_NEAR(var_b->grad().data()[0], 1.0f, 1e-4f); // Should be a[0] = 1.0
    EXPECT_NEAR(var_b->grad().data()[1], 2.0f, 1e-4f); // Should be a[1] = 2.0
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
