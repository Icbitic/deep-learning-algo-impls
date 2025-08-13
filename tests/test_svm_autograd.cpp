#include <chrono>
#include <cmath>
#include <gtest/gtest.h>
#include <vector>
#include "ml/svm.hpp"
#include "utils/autograd.hpp"
#include "utils/tensor.hpp"

using namespace utils;
using namespace ml;

class SVMAutogradTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple linearly separable 2D dataset
        // Class 1: points around (1, 1)
        // Class -1: points around (-1, -1)
        X_linear_ = Tensor<double>({
            {1.0, 1.0},
            {1.2, 0.8},
            {0.8, 1.2},
            {1.1, 1.1}, // Class 1
            {-1.0, -1.0},
            {-1.2, -0.8},
            {-0.8, -1.2},
            {-1.1, -1.1} // Class -1
        });
        y_linear_ = {1, 1, 1, 1, -1, -1, -1, -1};

        // Create a non-linearly separable dataset (XOR-like)
        X_nonlinear_ = Tensor<double>({
            {1.0, 1.0},
            {-1.0, -1.0}, // Class 1
            {1.0, -1.0},
            {-1.0, 1.0} // Class -1
        });
        y_nonlinear_ = {1, 1, -1, -1};

        // Create a simple 1D dataset
        X_1d_ = Tensor<double>::from_array({2.0, 3.0, 4.0, -2.0, -3.0, -4.0});
        y_1d_ = {1, 1, 1, -1, -1, -1};
    }

    Tensor<double> X_linear_, X_nonlinear_, X_1d_;
    std::vector<int> y_linear_, y_nonlinear_, y_1d_;
};

TEST_F(SVMAutogradTest, AutogradBasicTest) {
    // Test basic autograd functionality
    Tensor<double> data1({1.0, 2.0, 3.0, 4.0}, {2, 2});
    Tensor<double> data2({0.5, 1.5, 2.5, 3.5}, {2, 2});

    Variable<double> var1(data1, true);
    Variable<double> var2(data2, true);

    // Test addition
    auto result = var1 + var2;
    EXPECT_EQ(result->rows(), 2);
    EXPECT_EQ(result->cols(), 2);
    EXPECT_DOUBLE_EQ((*result)(0, 0), 1.5);
    EXPECT_DOUBLE_EQ((*result)(0, 1), 3.5);

    // Test backward pass
    result->backward();
    EXPECT_TRUE(var1.requires_grad());
    EXPECT_TRUE(var2.requires_grad());
}

TEST_F(SVMAutogradTest, ConstructorTest) {
    // Test valid constructors
    EXPECT_NO_THROW(SVM<double> svm());
    EXPECT_NO_THROW(SVM<double> svm(KernelType::LINEAR, 1.0));
    EXPECT_NO_THROW(SVM<double> svm(KernelType::RBF, 0.5, 0.1));
    EXPECT_NO_THROW(SVM<double> svm(KernelType::POLYNOMIAL, 2.0, 1.0, 3, 0.0));

    // Test invalid constructor (negative C)
    EXPECT_THROW(SVM<double> svm(KernelType::RBF, -1.0), std::invalid_argument);

    // Test invalid learning rate
    EXPECT_THROW(
        SVM<double> svm(KernelType::LINEAR, 1.0, 1.0, 3, 0.0, 1e-3, 1000, -0.01),
        std::invalid_argument);
}

TEST_F(SVMAutogradTest, LinearKernelAutogradTest) {
    SVM<double> svm(KernelType::LINEAR, 1.0, 1.0, 3, 0.0, 1e-3, 500, 0.01);

    // Test successful fit
    EXPECT_NO_THROW(svm.fit(X_linear_, y_linear_));

    // Test prediction
    std::vector<int> predictions = svm.predict(X_linear_);
    EXPECT_EQ(predictions.size(), y_linear_.size());

    // Check that predictions are valid
    for (int pred: predictions) {
        EXPECT_TRUE(pred == 1 || pred == -1);
    }

    // Test loss history
    std::vector<double> loss_history = svm.loss_history();
    EXPECT_GT(loss_history.size(), 0);

    // Loss should generally decrease (allowing for some fluctuation)
    if (loss_history.size() > 10) {
        double initial_loss = loss_history[0];
        double final_loss = loss_history.back();
        EXPECT_LE(final_loss, initial_loss * 2.0); // Allow some tolerance
    }
}

TEST_F(SVMAutogradTest, RBFKernelAutogradTest) {
    SVM<double> svm(KernelType::RBF, 1.0, 1.0, 3, 0.0, 1e-3, 200, 0.01);

    // Test fit and predict with RBF kernel
    EXPECT_NO_THROW(svm.fit(X_nonlinear_, y_nonlinear_));

    std::vector<int> predictions = svm.predict(X_nonlinear_);
    EXPECT_EQ(predictions.size(), y_nonlinear_.size());

    // Check that predictions are valid (-1 or 1)
    for (int pred: predictions) {
        EXPECT_TRUE(pred == 1 || pred == -1);
    }
}

TEST_F(SVMAutogradTest, PredictProbaTest) {
    SVM<double> svm(KernelType::LINEAR, 1.0, 1.0, 3, 0.0, 1e-3, 100, 0.01);
    svm.fit(X_linear_, y_linear_);

    Tensor<double> probabilities = svm.predict_proba(X_linear_);
    EXPECT_EQ(probabilities.rows(), X_linear_.rows());
    EXPECT_EQ(probabilities.cols(), 2);

    // Check that probabilities sum to 1 and are in [0, 1]
    for (size_t i = 0; i < probabilities.rows(); ++i) {
        double sum = probabilities(i, 0) + probabilities(i, 1);
        EXPECT_NEAR(sum, 1.0, 1e-6);
        EXPECT_GE(probabilities(i, 0), 0.0);
        EXPECT_LE(probabilities(i, 0), 1.0);
        EXPECT_GE(probabilities(i, 1), 0.0);
        EXPECT_LE(probabilities(i, 1), 1.0);
    }
}

TEST_F(SVMAutogradTest, DecisionFunctionTest) {
    SVM<double> svm(KernelType::LINEAR, 1.0, 1.0, 3, 0.0, 1e-3, 100, 0.01);
    svm.fit(X_linear_, y_linear_);

    std::vector<double> decision_values = svm.decision_function(X_linear_);
    EXPECT_EQ(decision_values.size(), X_linear_.rows());

    // Decision values should be consistent with predictions
    std::vector<int> predictions = svm.predict(X_linear_);
    for (size_t i = 0; i < decision_values.size(); ++i) {
        if (decision_values[i] >= 0) {
            EXPECT_EQ(predictions[i], 1);
        } else {
            EXPECT_EQ(predictions[i], -1);
        }
    }
}

TEST_F(SVMAutogradTest, ComparisonWithOriginalSVMTest) {
    // Compare autograd SVM with original SVM implementation

    // Original SVM
    SVM<double> original_svm(KernelType::LINEAR, 1.0);
    auto start_original = std::chrono::high_resolution_clock::now();
    original_svm.fit(X_linear_, y_linear_);
    auto end_original = std::chrono::high_resolution_clock::now();
    auto duration_original = std::chrono::duration_cast<
        std::chrono::milliseconds>(end_original - start_original);

    std::vector<int> original_predictions = original_svm.predict(X_linear_);

    // Autograd SVM
    SVM<double> autograd_svm(KernelType::LINEAR, 1.0, 1.0, 3, 0.0, 1e-3, 200,
                             0.01);
    auto start_autograd = std::chrono::high_resolution_clock::now();
    autograd_svm.fit(X_linear_, y_linear_);
    auto end_autograd = std::chrono::high_resolution_clock::now();
    auto duration_autograd = std::chrono::duration_cast<
        std::chrono::milliseconds>(end_autograd - start_autograd);

    std::vector<int> autograd_predictions = autograd_svm.predict(X_linear_);

    // Both should produce valid predictions
    EXPECT_EQ(original_predictions.size(), y_linear_.size());
    EXPECT_EQ(autograd_predictions.size(), y_linear_.size());

    // Calculate accuracy for both
    int original_correct = 0, autograd_correct = 0;
    for (size_t i = 0; i < y_linear_.size(); ++i) {
        if (original_predictions[i] == y_linear_[i])
            original_correct++;
        if (autograd_predictions[i] == y_linear_[i])
            autograd_correct++;
    }

    double original_accuracy = static_cast<double>(original_correct) / y_linear_.
                               size();
    double autograd_accuracy = static_cast<double>(autograd_correct) / y_linear_.
                               size();

    std::cout << "Original SVM accuracy: " << original_accuracy << ", Time: " <<
            duration_original.count() << "ms"
            << std::endl;
    std::cout << "Autograd SVM accuracy: " << autograd_accuracy << ", Time: " <<
            duration_autograd.count() << "ms"
            << std::endl;

    // Both should achieve reasonable accuracy
    EXPECT_GT(original_accuracy, 0.5);
    EXPECT_GT(autograd_accuracy, 0.5);
}

TEST_F(SVMAutogradTest, GradientComputationTest) {
    // Test that gradients are computed correctly
    Tensor<double> simple_data({1.0, -1.0}, {2, 1});
    std::vector<int> simple_labels = {1, -1};

    SVM<double> svm(KernelType::LINEAR, 1.0, 1.0, 3, 0.0, 1e-6, 50, 0.1);

    // Get initial loss
    std::vector<double> loss_before = svm.loss_history();

    svm.fit(simple_data, simple_labels);

    std::vector<double> loss_after = svm.loss_history();
    EXPECT_GT(loss_after.size(), 0);

    // Loss should change during training (indicating gradients are working)
    if (loss_after.size() > 1) {
        bool loss_changed = false;
        for (size_t i = 1; i < loss_after.size(); ++i) {
            if (std::abs(loss_after[i] - loss_after[i - 1]) > 1e-10) {
                loss_changed = true;
                break;
            }
        }
        EXPECT_TRUE(loss_changed);
    }
}

TEST_F(SVMAutogradTest, ErrorHandlingTest) {
    SVM<double> svm;

    // Test prediction before fitting
    EXPECT_THROW(svm.predict(X_linear_), std::runtime_error);
    EXPECT_THROW(svm.decision_function(X_linear_), std::runtime_error);

    // Test mismatched dimensions
    auto X_wrong = Tensor<double>::zeros({3, 2});
    std::vector<int> y_wrong = {1, -1}; // Wrong size
    EXPECT_THROW(svm.fit(X_wrong, y_wrong), std::invalid_argument);

    // Test multi-class (should fail)
    std::vector<int> y_multiclass = {1, 2, 3, 1, 2, 3, 1, 2};
    EXPECT_THROW(svm.fit(X_linear_, y_multiclass), std::invalid_argument);
}

TEST_F(SVMAutogradTest, KernelFunctionTest) {
    // Test different kernel types
    std::vector<KernelType> kernels = {
        KernelType::LINEAR, KernelType::RBF,
        KernelType::POLYNOMIAL,
        KernelType::SIGMOID
    };

    for (auto kernel: kernels) {
        SVM<double> svm(kernel, 1.0, 1.0, 2, 0.0, 1e-3, 50, 0.05);
        EXPECT_NO_THROW(svm.fit(X_1d_, y_1d_));

        std::vector<int> predictions = svm.predict(X_1d_);
        EXPECT_EQ(predictions.size(), y_1d_.size());

        // All predictions should be valid
        for (int pred: predictions) {
            EXPECT_TRUE(pred == 1 || pred == -1);
        }
    }
}
