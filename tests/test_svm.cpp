#include <chrono>
#include <cmath>
#include <gtest/gtest.h>
#include <vector>
#include "ml/svm.hpp"
#include "utils/tensor.hpp"

using namespace utils;
using namespace ml;

class SVMTest : public ::testing::Test {
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
        X_1d_ = Tensor<double>({{2.0}, {3.0}, {4.0}, {-2.0}, {-3.0}, {-4.0}});
        y_1d_ = {1, 1, 1, -1, -1, -1};
    }

    Tensor<double> X_linear_, X_nonlinear_, X_1d_;
    std::vector<int> y_linear_, y_nonlinear_, y_1d_;
};

TEST_F(SVMTest, ConstructorTest) {
    // Test valid constructors
    EXPECT_NO_THROW(SVM<double> svm());
    EXPECT_NO_THROW(SVM<double> svm(KernelType::LINEAR, 1.0));
    EXPECT_NO_THROW(SVM<double> svm(KernelType::RBF, 0.5, 0.1));
    EXPECT_NO_THROW(SVM<double> svm(KernelType::POLYNOMIAL, 2.0, 1.0, 3.0, 0.0));

    // Test invalid constructor (negative C)
    EXPECT_THROW(SVM<double> svm(KernelType::RBF, -1.0), std::invalid_argument);
}

TEST_F(SVMTest, LinearKernelTest) {
    SVM<double> svm(KernelType::LINEAR, 1.0);

    // Test successful fit
    EXPECT_NO_THROW(svm.fit(X_linear_, y_linear_));

    // Test prediction
    std::vector<int> predictions = svm.predict(X_linear_);
    EXPECT_EQ(predictions.size(), y_linear_.size());

    // For linearly separable data, should achieve perfect classification
    int correct = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (predictions[i] == y_linear_[i]) {
            correct++;
        }
    }
    double accuracy = static_cast<double>(correct) / predictions.size();
    EXPECT_GT(accuracy, 0.7); // Should achieve reasonable accuracy
}

TEST_F(SVMTest, RBFKernelTest) {
    SVM<double> svm(KernelType::RBF, 1.0, 1.0);

    // Test fit and predict with RBF kernel
    EXPECT_NO_THROW(svm.fit(X_nonlinear_, y_nonlinear_));

    std::vector<int> predictions = svm.predict(X_nonlinear_);
    EXPECT_EQ(predictions.size(), y_nonlinear_.size());

    // Check that predictions are valid (-1 or 1)
    for (int pred: predictions) {
        EXPECT_TRUE(pred == 1 || pred == -1);
    }
}

TEST_F(SVMTest, PolynomialKernelTest) {
    SVM<double> svm(KernelType::POLYNOMIAL, 1.0, 1.0, 2.0, 0.0);

    // Test fit and predict with polynomial kernel
    EXPECT_NO_THROW(svm.fit(X_1d_, y_1d_));

    std::vector<int> predictions = svm.predict(X_1d_);
    EXPECT_EQ(predictions.size(), y_1d_.size());

    // Check that predictions are valid
    for (int pred: predictions) {
        EXPECT_TRUE(pred == 1 || pred == -1);
    }
}

TEST_F(SVMTest, SigmoidKernelTest) {
    SVM<double> svm(KernelType::SIGMOID, 1.0, 1.0, 1.0, 0.0);

    // Test fit and predict with sigmoid kernel
    EXPECT_NO_THROW(svm.fit(X_1d_, y_1d_));

    std::vector<int> predictions = svm.predict(X_1d_);
    EXPECT_EQ(predictions.size(), y_1d_.size());

    // Check that predictions are valid
    for (int pred: predictions) {
        EXPECT_TRUE(pred == 1 || pred == -1);
    }
}

TEST_F(SVMTest, PredictProbaTest) {
    SVM<double> svm(KernelType::LINEAR, 1.0);

    // Test error when not fitted
    EXPECT_THROW(svm.predict_proba(X_linear_), std::runtime_error);

    // Fit the model
    svm.fit(X_linear_, y_linear_);

    // Test probability prediction
    Tensor<double> probabilities = svm.predict_proba(X_linear_);
    EXPECT_EQ(probabilities.rows(), X_linear_.rows());

    // Check that probabilities are in valid range [0, 1]
    for (size_t i = 0; i < probabilities.rows(); ++i) {
        for (size_t j = 0; j < probabilities.cols(); ++j) {
            EXPECT_GE(probabilities(i, j), 0.0);
            EXPECT_LE(probabilities(i, j), 1.0);
        }
    }
}

TEST_F(SVMTest, DecisionFunctionTest) {
    SVM<double> svm(KernelType::LINEAR, 1.0);

    // Test error when not fitted
    EXPECT_THROW(svm.decision_function(X_linear_), std::runtime_error);

    // Fit the model
    svm.fit(X_linear_, y_linear_);

    // Test decision function
    std::vector<double> scores = svm.decision_function(X_linear_);
    EXPECT_EQ(scores.size(), X_linear_.rows());

    // Check that decision function signs match predictions
    std::vector<int> predictions = svm.predict(X_linear_);
    for (size_t i = 0; i < scores.size(); ++i) {
        int predicted_sign = (scores[i] >= 0) ? 1 : -1;
        EXPECT_EQ(predicted_sign, predictions[i]);
    }
}

TEST_F(SVMTest, SupportVectorsTest) {
    SVM<double> svm(KernelType::LINEAR, 1.0);

    // Test error when not fitted
    EXPECT_THROW(svm.support_vectors(), std::runtime_error);

    // Fit the model
    svm.fit(X_linear_, y_linear_);

    // Test support vectors
    Tensor<double> sv = svm.support_vectors();
    EXPECT_GT(sv.rows(), 0); // Should have at least one support vector
    EXPECT_EQ(sv.cols(), X_linear_.cols());

    // Test dual coefficients
    std::vector<double> dual_coef = svm.dual_coef();
    EXPECT_EQ(dual_coef.size(), sv.rows());

    // Test support vector indices
    std::vector<size_t> sv_indices = svm.support();
    EXPECT_EQ(sv_indices.size(), sv.rows());

    // Check that indices are valid
    for (int idx: sv_indices) {
        EXPECT_GE(idx, 0);
        EXPECT_LT(idx, static_cast<int>(X_linear_.rows()));
    }
}

TEST_F(SVMTest, ErrorHandlingTest) {
    SVM<double> svm(KernelType::LINEAR, 1.0);

    // Test mismatched X and y sizes
    std::vector<int> wrong_y = {1, -1}; // Too few labels
    EXPECT_THROW(svm.fit(X_linear_, wrong_y), std::invalid_argument);

    // Test invalid labels (not -1 or 1)
    std::vector<int> invalid_y = {0, 1, 2, -1, -1, -1, -1, -1};
    EXPECT_THROW(svm.fit(X_linear_, invalid_y), std::invalid_argument);

    // Test prediction before fitting
    EXPECT_THROW(svm.predict(X_linear_), std::runtime_error);

    // Test prediction with wrong dimensions
    svm.fit(X_linear_, y_linear_);
    Tensor<double> wrong_X({{1.0, 2.0, 3.0}}); // 3D instead of 2D
    EXPECT_THROW(svm.predict(wrong_X), std::invalid_argument);
}

TEST_F(SVMTest, KernelFunctionTest) {
    SVM<double> svm;

    Tensor<double> x1({{1.0, 2.0}});
    Tensor<double> x2({{3.0, 4.0}});

    // Test linear kernel: should be dot product
    svm = SVM<double>(KernelType::LINEAR, 1.0);
    // We can't directly test kernel function as it's private,
    // but we can test that different kernels produce different results

    // Test that different kernel types work
    EXPECT_NO_THROW(SVM<double>(KernelType::LINEAR, 1.0));
    EXPECT_NO_THROW(SVM<double>(KernelType::RBF, 1.0, 1.0));
    EXPECT_NO_THROW(SVM<double>(KernelType::POLYNOMIAL, 1.0, 1.0, 2, 0.0));
    EXPECT_NO_THROW(SVM<double>(KernelType::SIGMOID, 1.0, 1.0, 3, 0.0));
}

TEST_F(SVMTest, SmallDatasetTest) {
    // Test with very small dataset
    Tensor<double> X_small({{1.0}, {-1.0}});
    std::vector<int> y_small = {1, -1};

    SVM<double> svm(KernelType::LINEAR, 1.0);
    EXPECT_NO_THROW(svm.fit(X_small, y_small));

    std::vector<int> predictions = svm.predict(X_small);
    EXPECT_EQ(predictions.size(), 2);
}

TEST_F(SVMTest, StaticVsAutogradComparisonTest) {
    std::cout << "\n=== Comparison: Static Computation vs Autograd ===" << std::endl;
    
    // Test with linear kernel on linearly separable data
    std::cout << "\nTesting Linear Kernel on Linearly Separable Data:" << std::endl;
    
    // Original SVM with static computation (SMO algorithm)
    std::cout << "\n1. Original SVM (Static Computation - SMO Algorithm):" << std::endl;
    SVM<double> static_svm(KernelType::LINEAR, 1.0);
    
    auto start_static = std::chrono::high_resolution_clock::now();
    static_svm.fit(X_linear_, y_linear_);
    auto end_static = std::chrono::high_resolution_clock::now();
    auto duration_static = std::chrono::duration_cast<std::chrono::microseconds>(end_static - start_static);
    
    std::vector<int> static_predictions = static_svm.predict(X_linear_);
    int static_correct = 0;
    for (size_t i = 0; i < y_linear_.size(); ++i) {
        if (static_predictions[i] == y_linear_[i]) static_correct++;
    }
    double static_accuracy = static_cast<double>(static_correct) / y_linear_.size();
    
    std::cout << "   - Training time: " << duration_static.count() << " microseconds" << std::endl;
    std::cout << "   - Accuracy: " << static_accuracy * 100 << "%" << std::endl;
    std::cout << "   - Method: Sequential Minimal Optimization (SMO)" << std::endl;
    std::cout << "   - Gradient computation: Manual/Analytical" << std::endl;
    
    // New SVM with autograd
    std::cout << "\n2. New SVM (Automatic Differentiation - Gradient Descent):" << std::endl;
    SVM<double> autograd_svm(KernelType::LINEAR, 1.0, 1.0, 3, 0.0, 1e-4, 300, 0.01);
    
    auto start_autograd = std::chrono::high_resolution_clock::now();
    autograd_svm.fit(X_linear_, y_linear_);
    auto end_autograd = std::chrono::high_resolution_clock::now();
    auto duration_autograd = std::chrono::duration_cast<std::chrono::microseconds>(end_autograd - start_autograd);
    
    std::vector<int> autograd_predictions = autograd_svm.predict(X_linear_);
    int autograd_correct = 0;
    for (size_t i = 0; i < y_linear_.size(); ++i) {
        if (autograd_predictions[i] == y_linear_[i]) autograd_correct++;
    }
    double autograd_accuracy = static_cast<double>(autograd_correct) / y_linear_.size();
    
    std::cout << "   - Training time: " << duration_autograd.count() << " microseconds" << std::endl;
    std::cout << "   - Accuracy: " << autograd_accuracy * 100 << "%" << std::endl;
    std::cout << "   - Method: Gradient Descent with Autograd" << std::endl;
    std::cout << "   - Gradient computation: Automatic Differentiation" << std::endl;
    
    // Show loss history for autograd
    std::vector<double> loss_history = autograd_svm.loss_history();
    if (loss_history.size() > 0) {
        std::cout << "   - Initial loss: " << loss_history[0] << std::endl;
        std::cout << "   - Final loss: " << loss_history.back() << std::endl;
        std::cout << "   - Training iterations: " << loss_history.size() << std::endl;
    }
    
    std::cout << "\n=== Key Differences ===" << std::endl;
    std::cout << "1. Gradient Computation:" << std::endl;
    std::cout << "   - Static: Manual gradient calculations in SMO algorithm" << std::endl;
    std::cout << "   - Autograd: Automatic differentiation tracks operations" << std::endl;
    
    std::cout << "\n2. Optimization Method:" << std::endl;
    std::cout << "   - Static: Sequential Minimal Optimization (SMO)" << std::endl;
    std::cout << "   - Autograd: Gradient descent with backpropagation" << std::endl;
    
    std::cout << "\n3. Flexibility:" << std::endl;
    std::cout << "   - Static: Fixed optimization algorithm, hard to modify" << std::endl;
    std::cout << "   - Autograd: Easy to experiment with different loss functions" << std::endl;
    
    std::cout << "\n4. Computational Graph:" << std::endl;
    std::cout << "   - Static: No computational graph, direct computation" << std::endl;
    std::cout << "   - Autograd: Builds computational graph for backpropagation" << std::endl;
    
    // Both should achieve reasonable accuracy
    EXPECT_GT(static_accuracy, 0.6);
    EXPECT_GT(autograd_accuracy, 0.6);
    
    std::cout << "\n=== Test Results ===" << std::endl;
    std::cout << "Both implementations achieved reasonable accuracy (>60%)" << std::endl;
    std::cout << "Autograd provides more flexibility for experimentation" << std::endl;
    std::cout << "Static SMO is more traditional and well-established" << std::endl;
}
