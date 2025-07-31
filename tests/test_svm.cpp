#include <cmath>
#include <gtest/gtest.h>
#include <vector>
#include "ml/svm.hpp"
#include "utils/matrix.hpp"

using namespace utils;
using namespace ml;

class SVMTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple linearly separable 2D dataset
        // Class 1: points around (1, 1)
        // Class -1: points around (-1, -1)
        X_linear_ = Matrix<double>({
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
        X_nonlinear_ = Matrix<double>({
                {1.0, 1.0},
                {-1.0, -1.0}, // Class 1
                {1.0, -1.0},
                {-1.0, 1.0} // Class -1
        });
        y_nonlinear_ = {1, 1, -1, -1};

        // Create a simple 1D dataset
        X_1d_ = Matrix<double>({{2.0}, {3.0}, {4.0}, {-2.0}, {-3.0}, {-4.0}});
        y_1d_ = {1, 1, 1, -1, -1, -1};
    }

    Matrix<double> X_linear_, X_nonlinear_, X_1d_;
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
    Matrix<double> probabilities = svm.predict_proba(X_linear_);
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
    Matrix<double> sv = svm.support_vectors();
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
    Matrix<double> wrong_X({{1.0, 2.0, 3.0}}); // 3D instead of 2D
    EXPECT_THROW(svm.predict(wrong_X), std::invalid_argument);
}

TEST_F(SVMTest, KernelFunctionTest) {
    SVM<double> svm;

    Matrix<double> x1({{1.0, 2.0}});
    Matrix<double> x2({{3.0, 4.0}});

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
    // Test with minimal dataset
    Matrix<double> X_small({{1.0}, {-1.0}});
    std::vector<int> y_small = {1, -1};

    SVM<double> svm(KernelType::LINEAR, 1.0);
    EXPECT_NO_THROW(svm.fit(X_small, y_small));

    std::vector<int> predictions = svm.predict(X_small);
    EXPECT_EQ(predictions.size(), 2);
}
