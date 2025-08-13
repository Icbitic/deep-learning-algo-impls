#include <gtest/gtest.h>
#include "ml/pca.hpp"
#include "utils/tensor.hpp"

using namespace utils;
using namespace ml;

class PCATest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple dataset for testing
        // 2D data with clear principal components
        data_ = Tensor<double>({
            {1.0, 1.0},
            {-1.0, -1.0},
            {2.0, 2.0},
            {-2.0, -2.0},
            {3.0, 3.0},
            {-3.0, -3.0},
        });

        // Create a more complex dataset
        // 3D data with different variances along different axes
        complex_data_ = Tensor<double>({
            {1.0, 2.0, 0.5},
            {-1.0, -2.0, -0.5},
            {2.0, 4.0, 1.0},
            {-2.0, -4.0, -1.0},
            {3.0, 6.0, 1.5},
            {-3.0, -6.0, -1.5},
            {0.5, 1.0, 0.25},
            {-0.5, -1.0, -0.25},
        });
    }

    Tensor<double> data_;
    Tensor<double> complex_data_;
};

TEST_F(PCATest, ConstructorTest) {
    // Test default constructor
    EXPECT_NO_THROW({ PCAD pca; });
}

TEST_F(PCATest, FitTest) {
    PCAD pca;

    // Test fitting the model
    EXPECT_NO_THROW({ pca.fit(data_); });

    // Test explained variance ratio
    auto variance_ratio = pca.explained_variance_ratio();
    EXPECT_EQ(variance_ratio.size(), 2);

    // First component should explain most of the variance (close to 1.0)
    EXPECT_NEAR(variance_ratio[0], 1.0, 1e-5);

    // Second component should explain very little variance (close to 0.0)
    EXPECT_NEAR(variance_ratio[1], 0.0, 1e-5);
}


TEST_F(PCATest, FitTransformTest) {
    PCAD pca;

    // Test fit_transform
    auto transformed = pca.fit_transform(data_, 1);

    // Check dimensions
    EXPECT_EQ(transformed.rows(), data_.rows());
    EXPECT_EQ(transformed.cols(), 1);
}

TEST_F(PCATest, ComponentsTest) {
    PCAD pca;
    pca.fit(data_);

    // Get components
    auto components = pca.components();

    // Check dimensions
    EXPECT_EQ(components.rows(), 2);
    EXPECT_EQ(components.cols(), 2);

    // First component should be approximately [1/sqrt(2), 1/sqrt(2)] or [-1/sqrt(2), -1/sqrt(2)]
    double comp_abs = std::abs(components(0, 0));
    EXPECT_NEAR(comp_abs, 1.0 / std::sqrt(2.0), 1e-5);

    // Components should be orthogonal
    double dot_product = components(0, 0) * components(0, 1) + components(1, 0) *
                         components(1, 1);
    EXPECT_NEAR(dot_product, 0.0, 1e-5);
}

TEST_F(PCATest, ComplexDataTest) {
    PCAD pca;
    pca.fit(complex_data_);

    // Test explained variance ratio
    auto variance_ratio = pca.explained_variance_ratio();
    EXPECT_EQ(variance_ratio.size(), 3);

    // First component should explain most of the variance
    EXPECT_GT(variance_ratio[0], 0.9);

    // Sum of all ratios should be 1.0
    double sum = 0.0;
    for (auto ratio: variance_ratio) {
        sum += ratio;
    }
    EXPECT_NEAR(sum, 1.0, 1e-5);

    // Transform to 2 dimensions
    auto transformed = pca.transform(complex_data_, 2);

    // Check dimensions
    EXPECT_EQ(transformed.rows(), complex_data_.rows());
    EXPECT_EQ(transformed.cols(), 2);
}

TEST_F(PCATest, ErrorHandlingTest) {
    PCAD pca;

    // Test with empty matrix
    Tensor<double> empty;
    EXPECT_THROW(pca.fit(empty), std::invalid_argument);

    // Test transform before fit
    EXPECT_THROW(pca.transform(data_), std::runtime_error);

    // Test with incompatible dimensions
    pca.fit(data_);
    auto wrong_dims = Tensor<double>::zeros({3, 3});
    EXPECT_THROW(pca.transform(wrong_dims), std::invalid_argument);
}
