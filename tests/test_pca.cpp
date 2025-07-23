#include <gtest/gtest.h>
#include "utils/matrix.hpp"
#include "utils/pca.hpp"

using namespace dl::utils;

class PCATest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple dataset for testing
        // 2D data with clear principal components
        data_ = Matrix<double>({
                {1.0, 1.0},
                {-1.0, -1.0},
                {2.0, 2.0},
                {-2.0, -2.0},
                {3.0, 3.0},
                {-3.0, -3.0},
        });

        // Create a more complex dataset
        // 3D data with different variances along different axes
        complex_data_ = Matrix<double>({
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

    Matrix<double> data_;
    Matrix<double> complex_data_;
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

TEST_F(PCATest, TransformTest) {
    PCAD pca;
    pca.fit(data_);

    // Transform to 1 dimension
    auto transformed = pca.transform(data_, 1);

    // Check dimensions
    EXPECT_EQ(transformed.rows(), data_.rows());
    EXPECT_EQ(transformed.cols(), 1);

    // Check that the transformed data preserves the relative distances
    // Points that were far apart should still be far apart
    double dist_original = std::sqrt(std::pow(data_(0, 0) - data_(5, 0), 2) + std::pow(data_(0, 1) - data_(5, 1), 2));
    double dist_transformed = std::abs(transformed(0, 0) - transformed(5, 0));

    // The ratio should be approximately constant
    EXPECT_NEAR(dist_original / dist_transformed, std::sqrt(2.0), 1e-5);
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
    double dot_product = components(0, 0) * components(0, 1) + components(1, 0) * components(1, 1);
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

TEST_F(PCATest, ScalingTest) {
    // Create data with different scales
    Matrix<double> scaled_data({
            {1.0, 100.0},
            {-1.0, -100.0},
            {2.0, 200.0},
            {-2.0, -200.0},
    });

    // Test without scaling
    PCAD pca1;
    pca1.fit(scaled_data, true, false);
    auto variance_ratio1 = pca1.explained_variance_ratio();

    // Without scaling, the second feature (with larger values) should dominate
    EXPECT_GT(variance_ratio1[0], 0.99);

    // Test with scaling
    PCAD pca2;
    pca2.fit(scaled_data, true, true);
    auto variance_ratio2 = pca2.explained_variance_ratio();

    // With scaling, the variance should be more evenly distributed
    EXPECT_NEAR(variance_ratio2[0], 0.5, 0.1);
    EXPECT_NEAR(variance_ratio2[1], 0.5, 0.1);
}

TEST_F(PCATest, ErrorHandlingTest) {
    PCAD pca;

    // Test with empty matrix
    Matrix<double> empty;
    EXPECT_THROW(pca.fit(empty), std::invalid_argument);

    // Test transform before fit
    EXPECT_THROW(pca.transform(data_), std::runtime_error);

    // Test with incompatible dimensions
    pca.fit(data_);
    Matrix<double> wrong_dims(3, 3);
    EXPECT_THROW(pca.transform(wrong_dims), std::invalid_argument);
}
