#include <gtest/gtest.h>
#include <set>
#include <vector>
#include "ml/kmeans.hpp"
#include "utils/tensor.hpp"

using namespace dl;
using namespace ml;

class KMeansTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple 2D dataset with clear clusters
        // Cluster 1: around (0, 0)
        // Cluster 2: around (5, 5)
        // Cluster 3: around (0, 5)
        data_ = Tensor<double>({
                                   0.0, 0.0,
                                   0.5, 0.2,
                                   -0.3, 0.1,
                                   0.1, -0.4, // Cluster 1
                                   5.0, 5.0,
                                   5.2, 4.8,
                                   4.9, 5.3,
                                   5.1, 4.7, // Cluster 2
                                   0.0, 5.0,
                                   0.3, 4.8,
                                   -0.2, 5.2,
                                   0.1, 4.9 // Cluster 3
                               }, {12, 2});

        // Create a 1D dataset for edge case testing
        simple_data_ = Tensor<double>({1.0, 1.1, 1.2, 5.0, 5.1, 5.2}, {6, 1});
    }

    Tensor<double> data_;
    Tensor<double> simple_data_;
};

TEST_F(KMeansTest, ConstructorTest) {
    // Test valid constructor
    EXPECT_NO_THROW(KMeans<double> kmeans(3));
    EXPECT_NO_THROW(KMeans<double> kmeans(3, 100, 1e-4, 42));

    // Test invalid constructor
    EXPECT_THROW(KMeans<double> kmeans(0), std::invalid_argument);
}

TEST_F(KMeansTest, FitTest) {
    KMeans<double> kmeans(3, 100, 1e-4, 42);

    // Test successful fit
    EXPECT_NO_THROW(kmeans.fit(data_));

    // Test that cluster centers are computed
    Tensor<double> centers = kmeans.cluster_centers();
    EXPECT_EQ(centers.rows(), 3u);
    EXPECT_EQ(centers.cols(), 2);

    // Test that inertia is computed
    double inertia = kmeans.inertia();
    EXPECT_GE(inertia, 0.0);

    // Test error case: too few samples
    Tensor<double> small_data({1.0, 2.0, 3.0, 4.0}, {2, 2});
    KMeans<double> kmeans_small(3);
    EXPECT_THROW(kmeans_small.fit(small_data), std::invalid_argument);
}

TEST_F(KMeansTest, PredictTest) {
    KMeans<double> kmeans(3, 100, 1e-4, 42);

    // Test error when not fitted
    EXPECT_THROW(kmeans.predict(data_), std::runtime_error);

    // Fit the model
    kmeans.fit(data_);

    // Test prediction
    std::vector<int> labels = kmeans.predict(data_);
    EXPECT_EQ(labels.size(), data_.rows());

    // Check that all labels are in valid range [0, k-1]
    for (int label: labels) {
        EXPECT_GE(label, 0);
        EXPECT_LT(label, 3);
    }

    // Test that we get 3 different clusters (with high probability)
    std::set<int> unique_labels(labels.begin(), labels.end());
    EXPECT_EQ(unique_labels.size(), 3);
}

TEST_F(KMeansTest, FitPredictTest) {
    KMeans<double> kmeans(3, 100, 1e-4, 42);

    // Test fit_predict
    std::vector<int> labels = kmeans.fit_predict(data_);
    EXPECT_EQ(labels.size(), data_.rows());

    // Verify that the model is fitted after fit_predict
    EXPECT_NO_THROW(kmeans.cluster_centers());
    EXPECT_NO_THROW(kmeans.inertia());
}

TEST_F(KMeansTest, ClusterCentersTest) {
    KMeans<double> kmeans(2, 100, 1e-4, 42);

    // Test error when not fitted
    EXPECT_THROW(kmeans.cluster_centers(), std::runtime_error);

    // Fit and test
    kmeans.fit(simple_data_);
    Tensor<double> centers = kmeans.cluster_centers();

    EXPECT_EQ(centers.rows(), 2);
    EXPECT_EQ(centers.cols(), 1);

    // Centers should be approximately at the cluster means
    // One cluster around 1.0, another around 5.0
    std::vector<double> center_values = {centers(0, 0), centers(1, 0)};
    std::sort(center_values.begin(), center_values.end());

    EXPECT_NEAR(center_values[0], 1.1, 0.5); // First cluster center
    EXPECT_NEAR(center_values[1], 5.1, 0.5); // Second cluster center
}

TEST_F(KMeansTest, InertiaTest) {
    KMeans<double> kmeans(2, 100, 1e-4, 42);

    // Test error when not fitted
    EXPECT_THROW(kmeans.inertia(), std::runtime_error);

    // Fit and test
    kmeans.fit(simple_data_);
    double inertia = kmeans.inertia();

    // Inertia should be non-negative
    EXPECT_GE(inertia, 0.0);

    // For well-separated clusters, inertia should be relatively small
    EXPECT_LT(inertia, 1.0);
}

TEST_F(KMeansTest, ConvergenceTest) {
    KMeans<double> kmeans(3, 100, 1e-4, 42);

    kmeans.fit(data_);

    // Test that algorithm converged (didn't hit max iterations)
    EXPECT_LT(kmeans.n_iter(), 100u);

    // Test with very tight tolerance (should take more iterations)
    KMeans<double> kmeans_tight(3, 100, 1e-10, 42);
    kmeans_tight.fit(data_);

    // Should still converge, but might take more iterations
    EXPECT_LE(kmeans_tight.n_iter(), 100u);
}

TEST_F(KMeansTest, ReproducibilityTest) {
    // Test that same random seed produces same results
    KMeans<double> kmeans1(3, 100, 1e-4, 42);
    KMeans<double> kmeans2(3, 100, 1e-4, 42);

    std::vector<int> labels1 = kmeans1.fit_predict(data_);
    std::vector<int> labels2 = kmeans2.fit_predict(data_);

    // Results should be identical with same seed
    EXPECT_EQ(labels1, labels2);

    // Cluster centers should also be the same
    Tensor<double> centers1 = kmeans1.cluster_centers();
    Tensor<double> centers2 = kmeans2.cluster_centers();

    for (size_t i = 0; i < centers1.rows(); ++i) {
        for (size_t j = 0; j < centers1.cols(); ++j) {
            EXPECT_NEAR(centers1(i, j), centers2(i, j), 1e-10);
        }
    }
}
