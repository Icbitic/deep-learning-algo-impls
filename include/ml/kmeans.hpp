#pragma once

#include <limits>
#include <random>
#include <vector>
#include "../utils/tensor.hpp"

/**
 * @file kmeans.hpp
 * @brief K-Means clustering algorithm implementation
 * @author Kalenitid
 * @version 1.0.0
 */

namespace ml {
    using namespace utils;

    /**
     * @brief A class implementing K-Means clustering algorithm
     *
     * This class provides functionality for unsupervised clustering using the
     * K-Means algorithm. It partitions data into k clusters by minimizing
     * within-cluster sum of squares.
     *
     * @tparam T The data type for matrix elements (typically float or double)
     *
     * @example
     * ```cpp
     * // Create a K-Means object with 3 clusters
     * KMeans<double> kmeans(3);
     *
     * // Fit K-Means to data
     * Tensor<double> data = your_data_tensor;
     * kmeans.fit(data);
     *
     * // Predict cluster labels for new data
     * std::vector<int> labels = kmeans.predict(data);
     * ```
     */
    template<typename T>
    class KMeans {
    public:
        /**
         * @brief Constructor with number of clusters
         *
         * @param k Number of clusters
         * @param max_iters Maximum number of iterations
         * @param tol Tolerance for convergence
         * @param random_state Random seed for reproducibility
         */
        KMeans(size_t k, size_t max_iters = 300, T tol = 1e-4, int random_state = -1);

        /**
         * @brief Fit the K-Means model to the data
         *
         * @param data Input data matrix where rows are samples and columns are features
         */
        void fit(const Tensor<T> &data);

        /**
         * @brief Predict cluster labels for the given data
         *
         * @param data Input data matrix
         * @return Vector of cluster labels (0 to k-1)
         */
        std::vector<int> predict(const Tensor<T> &data) const;

        /**
         * @brief Fit the model and predict cluster labels
         *
         * @param data Input data matrix
         * @return Vector of cluster labels (0 to k-1)
         */
        std::vector<int> fit_predict(const Tensor<T> &data);

        /**
         * @brief Get the cluster centroids
         *
         * @return Matrix containing cluster centroids
         */
        Tensor<T> cluster_centers() const;

        /**
         * @brief Get the within-cluster sum of squares (inertia)
         *
         * @return The inertia value
         */
        T inertia() const;

        /**
         * @brief Get the number of iterations performed
         *
         * @return Number of iterations
         */
        size_t n_iter() const;

    private:
        size_t k_; ///< Number of clusters
        size_t max_iters_; ///< Maximum number of iterations
        T tol_; ///< Tolerance for convergence
        int random_state_; ///< Random seed
        Tensor<T> centroids_; ///< Cluster centroids
        T inertia_; ///< Within-cluster sum of squares
        size_t n_iter_; ///< Number of iterations performed
        bool is_fitted_; ///< Whether the model has been fitted

        /**
         * @brief Initialize centroids randomly
         *
         * @param data Input data matrix
         */
        void init_centroids(const Tensor<T> &data);

        /**
         * @brief Assign each point to the nearest centroid
         *
         * @param data Input data matrix
         * @return Vector of cluster assignments
         */
        std::vector<int> assign_clusters(const Tensor<T> &data) const;

        /**
         * @brief Update centroids based on current assignments
         *
         * @param data Input data matrix
         * @param labels Current cluster assignments
         */
        void update_centroids(const Tensor<T> &data, const std::vector<int> &labels);

        /**
         * @brief Calculate squared Euclidean distance between two points
         *
         * @param point1 First point
         * @param point2 Second point
         * @return Squared distance
         */
        T squared_distance(const std::vector<T> &point1, const std::vector<T> &point2) const;
    };

    /// Type aliases for common use cases
    using KMeansF = KMeans<float>;
    using KMeansD = KMeans<double>;

} // namespace ml
