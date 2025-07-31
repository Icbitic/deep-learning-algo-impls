#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
#include "../utils/matrix.hpp"

/**
 * @file pca.hpp
 * @brief Principal Component Analysis implementation
 * @author Kalenitid
 * @version 1.0.0
 */

namespace ml {
    using namespace utils;
    /**
     * @brief A class implementing Principal Component Analysis (PCA)
     *
     * This class provides functionality for dimensionality reduction using PCA.
     * It computes principal components of the input data and allows projection
     * onto a lower-dimensional space.
     *
     * @tparam T The data type for matrix elements (typically float or double)
     *
     * @example
     * ```cpp
     * // Create a PCA object
     * PCA<double> pca;
     *
     * // Fit PCA to data
     * Matrix<double> data = your_data_matrix;
     * pca.fit(data);
     *
     * // Transform data to reduced dimensions
     * Matrix<double> transformed = pca.transform(data, 2); // Reduce to 2 dimensions
     * ```
     */
    template<typename T>
    class PCA {
    public:
        /**
         * @brief Default constructor
         */
        PCA() = default;

        /**
         * @brief Fit the PCA model to the data
         *
         * @param data Input data matrix where rows are samples and columns are features
         * @param center Whether to center the data before computing PCA
         * @param scale Whether to scale the data to unit variance before computing PCA
         */
        void fit(const Matrix<T> &data, bool center = true, bool scale = false);

        /**
         * @brief Transform data to the principal component space
         *
         * @param data Input data matrix where rows are samples and columns are features
         * @param n_components Number of components to keep (if 0, keep all components)
         * @return Transformed data matrix
         */
        Matrix<T> transform(const Matrix<T> &data, size_t n_components = 0) const;

        /**
         * @brief Fit the model and transform the data in one step
         *
         * @param data Input data matrix where rows are samples and columns are features
         * @param n_components Number of components to keep (if 0, keep all components)
         * @param center Whether to center the data before computing PCA
         * @param scale Whether to scale the data to unit variance before computing PCA
         * @return Transformed data matrix
         */
        Matrix<T> fit_transform(const Matrix<T> &data, size_t n_components = 0, bool center = true, bool scale = false);

        /**
         * @brief Get the explained variance ratio for each component
         *
         * @return Vector of explained variance ratios
         */
        std::vector<T> explained_variance_ratio() const;

        /**
         * @brief Get the principal components (eigenvectors)
         *
         * @return Matrix where each column is a principal component
         */
        Matrix<T> components() const;

        /**
         * @brief Get the singular values (square roots of eigenvalues)
         *
         * @return Vector of singular values
         */
        std::vector<T> singular_values() const;

    private:
        Matrix<T> components_; ///< Principal components (eigenvectors)
        std::vector<T> singular_values_; ///< Singular values
        std::vector<T> explained_variance_; ///< Explained variance for each component
        std::vector<T> explained_variance_ratio_; ///< Explained variance ratio for each component
        std::vector<T> mean_; ///< Mean of each feature
        std::vector<T> scale_; ///< Scale (std dev) of each feature
        bool is_fitted_ = false; ///< Whether the model has been fitted
    };

    // Type aliases for common use cases
    using PCAF = PCA<float>;
    using PCAD = PCA<double>;

} // namespace ml
