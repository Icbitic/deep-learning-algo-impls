#include "utils/pca.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <xtensor-blas/xlinalg.hpp>

namespace dl {
    namespace utils {

        template<typename T>
        void PCA<T>::fit(const Matrix<T> &data, bool center, bool scale) {
            if (data.rows() < 2 || data.cols() < 2) {
                throw std::invalid_argument("Data matrix must have at least 2 rows and 2 columns");
            }

            // Get data dimensions
            size_t n_samples = data.rows();
            size_t n_features = data.cols();

            // Center the data (subtract mean)
            mean_.resize(n_features);
            Matrix<T> centered_data = data;

            if (center) {
                // Compute mean for each feature
                for (size_t j = 0; j < n_features; ++j) {
                    T sum = 0;
                    for (size_t i = 0; i < n_samples; ++i) {
                        sum += data(i, j);
                    }
                    mean_[j] = sum / static_cast<T>(n_samples);

                    // Center the data
                    for (size_t i = 0; i < n_samples; ++i) {
                        centered_data(i, j) -= mean_[j];
                    }
                }
            } else {
                // If not centering, set mean to zero
                std::fill(mean_.begin(), mean_.end(), 0);
            }

            // Scale the data (divide by standard deviation)
            scale_.resize(n_features, 1.0);

            if (scale) {
                // Compute standard deviation for each feature
                for (size_t j = 0; j < n_features; ++j) {
                    T sum_sq = 0;
                    for (size_t i = 0; i < n_samples; ++i) {
                        T diff = centered_data(i, j);
                        sum_sq += diff * diff;
                    }
                    T variance = sum_sq / static_cast<T>(n_samples);
                    scale_[j] = std::sqrt(variance);

                    // Avoid division by zero
                    if (scale_[j] < 1e-10) {
                        scale_[j] = 1.0;
                    }

                    // Scale the data
                    for (size_t i = 0; i < n_samples; ++i) {
                        centered_data(i, j) /= scale_[j];
                    }
                }
            }

            // Compute SVD using xtensor-blas
            // X = U * S * V^T where V contains the principal components
            auto X = centered_data.data(); // Get xtensor array from Matrix

            // Compute covariance matrix (X^T * X) / (n_samples - 1)
            auto cov = xt::linalg::dot(xt::transpose(X), X) / static_cast<T>(n_samples - 1);

            // Compute eigendecomposition
            auto eigen_result = xt::linalg::eigh(cov);
            auto eigenvalues = std::get<0>(eigen_result);
            auto eigenvectors = std::get<1>(eigen_result);

            // Sort eigenvalues and eigenvectors in descending order
            std::vector<size_t> indices(eigenvalues.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::sort(indices.begin(), indices.end(),
                      [&eigenvalues](size_t i1, size_t i2) { return eigenvalues(i1) > eigenvalues(i2); });

            // Reorder eigenvalues and eigenvectors
            singular_values_.resize(n_features);
            explained_variance_.resize(n_features);
            explained_variance_ratio_.resize(n_features);

            T total_variance = 0;
            for (size_t i = 0; i < n_features; ++i) {
                explained_variance_[i] = eigenvalues(indices[i]);
                total_variance += explained_variance_[i];
            }

            // Create components matrix
            components_ = Matrix<T>(n_features, n_features);

            for (size_t i = 0; i < n_features; ++i) {
                // Compute singular values (sqrt of eigenvalues)
                singular_values_[i] = std::sqrt(explained_variance_[i]);

                // Compute explained variance ratio
                explained_variance_ratio_[i] = explained_variance_[i] / total_variance;

                // Store eigenvectors as columns in components matrix
                for (size_t j = 0; j < n_features; ++j) {
                    components_(j, i) = eigenvectors(j, indices[i]);
                }
            }

            is_fitted_ = true;
        }

        template<typename T>
        Matrix<T> PCA<T>::transform(const Matrix<T> &data, size_t n_components) const {
            if (!is_fitted_) {
                throw std::runtime_error("PCA model has not been fitted");
            }

            size_t n_samples = data.rows();
            size_t n_features = data.cols();

            if (n_features != components_.rows()) {
                throw std::invalid_argument("Data has wrong number of features");
            }

            // If n_components is 0 or greater than available components, use all components
            if (n_components == 0 || n_components > components_.cols()) {
                n_components = components_.cols();
            }

            // Center and scale the data
            Matrix<T> processed_data = data;

            // Apply centering and scaling
            for (size_t i = 0; i < n_samples; ++i) {
                for (size_t j = 0; j < n_features; ++j) {
                    processed_data(i, j) = (processed_data(i, j) - mean_[j]) / scale_[j];
                }
            }

            // Project data onto principal components
            // X_transformed = X * V[:, :n_components]
            Matrix<T> components_subset(n_features, n_components);
            for (size_t i = 0; i < n_features; ++i) {
                for (size_t j = 0; j < n_components; ++j) {
                    components_subset(i, j) = components_(i, j);
                }
            }

            return processed_data * components_subset;
        }

        template<typename T>
        Matrix<T> PCA<T>::fit_transform(const Matrix<T> &data, size_t n_components, bool center, bool scale) {
            fit(data, center, scale);
            return transform(data, n_components);
        }

        template<typename T>
        std::vector<T> PCA<T>::explained_variance_ratio() const {
            if (!is_fitted_) {
                throw std::runtime_error("PCA model has not been fitted");
            }
            return explained_variance_ratio_;
        }

        template<typename T>
        Matrix<T> PCA<T>::components() const {
            if (!is_fitted_) {
                throw std::runtime_error("PCA model has not been fitted");
            }
            return components_;
        }

        template<typename T>
        std::vector<T> PCA<T>::singular_values() const {
            if (!is_fitted_) {
                throw std::runtime_error("PCA model has not been fitted");
            }
            return singular_values_;
        }

        // Explicit template instantiations
        template class PCA<float>;
        template class PCA<double>;

    } // namespace utils
} // namespace dl
