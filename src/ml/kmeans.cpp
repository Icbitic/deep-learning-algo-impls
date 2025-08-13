#include "ml/kmeans.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace ml {
    template<typename T>
    KMeans<T>::KMeans(size_t k, size_t max_iters, T tol, int random_state) : k_(k),
                                                                             max_iters_(max_iters), tol_(tol),
                                                                             random_state_(random_state), inertia_(0),
                                                                             n_iter_(0),
                                                                             is_fitted_(false) {
        if (k == 0) {
            throw std::invalid_argument("Number of clusters must be greater than 0");
        }
    }

    template<typename T>
    void KMeans<T>::fit(const Tensor<T> &data) {
        if (data.rows() < k_) {
            throw std::invalid_argument("Number of samples must be at least k");
        }

        size_t n_samples = data.rows();
        size_t n_features = data.cols();

        // Initialize centroids
        init_centroids(data);

        std::vector<int> labels(n_samples);
        std::vector<int> prev_labels(n_samples, -1);

        for (n_iter_ = 0; n_iter_ < max_iters_; ++n_iter_) {
            // Assign points to clusters
            labels = assign_clusters(data);

            // Check for convergence
            bool converged = true;
            for (size_t i = 0; i < n_samples; ++i) {
                if (labels[i] != prev_labels[i]) {
                    converged = false;
                    break;
                }
            }

            if (converged) {
                break;
            }

            // Update centroids
            update_centroids(data, labels);
            prev_labels = labels;
        }

        // Calculate final inertia
        inertia_ = 0;
        for (size_t i = 0; i < n_samples; ++i) {
            std::vector<T> point(n_features);
            for (size_t j = 0; j < n_features; ++j) {
                point[j] = data(i, j);
            }
            std::vector<T> centroid(n_features);
            for (size_t j = 0; j < n_features; ++j) {
                centroid[j] = centroids_(labels[i], j);
            }
            inertia_ += squared_distance(point, centroid);
        }

        is_fitted_ = true;
    }

    template<typename T>
    std::vector<int> KMeans<T>::predict(const Tensor<T> &data) const {
        if (!is_fitted_) {
            throw std::runtime_error("Model must be fitted before prediction");
        }
        return assign_clusters(data);
    }

    template<typename T>
    std::vector<int> KMeans<T>::fit_predict(const Tensor<T> &data) {
        fit(data);
        return predict(data);
    }

    template<typename T>
    Tensor<T> KMeans<T>::cluster_centers() const {
        if (!is_fitted_) {
            throw std::runtime_error(
                "Model must be fitted before accessing cluster centers");
        }
        return centroids_;
    }

    template<typename T>
    T KMeans<T>::inertia() const {
        if (!is_fitted_) {
            throw std::runtime_error("Model must be fitted before accessing inertia");
        }
        return inertia_;
    }

    template<typename T>
    size_t KMeans<T>::n_iter() const {
        return n_iter_;
    }

    template<typename T>
    void KMeans<T>::init_centroids(const Tensor<T> &data) {
        size_t n_samples = data.rows();
        size_t n_features = data.cols();

        centroids_ = Tensor<T>::zeros({k_, n_features});

        // Use K-means++ initialization for better results
        std::mt19937 rng(random_state_ >= 0 ? random_state_ : std::random_device{}());
        std::uniform_int_distribution<size_t> dist(0, n_samples - 1);

        // Choose first centroid randomly
        size_t first_idx = dist(rng);
        for (size_t j = 0; j < n_features; ++j) {
            centroids_(0, j) = data(first_idx, j);
        }

        // Choose remaining centroids using K-means++
        for (size_t c = 1; c < k_; ++c) {
            std::vector<T> distances(n_samples);
            T total_distance = 0;

            // Calculate distances to nearest centroid
            for (size_t i = 0; i < n_samples; ++i) {
                T min_dist = std::numeric_limits<T>::max();
                for (size_t prev_c = 0; prev_c < c; ++prev_c) {
                    std::vector<T> point(n_features), centroid(n_features);
                    for (size_t j = 0; j < n_features; ++j) {
                        point[j] = data(i, j);
                        centroid[j] = centroids_(prev_c, j);
                    }
                    T dist = squared_distance(point, centroid);
                    min_dist = std::min(min_dist, dist);
                }
                distances[i] = min_dist;
                total_distance += min_dist;
            }

            // Choose next centroid with probability proportional to squared distance
            std::uniform_real_distribution<T> real_dist(0, total_distance);
            T target = real_dist(rng);
            T cumsum = 0;
            for (size_t i = 0; i < n_samples; ++i) {
                cumsum += distances[i];
                if (cumsum >= target) {
                    for (size_t j = 0; j < n_features; ++j) {
                        centroids_(c, j) = data(i, j);
                    }
                    break;
                }
            }
        }
    }

    template<typename T>
    std::vector<int> KMeans<T>::assign_clusters(const Tensor<T> &data) const {
        size_t n_samples = data.rows();
        size_t n_features = data.cols();
        std::vector<int> labels(n_samples);

        for (size_t i = 0; i < n_samples; ++i) {
            T min_distance = std::numeric_limits<T>::max();
            int best_cluster = 0;

            for (size_t c = 0; c < k_; ++c) {
                std::vector<T> point(n_features), centroid(n_features);
                for (size_t j = 0; j < n_features; ++j) {
                    point[j] = data(i, j);
                    centroid[j] = centroids_(c, j);
                }
                T distance = squared_distance(point, centroid);
                if (distance < min_distance) {
                    min_distance = distance;
                    best_cluster = c;
                }
            }
            labels[i] = best_cluster;
        }

        return labels;
    }

    template<typename T>
    void KMeans<T>::update_centroids(const Tensor<T> &data,
                                     const std::vector<int> &labels) {
        size_t n_samples = data.rows();
        size_t n_features = data.cols();

        // Reset centroids
        centroids_ = Tensor<T>::zeros({k_, n_features});
        std::vector<size_t> counts(k_, 0);

        // Sum points for each cluster
        for (size_t i = 0; i < n_samples; ++i) {
            int cluster = labels[i];
            counts[cluster]++;
            for (size_t j = 0; j < n_features; ++j) {
                centroids_(cluster, j) += data(i, j);
            }
        }

        // Average to get new centroids
        for (size_t c = 0; c < k_; ++c) {
            if (counts[c] > 0) {
                for (size_t j = 0; j < n_features; ++j) {
                    centroids_(c, j) /= counts[c];
                }
            }
        }
    }

    template<typename T>
    T KMeans<T>::squared_distance(const std::vector<T> &point1,
                                  const std::vector<T> &point2) const {
        T distance = 0;
        for (size_t i = 0; i < point1.size(); ++i) {
            T diff = point1[i] - point2[i];
            distance += diff * diff;
        }
        return distance;
    }

    // Explicit template instantiation
    template class KMeans<float>;
    template class KMeans<double>;
} // namespace ml
