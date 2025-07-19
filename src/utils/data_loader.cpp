#include "utils/data_loader.hpp"
#include <algorithm>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>


namespace dl {
    namespace utils {
        // Dataset Implementation
        template<typename T>
        Dataset<T>::Dataset() : features_(), labels_() {}

        template<typename T>
        Dataset<T>::Dataset(const MatrixD &features, const MatrixD &labels) : features_(features), labels_(labels) {
            // TODO: Validate that features and labels have compatible dimensions
            if (features.rows() != labels.rows()) {
                throw std::invalid_argument("Features and labels must have same number of samples");
            }
        }

        template<typename T>
        void Dataset<T>::add_sample(const std::vector<T> &feature, const std::vector<T> &label) {
            // TODO: Implement adding individual samples
            // This would require dynamic resizing of the matrices
        }

        template<typename T>
        size_t Dataset<T>::size() const {
            // TODO: Return number of samples
            return features_.rows();
        }

        template<typename T>
        std::pair<MatrixD, MatrixD> Dataset<T>::get_batch(size_t start_idx, size_t batch_size) const {
            // TODO: Implement batch extraction
            // Extract a subset of features and labels
            size_t end_idx = std::min(start_idx + batch_size, size());
            size_t actual_batch_size = end_idx - start_idx;

            // Placeholder implementation - would need proper matrix slicing
            MatrixD batch_features(actual_batch_size, features_.cols());
            MatrixD batch_labels(actual_batch_size, labels_.cols());

            return {batch_features, batch_labels};
        }

        template<typename T>
        void Dataset<T>::shuffle() {
            // TODO: Implement dataset shuffling
            // Shuffle the order of samples while maintaining feature-label correspondence
        }

        // DataLoader Implementation
        template<typename T>
        DataLoader<T>::DataLoader(const Dataset<T> &dataset, size_t batch_size, bool shuffle) :
            dataset_(dataset), batch_size_(batch_size), shuffle_(shuffle), current_idx_(0) {
            // TODO: Initialize data loader
        }

        template<typename T>
        bool DataLoader<T>::has_next() const {
            // TODO: Check if there are more batches
            return current_idx_ < dataset_.size();
        }

        template<typename T>
        std::pair<MatrixD, MatrixD> DataLoader<T>::next_batch() {
            // TODO: Get next batch
            if (!has_next()) {
                throw std::runtime_error("No more batches available");
            }

            auto batch = dataset_.get_batch(current_idx_, batch_size_);
            current_idx_ += batch_size_;

            return batch;
        }

        template<typename T>
        void DataLoader<T>::reset() {
            // TODO: Reset iterator to beginning
            current_idx_ = 0;
            if (shuffle_) {
                // Would shuffle dataset here
            }
        }

        // CSVLoader Implementation
        MatrixD CSVLoader::load_csv(const std::string &filename, bool has_header, char delimiter) {
            // TODO: Implement CSV loading
            std::ifstream file(filename);
            if (!file.is_open()) {
                throw std::runtime_error("Could not open file: " + filename);
            }

            std::vector<std::vector<double>> data;
            std::string line;

            // Skip header if present
            if (has_header && std::getline(file, line)) {
                // Header skipped
            }

            while (std::getline(file, line)) {
                std::vector<double> row;
                std::stringstream ss(line);
                std::string cell;

                while (std::getline(ss, cell, delimiter)) {
                    try {
                        row.push_back(std::stod(cell));
                    } catch (const std::exception &) {
                        // Handle parsing errors
                        row.push_back(0.0);
                    }
                }

                if (!row.empty()) {
                    data.push_back(row);
                }
            }

            if (data.empty()) {
                return MatrixD(0, 0);
            }

            // Convert to matrix
            size_t rows = data.size();
            size_t cols = data[0].size();
            MatrixD result(rows, cols);

            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols && j < data[i].size(); ++j) {
                    result(i, j) = data[i][j];
                }
            }

            return result;
        }

        std::pair<MatrixD, MatrixD> CSVLoader::load_features_labels(const std::string &filename,
                                                                    const std::vector<size_t> &feature_cols,
                                                                    const std::vector<size_t> &label_cols,
                                                                    bool has_header, char delimiter) {
            // TODO: Implement feature-label separation
            MatrixD full_data = load_csv(filename, has_header, delimiter);

            // Extract features and labels based on column indices
            MatrixD features(full_data.rows(), feature_cols.size());
            MatrixD labels(full_data.rows(), label_cols.size());

            // Placeholder implementation - would need proper column extraction

            return {features, labels};
        }

        // ImageLoader Implementation
        MatrixD ImageLoader::load_image(const std::string &filename, size_t target_width, size_t target_height) {
            // TODO: Implement image loading
            // This would typically use a library like OpenCV or STBI
            // For now, return a placeholder matrix
            return MatrixD(target_height, target_width);
        }

        std::vector<MatrixD> ImageLoader::load_images_from_directory(const std::string &directory_path,
                                                                     size_t target_width, size_t target_height) {
            // TODO: Implement batch image loading from directory
            // This would scan the directory and load all image files
            std::vector<MatrixD> images;
            return images;
        }

        // Preprocessor Implementation
        MatrixD Preprocessor::normalize(const MatrixD &data, double min_val, double max_val) {
            // TODO: Implement normalization
            // Scale data to [min_val, max_val] range
            MatrixD result(data.rows(), data.cols());

            // Find min and max values in data
            // Apply normalization formula: (x - data_min) / (data_max - data_min) * (max_val - min_val) + min_val

            return result;
        }

        MatrixD Preprocessor::standardize(const MatrixD &data) {
            // TODO: Implement standardization (z-score normalization)
            // Formula: (x - mean) / std_dev
            MatrixD result(data.rows(), data.cols());

            // Calculate mean and standard deviation for each feature
            // Apply standardization formula

            return result;
        }

        MatrixD Preprocessor::one_hot_encode(const std::vector<int> &labels, size_t num_classes) {
            // TODO: Implement one-hot encoding
            MatrixD result(labels.size(), num_classes);

            for (size_t i = 0; i < labels.size(); ++i) {
                if (labels[i] >= 0 && static_cast<size_t>(labels[i]) < num_classes) {
                    result(i, labels[i]) = 1.0;
                }
            }

            return result;
        }

        // Removed train_test_split method - using train_val_test_split instead

        std::tuple<Dataset<double>, Dataset<double>, Dataset<double>>
        Preprocessor::train_val_test_split(const Dataset<double> &data, double train_ratio, double val_ratio) {
            // TODO: Implement train-validation-test split
            // For now, return empty datasets
            Dataset<double> train_set;
            Dataset<double> val_set;
            Dataset<double> test_set;

            return {train_set, val_set, test_set};
        }

        // Explicit template instantiations
        template class Dataset<double>;
        template class Dataset<float>;
        template class DataLoader<double>;
        template class DataLoader<float>;
    } // namespace utils
} // namespace dl
