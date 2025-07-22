#pragma once

#include <memory>
#include <string>
#include <vector>
#include "matrix.hpp"

/**
 * @file data_loader.hpp
 * @brief Data loading and preprocessing utilities for deep learning
 * @author Kalenitid
 * @version 1.0.0
 *
 * This file provides comprehensive data handling capabilities including:
 * - Dataset containers for training data
 * - Batch processing and iteration
 * - CSV and image data loading
 * - Data preprocessing and augmentation utilities
 */

namespace dl {
    namespace utils {
        // Forward declarations
        using MatrixD = Matrix<double>;

        /**
         * @brief Dataset container for machine learning data
         *
         * A flexible container class that holds features and labels for training,
         * validation, or testing. Supports batch processing, data shuffling,
         * and efficient memory management.
         *
         * @tparam T Data type for the dataset elements (typically double or float)
         *
         * @example
         * ```cpp
         * // Create dataset from matrices
         * MatrixD features(100, 784);  // 100 samples, 784 features
         * MatrixD labels(100, 10);     // 100 samples, 10 classes
         * Dataset<double> dataset(features, labels);
         *
         * // Add individual samples
         * std::vector<double> sample_features = {1.0, 2.0, 3.0};
         * std::vector<double> sample_label = {0.0, 1.0, 0.0};
         * dataset.add_sample(sample_features, sample_label);
         *
         * // Get batches for training
         * auto batch = dataset.get_batch(0, 32);  // First 32 samples
         * ```
         */
        template<typename T>
        class Dataset {
        public:
            /**
             * @brief Default constructor for empty dataset
             */
            Dataset();

            /**
             * @brief Constructor with feature and label matrices
             * @param features Matrix containing input features (samples x features)
             * @param labels Matrix containing target labels (samples x outputs)
             *
             * @note Features and labels must have the same number of rows (samples)
             */
            Dataset(const MatrixD &features, const MatrixD &labels);

            /**
             * @brief Add a single sample to the dataset
             * @param feature Feature vector for the sample
             * @param label Label vector for the sample
             *
             * @note This method dynamically grows the dataset
             */
            void add_sample(const std::vector<T> &feature, const std::vector<T> &label);

            /**
             * @brief Get the number of samples in the dataset
             * @return Number of samples
             */
            size_t size() const;

            /**
             * @brief Extract a batch of samples from the dataset
             * @param start_idx Starting index for the batch
             * @param batch_size Number of samples in the batch
             * @return Pair of (features, labels) matrices for the batch
             *
             * @note If start_idx + batch_size exceeds dataset size,
             *       returns samples from start_idx to the end
             */
            std::pair<MatrixD, MatrixD> get_batch(size_t start_idx, size_t batch_size) const;

            /**
             * @brief Randomly shuffle the dataset samples
             *
             * @note Maintains correspondence between features and labels
             */
            void shuffle();

        private:
            MatrixD features_; ///< Feature matrix (samples x features)
            MatrixD labels_; ///< Label matrix (samples x outputs)
        };

        /**
         * @brief Data loader for efficient batch processing
         *
         * Provides an iterator-like interface for processing datasets in batches.
         * Supports configurable batch sizes, data shuffling, and automatic
         * epoch management.
         *
         * @tparam T Data type for the dataset elements
         *
         * @example
         * ```cpp
         * Dataset<double> dataset(features, labels);
         * DataLoader<double> loader(dataset, batch_size=32, shuffle=true);
         *
         * // Training loop
         * while (loader.has_next()) {
         *     auto [batch_features, batch_labels] = loader.next_batch();
         *     // Train on batch...
         * }
         * loader.reset();  // Start next epoch
         * ```
         */
        template<typename T>
        class DataLoader {
        public:
            /**
             * @brief Constructor for data loader
             * @param dataset Reference to the dataset to iterate over
             * @param batch_size Number of samples per batch
             * @param shuffle Whether to shuffle data at the start of each epoch
             */
            DataLoader(const Dataset<T> &dataset, size_t batch_size, bool shuffle = false);

            /**
             * @brief Check if more batches are available in current epoch
             * @return True if more batches exist, false otherwise
             */
            bool has_next() const;

            /**
             * @brief Get the next batch of data
             * @return Pair of (features, labels) matrices for the next batch
             *
             * @note Advances the internal iterator. Call has_next() first to check availability
             */
            std::pair<MatrixD, MatrixD> next_batch();

            /**
             * @brief Reset iterator to start of dataset
             *
             * @note If shuffle is enabled, reshuffles the dataset
             */
            void reset();

        private:
            const Dataset<T> &dataset_; ///< Reference to the dataset
            size_t batch_size_; ///< Size of each batch
            bool shuffle_; ///< Whether to shuffle data
            size_t current_idx_; ///< Current position in dataset
        };

        /**
         * @brief CSV file data loader utility
         *
         * Provides static methods for loading data from CSV files with support
         * for headers, custom delimiters, and automatic type conversion.
         *
         * @example
         * ```cpp
         * // Load entire CSV as matrix
         * auto data = CSVLoader::load_csv("data.csv", has_header=true);
         *
         * // Load specific columns as features and labels
         * std::vector<size_t> feature_cols = {0, 1, 2, 3};
         * std::vector<size_t> label_cols = {4};
         * auto [features, labels] = CSVLoader::load_features_labels(
         *     "iris.csv", feature_cols, label_cols);
         * ```
         */
        class CSVLoader {
        public:
            /**
             * @brief Load CSV file into a matrix
             * @param filename Path to the CSV file
             * @param has_header Whether the file contains a header row
             * @param delimiter Character used to separate values
             * @return Matrix containing the loaded data
             *
             * @note Missing values are handled by setting them to 0.0
             */
            static MatrixD load_csv(const std::string &filename, bool has_header = true, char delimiter = ',');

            /**
             * @brief Load specific columns as features and labels
             * @param filename Path to the CSV file
             * @param feature_cols Column indices to use as features
             * @param label_cols Column indices to use as labels
             * @param has_header Whether the file contains a header row
             * @param delimiter Character used to separate values
             * @return Pair of (features, labels) matrices
             */
            static std::pair<MatrixD, MatrixD> load_features_labels(const std::string &filename,
                                                                    const std::vector<size_t> &feature_cols,
                                                                    const std::vector<size_t> &label_cols,
                                                                    bool has_header = true, char delimiter = ',');
        };

        /**
         * @brief Image data loader utility
         *
         * Provides static methods for loading and preprocessing image data
         * with support for common formats and automatic resizing.
         *
         * @example
         * ```cpp
         * // Load single image with resizing
         * auto image = ImageLoader::load_image("photo.jpg", 224, 224);
         *
         * // Load all images from directory
         * auto images = ImageLoader::load_images_from_directory(
         *     "./dataset/", 128, 128);
         * ```
         *
         * @note Supports common formats: JPEG, PNG, BMP, TIFF
         */
        class ImageLoader {
        public:
            /**
             * @brief Load a single image file
             * @param filename Path to the image file
             * @param target_width Target width for resizing (0 = no resize)
             * @param target_height Target height for resizing (0 = no resize)
             * @return Matrix containing normalized pixel values [0,1]
             *
             * @note Images are converted to grayscale and normalized
             */
            static MatrixD load_image(const std::string &filename, size_t target_width = 0, size_t target_height = 0);

            /**
             * @brief Load all images from a directory
             * @param directory_path Path to directory containing images
             * @param target_width Target width for resizing (0 = no resize)
             * @param target_height Target height for resizing (0 = no resize)
             * @return Vector of matrices, each containing an image
             *
             * @note Processes all supported image files in the directory
             */
            static std::vector<MatrixD> load_images_from_directory(const std::string &directory_path,
                                                                   size_t target_width = 0, size_t target_height = 0);
        };

        /**
         * @brief Data preprocessing utilities
         *
         * Collection of static methods for common data preprocessing tasks
         * including normalization, standardization, encoding, and dataset splitting.
         *
         * @example
         * ```cpp
         * // Normalize data to [0,1] range
         * auto normalized = Preprocessor::normalize(raw_data);
         *
         * // Standardize to zero mean, unit variance
         * auto standardized = Preprocessor::standardize(raw_data);
         *
         * // One-hot encode categorical labels
         * std::vector<int> labels = {0, 1, 2, 1, 0};
         * auto encoded = Preprocessor::one_hot_encode(labels, 3);
         *
         * // Split dataset for training
         * auto [train, val, test] = Preprocessor::train_val_test_split(
         *     dataset, train_ratio=0.7, val_ratio=0.15);
         * ```
         */
        class Preprocessor {
        public:
            /**
             * @brief Normalize data to specified range
             * @param data Input data matrix
             * @param min_val Minimum value of output range
             * @param max_val Maximum value of output range
             * @return Normalized data matrix
             *
             * @note Uses min-max normalization: (x - min) / (max - min) * (max_val - min_val) + min_val
             */
            static MatrixD normalize(const MatrixD &data, double min_val = 0.0, double max_val = 1.0);

            /**
             * @brief Standardize data to zero mean and unit variance
             * @param data Input data matrix
             * @return Standardized data matrix
             *
             * @note Uses z-score normalization: (x - mean) / std_dev
             */
            static MatrixD standardize(const MatrixD &data);

            /**
             * @brief Convert categorical labels to one-hot encoding
             * @param labels Vector of integer class labels
             * @param num_classes Total number of classes
             * @return Matrix with one-hot encoded labels
             *
             * @note Each row represents one sample, columns represent classes
             */
            static MatrixD one_hot_encode(const std::vector<int> &labels, size_t num_classes);

            /**
             * @brief Split dataset into training, validation, and test sets
             * @param data Input dataset to split
             * @param train_ratio Fraction of data for training (default: 0.7)
             * @param val_ratio Fraction of data for validation (default: 0.15)
             * @return Tuple of (training, validation, test) datasets
             *
             * @note Test ratio is automatically calculated as 1 - train_ratio - val_ratio
             */
            static std::tuple<Dataset<double>, Dataset<double>, Dataset<double>>
            train_val_test_split(const Dataset<double> &data, double train_ratio = 0.7, double val_ratio = 0.15);
        };
    } // namespace utils
} // namespace dl
