#pragma once

#include <memory>
#include <string>
#include <vector>
#include "matrix.hpp"

namespace dl {
    namespace utils {
        // Forward declarations
        using MatrixD = Matrix<double>;

        /**
         * Dataset structure to hold training data
         * TODO: Implement dataset container with:
         * - Features and labels storage
         * - Batch iteration support
         * - Data shuffling
         */
        template<typename T>
        class Dataset {
        public:
            // TODO: Add constructors
            Dataset();

            Dataset(const MatrixD &features, const MatrixD &labels);

            // TODO: Add utility methods
            void add_sample(const std::vector<T> &feature, const std::vector<T> &label);

            size_t size() const;

            std::pair<MatrixD, MatrixD> get_batch(size_t start_idx, size_t batch_size) const;

            void shuffle();

        private:
            MatrixD features_;
            MatrixD labels_;
        };

        /**
         * Data Loader for batch processing
         * TODO: Implement data loader with:
         * - Batch size configuration
         * - Shuffling support
         * - Iterator interface
         */
        template<typename T>
        class DataLoader {
        public:
            DataLoader(const Dataset<T> &dataset, size_t batch_size, bool shuffle = false);

            // Batch iteration methods
            bool has_next() const;

            std::pair<MatrixD, MatrixD> next_batch();

            void reset();

        private:
            const Dataset<T> &dataset_;
            size_t batch_size_;
            bool shuffle_;
            size_t current_idx_;
        };

        /**
         * CSV Data Loader
         * TODO: Implement CSV file reading with:
         * - Header support
         * - Type conversion
         * - Missing value handling
         */
        class CSVLoader {
        public:
            static MatrixD load_csv(const std::string &filename, bool has_header = true, char delimiter = ',');

            static std::pair<MatrixD, MatrixD> load_features_labels(const std::string &filename,
                                                                    const std::vector<size_t> &feature_cols,
                                                                    const std::vector<size_t> &label_cols,
                                                                    bool has_header = true, char delimiter = ',');
        };

        /**
         * Image Data Loader
         * TODO: Implement image loading utilities:
         * - Common image formats support
         * - Preprocessing (resize, normalize)
         * - Augmentation hooks
         */
        class ImageLoader {
        public:
            static MatrixD load_image(const std::string &filename, size_t target_width = 0, size_t target_height = 0);

            static std::vector<MatrixD> load_images_from_directory(const std::string &directory_path,
                                                                   size_t target_width = 0, size_t target_height = 0);
        };

        /**
         * Data Preprocessing Utilities
         * TODO: Implement common preprocessing functions:
         * - Normalization
         * - Standardization
         * - One-hot encoding
         * - Train/validation/test splits
         */
        class Preprocessor {
        public:
            static MatrixD normalize(const MatrixD &data, double min_val = 0.0, double max_val = 1.0);

            static MatrixD standardize(const MatrixD &data);

            static MatrixD one_hot_encode(const std::vector<int> &labels, size_t num_classes);

            static std::tuple<Dataset<double>, Dataset<double>, Dataset<double>>
            train_val_test_split(const Dataset<double> &data, double train_ratio = 0.7, double val_ratio = 0.15);
        };
    } // namespace utils
} // namespace dl
