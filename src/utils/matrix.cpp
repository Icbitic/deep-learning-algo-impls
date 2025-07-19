#include "utils/matrix.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>


namespace dl {
    namespace utils {
        // Matrix Implementation
        template<typename T>
        Matrix<T>::Matrix(size_t rows, size_t cols) : rows_(rows), cols_(cols) {
            // TODO: Implement matrix constructor
            // Initialize data vector with appropriate size
            data_.resize(rows * cols, T{});
        }

        template<typename T>
        Matrix<T>::Matrix(size_t rows, size_t cols, T value) : rows_(rows), cols_(cols) {
            // TODO: Implement matrix constructor with initial value
            data_.resize(rows * cols, value);
        }

        template<typename T>
        T &Matrix<T>::operator()(size_t row, size_t col) {
            // TODO: Implement element access operator
            if (row >= rows_ || col >= cols_) {
                throw std::out_of_range("Matrix index out of bounds");
            }
            return data_[row * cols_ + col];
        }

        template<typename T>
        const T &Matrix<T>::operator()(size_t row, size_t col) const {
            // TODO: Implement const element access operator
            if (row >= rows_ || col >= cols_) {
                throw std::out_of_range("Matrix index out of bounds");
            }
            return data_[row * cols_ + col];
        }

        template<typename T>
        Matrix<T> Matrix<T>::operator+(const Matrix<T> &other) const {
            // TODO: Implement matrix addition
            if (rows_ != other.rows_ || cols_ != other.cols_) {
                throw std::invalid_argument("Matrix dimensions must match for addition");
            }

            Matrix<T> result(rows_, cols_);
            for (size_t i = 0; i < data_.size(); ++i) {
                result.data_[i] = data_[i] + other.data_[i];
            }
            return result;
        }

        template<typename T>
        Matrix<T> Matrix<T>::operator-(const Matrix<T> &other) const {
            // TODO: Implement matrix subtraction
            if (rows_ != other.rows_ || cols_ != other.cols_) {
                throw std::invalid_argument("Matrix dimensions must match for subtraction");
            }

            Matrix<T> result(rows_, cols_);
            for (size_t i = 0; i < data_.size(); ++i) {
                result.data_[i] = data_[i] - other.data_[i];
            }
            return result;
        }

        template<typename T>
        Matrix<T> Matrix<T>::operator*(const Matrix<T> &other) const {
            // TODO: Implement matrix multiplication
            if (cols_ != other.rows_) {
                throw std::invalid_argument("Invalid dimensions for matrix multiplication");
            }

            Matrix<T> result(rows_, other.cols_);
            for (size_t i = 0; i < rows_; ++i) {
                for (size_t j = 0; j < other.cols_; ++j) {
                    T sum = T{};
                    for (size_t k = 0; k < cols_; ++k) {
                        sum += (*this)(i, k) * other(k, j);
                    }
                    result(i, j) = sum;
                }
            }
            return result;
        }

        template<typename T>
        Matrix<T> Matrix<T>::transpose() const {
            // TODO: Implement matrix transpose
            Matrix<T> result(cols_, rows_);
            for (size_t i = 0; i < rows_; ++i) {
                for (size_t j = 0; j < cols_; ++j) {
                    result(j, i) = (*this)(i, j);
                }
            }
            return result;
        }

        template<typename T>
        Matrix<T> Matrix<T>::reshape(size_t new_rows, size_t new_cols) const {
            // TODO: Implement matrix reshape
            if (new_rows * new_cols != rows_ * cols_) {
                throw std::invalid_argument("New dimensions must have same total size");
            }

            Matrix<T> result(new_rows, new_cols);
            result.data_ = data_;
            return result;
        }

        template<typename T>
        T Matrix<T>::determinant() const {
            // TODO: Implement determinant calculation
            if (rows_ != cols_) {
                throw std::invalid_argument("Determinant only defined for square matrices");
            }

            // Placeholder implementation
            return T{};
        }

        template<typename T>
        Matrix<T> Matrix<T>::inverse() const {
            // TODO: Implement matrix inverse
            if (rows_ != cols_) {
                throw std::invalid_argument("Inverse only defined for square matrices");
            }

            // Placeholder implementation
            return Matrix<T>(rows_, cols_);
        }

        template<typename T>
        Matrix<T> Matrix<T>::zeros(size_t rows, size_t cols) {
            // TODO: Implement zeros factory method
            return Matrix<T>(rows, cols, T{});
        }

        template<typename T>
        Matrix<T> Matrix<T>::ones(size_t rows, size_t cols) {
            // TODO: Implement ones factory method
            return Matrix<T>(rows, cols, T{1});
        }

        template<typename T>
        Matrix<T> Matrix<T>::identity(size_t size) {
            // TODO: Implement identity matrix factory method
            Matrix<T> result(size, size);
            for (size_t i = 0; i < size; ++i) {
                result(i, i) = T{1};
            }
            return result;
        }

        template<typename T>
        Matrix<T> Matrix<T>::random(size_t rows, size_t cols, T min_val, T max_val) {
            // TODO: Implement random matrix factory method
            Matrix<T> result(rows, cols);

            std::random_device rd;
            std::mt19937 gen(rd());

            if constexpr (std::is_floating_point_v<T>) {
                std::uniform_real_distribution<T> dis(min_val, max_val);
                for (size_t i = 0; i < result.data_.size(); ++i) {
                    result.data_[i] = dis(gen);
                }
            } else {
                std::uniform_int_distribution<T> dis(min_val, max_val);
                for (size_t i = 0; i < result.data_.size(); ++i) {
                    result.data_[i] = dis(gen);
                }
            }

            return result;
        }

        // Explicit template instantiations
        template class Matrix<float>;
        template class Matrix<double>;
        template class Matrix<int>;
    } // namespace utils
} // namespace dl
