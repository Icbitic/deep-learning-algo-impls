#pragma once

#include <vector>
#include <initializer_list>
#include <iostream>

namespace dl {
    namespace utils {
        /**
         * Matrix class for mathematical operations
         * TODO: Implement basic matrix operations:
         * - Matrix multiplication
         * - Element-wise operations
         * - Transpose
         * - Reshape
         * - Random initialization
         */
        template<typename T>
        class Matrix {
        public:
            // Constructors
            Matrix() : rows_(0), cols_(0) {
            }

            Matrix(size_t rows, size_t cols);

            Matrix(size_t rows, size_t cols, T value);

            // Element access operators
            T &operator()(size_t row, size_t col);

            const T &operator()(size_t row, size_t col) const;

            // Matrix arithmetic operators
            Matrix operator+(const Matrix &other) const;

            Matrix operator-(const Matrix &other) const;

            Matrix operator*(const Matrix &other) const;

            // Matrix operations
            Matrix transpose() const;

            Matrix reshape(size_t new_rows, size_t new_cols) const;

            T determinant() const;

            Matrix inverse() const;

            // Utility methods
            size_t rows() const { return rows_; }
            size_t cols() const { return cols_; }
            size_t size() const { return rows_ * cols_; }

            // Static factory methods
            static Matrix zeros(size_t rows, size_t cols);

            static Matrix ones(size_t rows, size_t cols);

            static Matrix identity(size_t size);

            static Matrix random(size_t rows, size_t cols, T min, T max);

        private:
            std::vector<T> data_;
            size_t rows_, cols_;
        };

        // TODO: Add non-member operators and functions
        // template<typename T>
        // std::ostream& operator<<(std::ostream& os, const Matrix<T>& matrix)

        // TODO: Add specialized mathematical functions
        // template<typename T>
        // Matrix<T> dot(const Matrix<T>& a, const Matrix<T>& b)

        // template<typename T>
        // T sum(const Matrix<T>& matrix)

        // template<typename T>
        // T mean(const Matrix<T>& matrix)

        using MatrixF = Matrix<float>;
        using MatrixD = Matrix<double>;
    } // namespace utils
} // namespace dl
