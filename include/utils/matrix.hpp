#pragma once

#include <initializer_list>
#include <xtensor/xview.hpp>

/**
 * @file matrix.hpp
 * @brief Matrix utility class for deep learning operations
 * @author Kalenitid
 * @version 1.0.0
 */

namespace dl {
    namespace utils {
        /**
         * @brief A templated matrix class for mathematical operations in deep learning
         *
         * This class provides a comprehensive matrix implementation with support for
         * common mathematical operations required in deep learning algorithms including
         * matrix multiplication, element-wise operations, transpose, and various
         * initialization methods.
         *
         * This implementation uses xtensor as the backend for efficient matrix operations.
         *
         * @tparam T The data type for matrix elements (typically float or double)
         *
         * @example
         * ```cpp
         * // Create a 3x3 identity matrix
         * auto identity = Matrix<float>::identity(3);
         *
         * // Create a random matrix
         * auto random_matrix = Matrix<float>::random(2, 3, 0.0f, 1.0f);
         *
         * // Matrix multiplication
         * auto result = identity * random_matrix;
         * ```
         */
        template<typename T>
        class Matrix {
        public:
            /**
             * @name Constructors
             * @{
             */

            /**
             * @brief Default constructor creating an empty matrix
             */
            Matrix() : rows_(0), cols_(0) {}

            /**
             * @brief Constructor creating a matrix with specified dimensions
             * @param rows Number of rows
             * @param cols Number of columns
             */
            Matrix(size_t rows, size_t cols);


            /**
             * @brief Constructor creating a matrix filled with a specific value
             * @param rows Number of rows
             * @param cols Number of columns
             * @param value Value to fill the matrix with
             */
            Matrix(size_t rows, size_t cols, T value);

            /**
             * @brief Constructor from initializer list
             * @param list Nested initializer list representing matrix data
             */
            Matrix(std::initializer_list<std::initializer_list<T>> list);

            /** @} */

            template<typename U>
            friend std::ostream &operator<<(std::ostream &os, const Matrix<U> &matrix);
            template<typename U>
            friend Matrix<U> dot(const Matrix<U> &a, const Matrix<U> &b);
            template<typename U>
            friend U sum(const Matrix<U> &matrix);
            template<typename U>
            friend U mean(const Matrix<U> &matrix);

            /**
             * @name Element Access
             * @{
             */

            /**
             * @brief Access matrix element at specified position (mutable)
             * @param row Row index (0-based)
             * @param col Column index (0-based)
             * @return Reference to the element
             * @throw std::out_of_range if indices are invalid
             */
            T &operator()(size_t row, size_t col);

            /**
             * @brief Access matrix element at specified position (const)
             * @param row Row index (0-based)
             * @param col Column index (0-based)
             * @return Const reference to the element
             * @throw std::out_of_range if indices are invalid
             */
            const T &operator()(size_t row, size_t col) const;

            /** @} */

            /**
             * @name Arithmetic Operations
             * @{
             */

            /**
             * @brief Matrix addition operator
             * @param other Matrix to add
             * @return Result of matrix addition
             * @throw std::invalid_argument if matrix dimensions don't match
             */
            Matrix operator+(const Matrix &other) const;

            /**
             * @brief Matrix subtraction operator
             * @param other Matrix to subtract
             * @return Result of matrix subtraction
             * @throw std::invalid_argument if matrix dimensions don't match
             */
            Matrix operator-(const Matrix &other) const;

            /**
             * @brief Matrix multiplication operator
             * @param other Matrix to multiply with
             * @return Result of matrix multiplication
             * @throw std::invalid_argument if matrix dimensions are incompatible
             */
            Matrix operator*(const Matrix &other) const;

            /** @} */

            /**
             * @name Matrix Operations
             * @{
             */

            /**
             * @brief Compute the transpose of the matrix
             * @return Transposed matrix
             */
            Matrix transpose() const;

            /**
             * @brief Reshape the matrix to new dimensions
             * @param new_rows New number of rows
             * @param new_cols New number of columns
             * @return Reshaped matrix
             * @throw std::invalid_argument if total size doesn't match
             */
            Matrix reshape(size_t new_rows, size_t new_cols) const;

            /**
             * @brief Calculate the determinant of the matrix
             * @return Determinant value
             * @throw std::invalid_argument if matrix is not square
             */
            T determinant() const;

            /**
             * @brief Calculate the inverse of the matrix
             * @return Inverse matrix
             * @throw std::invalid_argument if matrix is not square or singular
             */
            Matrix inverse() const;

            /** @} */

            /**
             * @name Utility Methods
             * @{
             */

            /**
             * @brief Get the number of rows
             * @return Number of rows
             */
            size_t rows() const { return rows_; }

            /**
             * @brief Get the number of columns
             * @return Number of columns
             */
            size_t cols() const { return cols_; }

            /**
             * @brief Get the total number of elements
             * @return Total size (rows * cols)
             */
            size_t size() const { return rows_ * cols_; }

            /** @} */

            /**
             * @name Static Factory Methods
             * @{
             */

            /**
             * @brief Create a matrix filled with zeros
             * @param rows Number of rows
             * @param cols Number of columns
             * @return Zero-filled matrix
             */
            static Matrix zeros(size_t rows, size_t cols);

            /**
             * @brief Create a matrix filled with ones
             * @param rows Number of rows
             * @param cols Number of columns
             * @return One-filled matrix
             */
            static Matrix ones(size_t rows, size_t cols);

            /**
             * @brief Create an identity matrix
             * @param size Size of the square identity matrix
             * @return Identity matrix
             */
            static Matrix identity(size_t size);

            /**
             * @brief Create a matrix with random values
             * @param rows Number of rows
             * @param cols Number of columns
             * @param min Minimum random value
             * @param max Maximum random value
             * @return Matrix with random values
             */
            static Matrix random(size_t rows, size_t cols, T min, T max);

            /** @} */

            /**
             * @brief Get the underlying xtensor array
             * @return Reference to the xtensor array
             */
            xt::xarray<T> &data() { return data_; }

            /**
             * @brief Get the underlying xtensor array (const)
             * @return Const reference to the xtensor array
             */
            const xt::xarray<T> &data() const { return data_; }

        private:
            /**
             * @brief Internal data storage using xtensor
             */
            xt::xarray<T> data_;

            /**
             * @brief Number of rows and columns
             */
            size_t rows_, cols_;
        };

        /**
         * @name Non-member Functions
         * @{
         */

        /**
         * @brief Output stream operator for matrix visualization
         * @param os Output stream
         * @param matrix Matrix to output
         * @return Reference to the output stream
         */
        template<typename T>
        std::ostream &operator<<(std::ostream &os, const Matrix<T> &matrix);

        /**
         * @brief Compute dot product of two matrices
         * @param a First matrix
         * @param b Second matrix
         * @return Dot product result
         */
        template<typename T>
        Matrix<T> dot(const Matrix<T> &a, const Matrix<T> &b);

        /**
         * @brief Calculate sum of all matrix elements
         * @param matrix Input matrix
         * @return Sum of all elements
         */
        template<typename T>
        T sum(const Matrix<T> &matrix);

        /**
         * @brief Calculate mean of all matrix elements
         * @param matrix Input matrix
         * @return Mean of all elements
         */
        template<typename T>
        T mean(const Matrix<T> &matrix);

        /** @} */

        /**
         * @name Type Aliases
         * @{
         */

        /**
         * @brief Single-precision floating point matrix
         */
        using MatrixF = Matrix<float>;

        /**
         * @brief Double-precision floating point matrix
         */
        using MatrixD = Matrix<double>;

        /** @} */
    } // namespace utils
} // namespace dl
