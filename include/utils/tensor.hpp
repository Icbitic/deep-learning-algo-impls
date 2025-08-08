#pragma once

#include <initializer_list>
#include <iostream>
#include <tuple>
#include <vector>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/generators/xrandom.hpp>
#include <xtensor/io/xio.hpp>
#include <xtensor/reducers/xreducer.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor/core/xshape.hpp>

/**
 * @file matrix.hpp
 * @brief Tensor utility class for deep learning operations (n-dimensional arrays)
 * @author Kalenitid
 * @version 2.0.0
 */

namespace utils {
    /**
     * @brief A templated tensor class for mathematical operations in deep learning
     *
     * This class provides a comprehensive n-dimensional tensor implementation with support for
     * common mathematical operations required in deep learning algorithms including
     * tensor operations, element-wise operations, reshaping, and various
     * initialization methods. It maintains backward compatibility with 2D matrix operations.
     *
     * This implementation uses xtensor as the backend for efficient tensor operations.
     *
     * @tparam T The data type for tensor elements (typically float or double)
     *
     * @example
     * ```cpp
     * // Create a 3x3 identity matrix (2D tensor)
     * auto identity = Tensor<float>::identity(3);
     *
     * // Create a random 3D tensor
     * auto random_tensor = Tensor<float>::random({2, 3, 4}, 0.0f, 1.0f);
     *
     * // Tensor operations
     * auto result = identity.matmul(random_tensor.view({3, 8}));
     * ```
     */
    template<typename T>
    class Tensor {
    public:
        /**
         * @name Constructors
         * @{
         */

        /**
         * @brief Default constructor creating an empty tensor
         */
        Tensor() : data_(xt::xarray<T>::from_shape({0})), rows_(0), cols_(0) {}

        /**
         * @brief Constructor creating a tensor with specified shape
         * @param shape Shape of the tensor
         */
        explicit Tensor(const std::vector<size_t>& shape);

        /**
         * @brief Constructor creating a 2D tensor (matrix) with specified dimensions
         * @param rows Number of rows
         * @param cols Number of columns
         */
        Tensor(size_t rows, size_t cols);

        /**
         * @brief Constructor creating a tensor filled with a specific value
         * @param shape Shape of the tensor
         * @param value Value to fill the tensor with
         */
        Tensor(const std::vector<size_t>& shape, T value);

        /**
         * @brief Constructor creating a 2D tensor (matrix) filled with a specific value
         * @param rows Number of rows
         * @param cols Number of columns
         * @param value Value to fill the tensor with
         */
        Tensor(size_t rows, size_t cols, T value);

        /**
         * @brief Constructor from initializer list (2D matrix)
         * @param list Nested initializer list representing matrix data
         */
        Tensor(std::initializer_list<std::initializer_list<T>> list);

        /**
         * @brief Constructor from xtensor array
         * @param data xtensor array data
         */
        explicit Tensor(const xt::xarray<T>& data);

        /** @} */

        template<typename U>
        friend std::ostream &operator<<(std::ostream &os, const Tensor<U> &tensor);
        template<typename U>
        friend Tensor<U> dot(const Tensor<U> &a, const Tensor<U> &b);
        template<typename U>
        friend U sum(const Tensor<U> &tensor);
        template<typename U>
        friend U mean(const Tensor<U> &tensor);

        /**
         * @name Element Access
         * @{
         */

        /**
         * @brief Access tensor element at specified position (mutable)
         * @param indices Variable number of indices for n-dimensional access
         * @return Reference to the element
         * @throw std::out_of_range if indices are invalid
         */
        template<typename... Args>
        T &operator()(Args... indices);

        /**
         * @brief Access tensor element at specified position (const)
         * @param indices Variable number of indices for n-dimensional access
         * @return Const reference to the element
         * @throw std::out_of_range if indices are invalid
         */
        template<typename... Args>
        const T &operator()(Args... indices) const;

        /**
         * @brief Access 2D tensor element at specified position (mutable) - backward compatibility
         * @param row Row index (0-based)
         * @param col Column index (0-based)
         * @return Reference to the element
         * @throw std::out_of_range if indices are invalid
         */
        T &at(size_t row, size_t col);

        /**
         * @brief Access 2D tensor element at specified position (const) - backward compatibility
         * @param row Row index (0-based)
         * @param col Column index (0-based)
         * @return Const reference to the element
         * @throw std::out_of_range if indices are invalid
         */
        const T &at(size_t row, size_t col) const;

        /** @} */

        /**
         * @name Arithmetic Operations
         * @{
         */

        /**
         * @brief Tensor element-wise addition operator
         * @param other Tensor to add
         * @return Result of tensor addition
         * @throw std::invalid_argument if tensor shapes don't match
         */
        Tensor operator+(const Tensor &other) const;

        /**
         * @brief Tensor element-wise subtraction operator
         * @param other Tensor to subtract
         * @return Result of tensor subtraction
         * @throw std::invalid_argument if tensor shapes don't match
         */
        Tensor operator-(const Tensor &other) const;

        /**
         * @brief Tensor element-wise multiplication operator
         * @param other Tensor to multiply with
         * @return Result of element-wise tensor multiplication
         * @throw std::invalid_argument if tensor shapes don't match
         */
        Tensor operator*(const Tensor &other) const;

        /**
         * @brief Matrix multiplication operator (for 2D tensors)
         * @param other Tensor to multiply with
         * @return Result of matrix multiplication
         * @throw std::invalid_argument if tensor dimensions are incompatible for matrix multiplication
         */
        Tensor matmul(const Tensor &other) const;

        /**
         * @brief Scalar multiplication operator
         * @param scalar Scalar value to multiply with
         * @return Result of scalar multiplication
         */
        Tensor operator*(T scalar) const;

        /** @} */

        /**
         * @name Tensor Operations
         * @{
         */

        /**
         * @brief Compute the transpose of the tensor (for 2D tensors)
         * @return Transposed tensor
         * @throw std::invalid_argument if tensor is not 2D
         */
        Tensor transpose() const;

        /**
         * @brief Transpose along specified axes
         * @param axes Axes to transpose
         * @return Transposed tensor
         */
        Tensor transpose(const std::vector<size_t>& axes) const;

        /**
         * @brief Reshape the tensor to new dimensions
         * @param new_shape New shape for the tensor
         * @return Reshaped tensor
         * @throw std::invalid_argument if total size doesn't match
         */
        Tensor reshape(const std::vector<size_t>& new_shape) const;

        /**
         * @brief Reshape the tensor to new 2D dimensions (backward compatibility)
         * @param new_rows New number of rows
         * @param new_cols New number of columns
         * @return Reshaped tensor
         * @throw std::invalid_argument if total size doesn't match
         */
        Tensor reshape(size_t new_rows, size_t new_cols) const;

        /**
         * @brief Create a view of the tensor with new shape
         * @param new_shape New shape for the view
         * @return Tensor view with new shape
         * @throw std::invalid_argument if total size doesn't match
         */
        Tensor view(const std::vector<size_t>& new_shape) const;

        /**
         * @brief Squeeze dimensions of size 1
         * @param axis Optional axis to squeeze (if -1, squeeze all dimensions of size 1)
         * @return Squeezed tensor
         */
        Tensor squeeze(int axis = -1) const;

        /**
         * @brief Add a dimension of size 1
         * @param axis Axis where to add the dimension
         * @return Tensor with added dimension
         */
        Tensor unsqueeze(size_t axis) const;

        /**
         * @brief Calculate the determinant of the matrix (for 2D square tensors)
         * @return Determinant value
         * @throw std::invalid_argument if tensor is not 2D square
         */
        T determinant() const;

        /**
         * @brief Calculate the inverse of the matrix (for 2D square tensors)
         * @return Inverse tensor
         * @throw std::invalid_argument if tensor is not 2D square or singular
         */
        Tensor inverse() const;

        /**
         * @brief Calculate eigenvalues of the matrix (for 2D square tensors)
         * @return Eigenvalues
         * @throw std::invalid_argument if tensor is not 2D square
         */
        auto eigenvalues() const;

        /** @} */

        /**
         * @name Utility Methods
         * @{
         */

        /**
         * @brief Get the number of rows
         * @return Number of rows
         */
        [[nodiscard]] size_t rows() const { return rows_; }

        /**
         * @brief Get the number of columns
         * @return Number of columns
         */
        [[nodiscard]] size_t cols() const { return cols_; }

        /**
         * @brief Get the total number of elements
         * @return Total size (rows * cols)
         */
        [[nodiscard]] size_t size() const { return rows_ * cols_; }

        /**
         * @brief Get the shape of the matric in one step
         * @return Shape (rows, cols) in tuple
         */
        [[nodiscard]] std::tuple<size_t, size_t> shape() const { return {rows_, cols_}; }

        /** @} */

        /**
         * @name Static Factory Methods
         * @{
         */

        /**
         * @brief Create a zero tensor
         * @param shape Shape of the tensor
         * @return Zero tensor
         */
        static Tensor zeros(const std::vector<size_t>& shape);

        /**
         * @brief Create a zero matrix (backward compatibility)
         * @param rows Number of rows
         * @param cols Number of columns
         * @return Zero tensor with 2D shape
         */
        static Tensor zeros(size_t rows, size_t cols);

        /**
         * @brief Create a tensor filled with ones
         * @param shape Shape of the tensor
         * @return Tensor filled with ones
         */
        static Tensor ones(const std::vector<size_t>& shape);

        /**
         * @brief Create a matrix filled with ones (backward compatibility)
         * @param rows Number of rows
         * @param cols Number of columns
         * @return Tensor filled with ones with 2D shape
         */
        static Tensor ones(size_t rows, size_t cols);

        /**
         * @brief Create a tensor filled with a specific value
         * @param shape Shape of the tensor
         * @param value Value to fill the tensor with
         * @return Tensor filled with the specified value
         */
        static Tensor full(const std::vector<size_t>& shape, T value);

        /**
         * @brief Create an identity matrix (2D tensor)
         * @param size Size of the square identity matrix
         * @return Identity tensor
         */
        static Tensor identity(size_t size);

        /**
         * @brief Create a random tensor with values between 0 and 1
         * @param shape Shape of the tensor
         * @return Random tensor
         */
        static Tensor random(const std::vector<size_t>& shape);

        /**
         * @brief Create a random tensor with values between min and max
         * @param shape Shape of the tensor
         * @param min Minimum random value
         * @param max Maximum random value
         * @return Random tensor
         */
        static Tensor random(const std::vector<size_t>& shape, T min, T max);

        /**
         * @brief Create a matrix with random values between 0 and 1 (backward compatibility)
         * @param rows Number of rows
         * @param cols Number of columns
         * @return Random tensor with 2D shape
         */
        static Tensor random(size_t rows, size_t cols);

        /**
         * @brief Create a matrix with random values (backward compatibility)
         * @param rows Number of rows
         * @param cols Number of columns
         * @param min Minimum random value
         * @param max Maximum random value
         * @return Random tensor with 2D shape
         */
        static Tensor random(size_t rows, size_t cols, T min, T max);

        /**
         * @brief Create a tensor from an existing xt::xarray
         * @param array The xt::xarray to wrap
         * @return Tensor wrapping the array
         */
        static Tensor from_array(const xt::xarray<T>& array);

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

    // Template method implementations
    template<typename T>
    template<typename... Args>
    T& Tensor<T>::operator()(Args... indices) {
        return data_(indices...);
    }

    template<typename T>
    template<typename... Args>
    const T& Tensor<T>::operator()(Args... indices) const {
        return data_(indices...);
    }

    /**
     * @name Non-member Functions
     * @{
     */

    /**
     * @brief Output stream operator for tensor visualization
     * @param os Output stream
     * @param tensor Tensor to output
     * @return Reference to the output stream
     */
    template<typename T>
    std::ostream &operator<<(std::ostream &os, const Tensor<T> &tensor);

    /**
     * @brief Compute dot product of two tensors
     * @param a First tensor
     * @param b Second tensor
     * @return Dot product result
     */
    template<typename T>
    Tensor<T> dot(const Tensor<T> &a, const Tensor<T> &b);

    /**
     * @brief Calculate sum of all tensor elements
     * @param tensor Input tensor
     * @return Sum of all elements
     */
    template<typename T>
    T sum(const Tensor<T> &tensor);

    /**
     * @brief Calculate mean of all tensor elements
     * @param tensor Input tensor
     * @return Mean of all elements
     */
    template<typename T>
    T mean(const Tensor<T> &tensor);

    /** @} */

    /**
     * @name Type Aliases
     * @{
     */

    /**
     * @brief Single-precision floating point tensor
     */
    using TensorF = Tensor<float>;

    /**
     * @brief Double-precision floating point tensor
     */
    using TensorD = Tensor<double>;

    /**
     * @brief Single-precision floating point matrix (backward compatibility)
     */
    using MatrixF = Tensor<float>;

    /**
     * @brief Double-precision floating point matrix (backward compatibility)
     */
    using MatrixD = Tensor<double>;

    /** @} */
} // namespace utils
