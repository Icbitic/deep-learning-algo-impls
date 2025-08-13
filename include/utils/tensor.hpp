#pragma once

#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <vector>
#include <xtensor/containers/xarray.hpp>

/**
 * @file tensor.hpp
 * @brief Tensor utility class for deep learning operations (n-dimensional arrays)
 * @author Kalenitid
 * @version 2.0.0
 */

namespace utils {
    // Forward declaration
    template<typename T>
    class Variable;

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
        Tensor() : shape_() {
            data_.resize({0});
        }

        /**
         * @brief Constructor for tensor from data with shape (PyTorch-style)
         * @param data Vector of data elements
         * @param shape Vector specifying the dimensions of the tensor
         * @example
         *       Tensor<float> t({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}); // Creates a 2x2 tensor
         */
        Tensor(const std::vector<T> &data, const std::vector<size_t> &shape);

        /**
         * @brief Constructor from initializer list with shape specification
         * @param data Initializer list containing the tensor data
         * @param shape Shape of the tensor
         * @note This constructor allows creating tensors with specified data and shape:
         *       Tensor<double> t({1.0, 1.1, 1.2, 5.0, 5.1, 5.2}, {6});
         *       Tensor<double> t({1.0, 2.0, 3.0, 4.0}, {2, 2});
         *       Tensor<float> t({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, {2, 2, 2});
         */
        Tensor(std::initializer_list<T> data, const std::vector<size_t> &shape);

        /**
         * @brief Constructor from xtensor array
         * @param data xtensor array data
         */
        explicit Tensor(const xt::xarray<T> &data);

        /**
         * @brief Constructor for scalar tensor (0-dimensional)
         * @param value Scalar value to create a 0-dimensional tensor
         * @example
         *       Tensor<double> scalar(42.0); // Creates a 0-dimensional tensor with value 42.0
         */
        explicit Tensor(T value);

        /** @} */

        template<typename U>
        friend std::ostream &operator<<(std::ostream &os, const Tensor<U> &tensor);

        template<typename U>
        friend Tensor<U> dot(const Tensor<U> &a, const Tensor<U> &b);

        template<typename U>
        friend U sum(const Tensor<U> &tensor);

        template<typename U>
        friend U mean(const Tensor<U> &tensor);

        template<typename U, typename V>
        friend Tensor<U> cast_tensor(const Tensor<V> &tensor);

        template<typename U>
        friend Tensor<bool> compare_greater(const Tensor<U> &lhs, const Tensor<U> &rhs);

        template<typename U>
        friend class Variable;

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
         * @brief Access scalar value from 0-dimensional tensor
         * @return Const reference to the scalar value
         * @throw std::invalid_argument if tensor is not 0-dimensional
         */
        const T &scalar() const;


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
         * @brief Element-wise division
         * @param other Tensor to divide by
         * @return New tensor with element-wise division result
         * @note This operation creates a new tensor and does not modify the original
         */
        Tensor operator/(const Tensor &other) const;

        /**
         * @brief Unary minus operator
         * @return New tensor with negated values
         */
        Tensor operator-() const;

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

        /**
         * @brief Element-wise comparison (greater than)
         * @param other Tensor to compare with
         * @return New tensor with boolean results
         */
        Tensor<bool> operator>(const Tensor &other) const;

        /**
         * @brief Element-wise maximum with another tensor
         * @param other Tensor to compare with
         * @return New tensor with element-wise maximum values
         */
        Tensor max(const Tensor &other) const;

        /**
         * @brief Element-wise exponential function
         * @return New tensor with element-wise exponential values
         */
        Tensor exp() const;

        /**
         * @brief Element-wise logarithm function
         * @return New tensor with element-wise logarithm values
         */
        Tensor log() const;

        /**
         * @brief Element-wise sigmoid function
         * @return New tensor with element-wise sigmoid values
         */
        Tensor sigmoid() const;

        /**
         * @brief Element-wise tanh function
         * @return New tensor with element-wise tanh values
         */
        Tensor tanh() const;

        /**
         * @brief Element-wise ReLU function
         * @return New tensor with element-wise ReLU values
         */
        Tensor relu() const;

        /**
         * @brief Sum all elements in the tensor
         * @return New tensor with sum of all elements
         */
        Tensor sum() const;

        /**
         * @brief Sum along specified axes
         * @param axes Axes to sum along
         * @param keepdims Whether to keep dimensions
         * @return New tensor with sum along specified axes
         */
        Tensor sum(const std::vector<int> &axes, bool keepdims = false) const;

        /**
         * @brief Mean of all elements in the tensor
         * @return New tensor with mean of all elements
         */
        Tensor mean() const;

        /**
         * @brief Mean along specified axes
         * @param axes Axes to compute mean along
         * @param keepdims Whether to keep dimensions
         * @return New tensor with mean along specified axes
         */
        Tensor mean(const std::vector<int> &axes, bool keepdims = false) const;

        /**
         * @brief Check if tensor is empty
         * @return True if tensor has no elements
         */
        bool empty() const;

        /**
         * @brief Clear the tensor, making it empty
         */
        void clear();

        /**
         * @brief Cast tensor to different type
         * @return New tensor with casted values
         */
        template<typename U>
        Tensor<U> cast() const;

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
        Tensor transpose(const std::vector<size_t> &axes) const;

        /**
         * @brief Reshape the tensor to new dimensions
         * @param new_shape New shape for the tensor
         * @return Reshaped tensor
         * @throw std::invalid_argument if total size doesn't match
         */
        Tensor reshape(const std::vector<size_t> &new_shape) const;


        /**
         * @brief Create a view of the tensor with new shape
         * @param new_shape New shape for the view
         * @return Tensor view with new shape
         * @throw std::invalid_argument if total size doesn't match
         */
        Tensor view(const std::vector<size_t> &new_shape) const;

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
         * @brief Get the number of rows (for 2D tensors)
         * @return Number of rows
         */
        [[nodiscard]] size_t rows() const {
            return shape_.size() > 0 ? shape_[0] : 0;
        }

        /**
         * @brief Get the number of columns (for 2D tensors)
         * @return Number of columns
         */
        [[nodiscard]] size_t cols() const {
            return shape_.size() > 1 ? shape_[1] : (shape_.size() == 1 ? 1 : 0);
        }

        /**
         * @brief Get the total number of elements
         * @return Total size of the tensor
         */
        [[nodiscard]] size_t size() const { return data_.size(); }

        /**
         * @brief Get the number of dimensions
         * @return Number of dimensions
         */
        [[nodiscard]] size_t ndim() const { return shape_.size(); }

        /**
         * @brief Get the shape of the tensor
         * @return Shape vector
         */
        [[nodiscard]] const std::vector<size_t> &shape() const { return shape_; }

        /**
         * @brief Get the shape of the tensor as tuple (for 2D compatibility)
         * @return Shape (rows, cols) in tuple
         */
        [[nodiscard]] std::tuple<size_t, size_t> shape2d() const {
            return {rows(), cols()};
        }

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
        static Tensor zeros(const std::vector<size_t> &shape);


        /**
         * @brief Create a tensor filled with ones
         * @param shape Shape of the tensor
         * @return Tensor filled with ones
         */
        static Tensor ones(const std::vector<size_t> &shape);


        /**
         * @brief Create a tensor filled with a specific value
         * @param shape Shape of the tensor
         * @param value Value to fill the tensor with
         * @return Tensor filled with the specified value
         */
        static Tensor full(const std::vector<size_t> &shape, T value);

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
        static Tensor random(const std::vector<size_t> &shape);

        /**
         * @brief Create a random tensor with values between min and max
         * @param shape Shape of the tensor
         * @param min Minimum random value
         * @param max Maximum random value
         * @return Random tensor
         */
        static Tensor random(const std::vector<size_t> &shape, T min, T max);


        /**
         * @brief Create a tensor from an existing xt::xarray
         * @param array The xt::xarray to wrap
         * @return Tensor wrapping the array
         */
        static Tensor from_array(const xt::xarray<T> &array) {
            return Tensor(array);
        }

        /**
         * @brief Create a tensor of ones with the same shape as input
         * @param tensor Input tensor to match shape
         * @return New tensor filled with ones
         */
        static Tensor ones_like(const Tensor &tensor);

        /**
         * @brief Create a tensor of zeros with the same shape as input
         * @param tensor Input tensor to match shape
         * @return New tensor filled with zeros
         */
        static Tensor zeros_like(const Tensor &tensor);

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
         * @brief Shape of the tensor
         */
        std::vector<size_t> shape_;
    };

    // Template method implementations
    template<typename T>
    template<typename... Args>
    T &Tensor<T>::operator()(Args... indices) {
        static_assert(sizeof...(indices) > 0, "At least one index required");
        std::vector<size_t> idx_vec = {static_cast<size_t>(indices)...};
        if (idx_vec.size() != shape_.size()) {
            std::cerr << "DEBUG: Tensor access error - indices: " << idx_vec.size() 
                      << ", shape dimensions: " << shape_.size() << std::endl;
            throw std::out_of_range(
                "Number of indices does not match tensor dimensions");
        }
        for (size_t i = 0; i < idx_vec.size(); ++i) {
            if (idx_vec[i] >= shape_[i]) {
                throw std::out_of_range("Index out of bounds");
            }
        }
        return data_(indices...);
    }

    template<typename T>
    template<typename... Args>
    const T &Tensor<T>::operator()(Args... indices) const {
        static_assert(sizeof...(indices) > 0, "At least one index required");
        std::vector<size_t> idx_vec = {static_cast<size_t>(indices)...};
        if (idx_vec.size() != shape_.size()) {
            std::cerr << "DEBUG: Tensor const access error - indices: " << idx_vec.size() 
                      << ", shape dimensions: " << shape_.size() << std::endl;
            std::cerr << "DEBUG: Tensor shape: [";
            for (size_t i = 0; i < shape_.size(); ++i) {
                std::cerr << shape_[i];
                if (i < shape_.size() - 1) std::cerr << ", ";
            }
            std::cerr << "]" << std::endl;
            std::cerr << "DEBUG: Tensor size: " << data_.size() << std::endl;
            throw std::out_of_range(
                "Number of indices does not match tensor dimensions");
        }
        for (size_t i = 0; i < idx_vec.size(); ++i) {
            if (idx_vec[i] >= shape_[i]) {
                throw std::out_of_range("Index out of bounds");
            }
        }
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
