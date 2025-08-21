#include "utils/tensor.hpp"
#include <limits>
#include <numeric>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/generators/xrandom.hpp>
#include <xtensor/views/xview.hpp>

namespace dl {
    // Default constructor is defined inline in header

    // Constructor from data vector with shape (PyTorch-style)
    template<typename T>
    Tensor<T>::Tensor(const std::vector<T> &data, const std::vector<size_t> &shape)
        : shape_(shape) {
        // Calculate total size from shape
        size_t total_size = 1;
        for (size_t dim: shape) {
            total_size *= dim;
        }

        // Check if data size matches shape
        if (data.size() != total_size) {
            throw std::invalid_argument("Data size (" + std::to_string(data.size()) +
                                        ") does not match shape size (" + std::to_string(total_size) + ")");
        }

        // Create xtensor array from vector and reshape
        data_ = xt::adapt(data, shape);
    }


    // Constructor from initializer list with shape
    template<typename T>
    Tensor<T>::Tensor(std::initializer_list<T> data, const std::vector<size_t> &shape)
        : shape_(shape) {
        // Calculate total size from shape
        size_t total_size = 1;
        for (size_t dim: shape) {
            total_size *= dim;
        }

        // Check if data size matches shape
        if (data.size() != total_size) {
            throw std::invalid_argument("Data size (" + std::to_string(data.size()) +
                                        ") does not match shape size (" + std::to_string(total_size) + ")");
        }

        // Create xtensor array from initializer list and reshape
        data_ = xt::adapt(std::vector<T>(data), shape);
    }

    // Constructor from xt::xarray
    template<typename T>
    Tensor<T>::Tensor(const xt::xarray<T> &array) : data_(array) {
        auto array_shape = array.shape();
        shape_.assign(array_shape.begin(), array_shape.end());
    }

    // Scalar constructor
    template<typename T>
    Tensor<T>::Tensor(T value) : data_(value), shape_() {
        // 0-dimensional tensor (scalar)
    }

    // operator() methods are defined as variadic templates in header

    // Scalar access method for 0-dimensional tensors
    template<typename T>
    const T &Tensor<T>::scalar() const {
        if (shape_.size() != 0) {
            throw std::invalid_argument("scalar() can only be called on 0-dimensional tensors");
        }
        if (data_.size() == 0) {
            throw std::invalid_argument("Cannot call scalar() on empty tensor");
        }
        return data_();
    }


    // Getters
    // size(), rows(), cols(), and shape() are defined inline in the header

    // data() methods are defined inline in header

    // Arithmetic operations
    template<typename T>
    Tensor<T> Tensor<T>::operator+(const Tensor<T> &other) const {
        if (shape_ != other.shape_) {
            throw std::invalid_argument("Tensor dimensions must match for addition");
        }

        Tensor<T> result = Tensor<T>::zeros(shape_);
        result.data_ = data_ + other.data_;
        return result;
    }

    template<typename T>
    Tensor<T> Tensor<T>::operator-(const Tensor<T> &other) const {
        if (shape_ != other.shape_) {
            throw std::invalid_argument("Tensor dimensions must match for subtraction");
        }

        Tensor<T> result = Tensor<T>::zeros(shape_);
        result.data_ = data_ - other.data_;
        return result;
    }

    template<typename T>
    Tensor<T> Tensor<T>::operator*(const Tensor<T> &other) const {
        if (shape_ != other.shape_) {
            throw std::invalid_argument(
                "Tensor dimensions must match for element-wise multiplication");
        }

        Tensor<T> result = Tensor<T>::zeros(shape_);
        result.data_ = data_ * other.data_;
        return result;
    }

    template<typename T>
    Tensor<T> Tensor<T>::operator/(const Tensor<T> &other) const {
        if (shape_ != other.shape_) {
            throw std::invalid_argument("Tensor shapes must match for element-wise division");
        }
        Tensor<T> result = *this;
        result.data_ = data_ / other.data_;
        return result;
    }

    template<typename T>
    Tensor<T> Tensor<T>::operator-() const {
        Tensor<T> result = *this;
        result.data_ = -data_;
        return result;
    }

    template<typename T>
    Tensor<T> Tensor<T>::operator*(T scalar) const {
        Tensor<T> result = *this;
        result.data_ = data_ * scalar;
        return result;
    }

    template<typename T>
    Tensor<bool> compare_greater(const Tensor<T> &lhs, const Tensor<T> &rhs) {
        if (lhs.shape() != rhs.shape()) {
            throw std::invalid_argument("Tensor shapes must match for comparison");
        }
        auto result_data = lhs.data() > rhs.data();
        return Tensor<bool>(result_data);
    }

    template<typename T>
    Tensor<bool> Tensor<T>::operator>(const Tensor<T> &other) const {
        return compare_greater(*this, other);
    }

    template<typename T>
    Tensor<T> Tensor<T>::max(const Tensor<T> &other) const {
        if (shape_ != other.shape_) {
            throw std::invalid_argument("Tensor shapes must match for max operation");
        }
        Tensor<T> result = *this;
        result.data_ = xt::maximum(data_, other.data_);
        return result;
    }

    template<typename T>
    Tensor<T> Tensor<T>::exp() const {
        Tensor<T> result = *this;
        result.data_ = xt::exp(data_);
        return result;
    }

    template<typename T, typename U>
    Tensor<U> cast_tensor(const Tensor<T> &tensor) {
        auto casted_data = xt::cast<U>(tensor.data());
        return Tensor<U>(casted_data);
    }

    template<typename T>
    template<typename U>
    Tensor<U> Tensor<T>::cast() const {
        return cast_tensor<T, U>(*this);
    }

    template<typename T>
    Tensor<T> Tensor<T>::matmul(const Tensor<T> &other) const {
        if (shape_.size() < 2 || other.shape_.size() < 2) {
            throw std::invalid_argument(
                "Matrix multiplication requires at least 2D tensors");
        }
        if (shape_[shape_.size() - 1] != other.shape_[other.shape_.size() - 2]) {
            throw std::invalid_argument(
                "Matrix dimensions are incompatible for multiplication");
        }

        return Tensor<T>::from_array(xt::linalg::dot(data_, other.data_));
    }

    // Tensor operations
    template<typename T>
    Tensor<T> Tensor<T>::transpose() const {
        if (shape_.size() != 2) {
            throw std::invalid_argument("Transpose requires 2D tensor");
        }
        std::vector<size_t> new_shape = {shape_[1], shape_[0]};
        Tensor<T> result = Tensor<T>::zeros(new_shape);
        result.data_ = xt::transpose(data_);
        return result;
    }

    template<typename T>
    Tensor<T> Tensor<T>::transpose(const std::vector<size_t> &axes) const {
        if (axes.size() != data_.dimension()) {
            throw std::invalid_argument("Number of axes must match tensor dimensions");
        }

        std::vector<size_t> new_shape(axes.size());
        for (size_t i = 0; i < axes.size(); ++i) {
            new_shape[i] = shape_[axes[i]];
        }

        Tensor<T> result = Tensor<T>::zeros(new_shape);
        result.data_ = xt::transpose(data_, axes);
        return result;
    }

    template<typename T>
    Tensor<T> Tensor<T>::reshape(const std::vector<size_t> &new_shape) const {
        size_t old_size = std::accumulate(shape_.begin(), shape_.end(), 1UL,
                                          std::multiplies<size_t>());
        size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1UL,
                                          std::multiplies<size_t>());

        if (old_size != new_size) {
            throw std::invalid_argument("Total size must remain the same for reshape");
        }

        Tensor<T> result = Tensor<T>::zeros(new_shape);
        result.data_ = xt::reshape_view(data_, new_shape);
        return result;
    }


    template<typename T>
    Tensor<T> Tensor<T>::view(const std::vector<size_t> &new_shape) const {
        size_t old_size = std::accumulate(shape_.begin(), shape_.end(), 1UL,
                                          std::multiplies<size_t>());
        size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1UL,
                                          std::multiplies<size_t>());

        if (old_size != new_size) {
            throw std::invalid_argument("Total size must remain the same for view");
        }

        Tensor<T> result = Tensor<T>::zeros(new_shape);
        result.data_ = xt::reshape_view(data_, new_shape);
        return result;
    }

    template<typename T>
    Tensor<T> Tensor<T>::squeeze(int axis) const {
        std::vector<size_t> new_shape;

        if (axis == -1) {
            // Squeeze all dimensions of size 1
            for (size_t dim: shape_) {
                if (dim != 1) {
                    new_shape.push_back(dim);
                }
            }
        } else {
            // Squeeze specific axis
            if (axis >= static_cast<int>(shape_.size()) || shape_[axis] != 1) {
                throw std::invalid_argument("Cannot squeeze dimension that is not 1");
            }
            for (size_t i = 0; i < shape_.size(); ++i) {
                if (i != static_cast<size_t>(axis)) {
                    new_shape.push_back(shape_[i]);
                }
            }
        }

        if (new_shape.empty()) {
            new_shape.push_back(1); // Ensure at least 1D
        }

        return view(new_shape);
    }

    template<typename T>
    Tensor<T> Tensor<T>::unsqueeze(size_t axis) const {
        if (axis > shape_.size()) {
            throw std::invalid_argument("Axis out of range for unsqueeze");
        }

        std::vector<size_t> new_shape = shape_;
        new_shape.insert(new_shape.begin() + axis, 1);

        return view(new_shape);
    }

    template<typename T>
    T Tensor<T>::determinant() const {
        if (data_.dimension() != 2 || shape_[0] != shape_[1]) {
            throw std::invalid_argument("Determinant requires square 2D tensor");
        }

        // For 2x2 matrices, compute determinant directly
        if (shape_[0] == 2) {
            return data_(0, 0) * data_(1, 1) - data_(0, 1) * data_(1, 0);
        }

        // For larger matrices, use xtensor-blas when available
        throw std::runtime_error(
            "Determinant computation for matrices larger than 2x2 temporarily disabled due to "
            "xtensor-blas compatibility issue");
        // return xt::linalg::det(data_);
    }

    template<typename T>
    Tensor<T> Tensor<T>::inverse() const {
        if (data_.dimension() != 2 || shape_[0] != shape_[1]) {
            throw std::invalid_argument("Inverse requires square 2D tensor");
        }

        // For 2x2 matrices, compute inverse directly
        if (shape_[0] == 2) {
            T det = determinant();
            if (std::abs(det) < std::numeric_limits<T>::epsilon()) {
                throw std::invalid_argument("Matrix is singular and cannot be inverted");
            }

            std::vector<size_t> result_shape = {2, 2};
            Tensor<T> result = Tensor<T>::zeros(result_shape);
            result.data_(0, 0) = data_(1, 1) / det;
            result.data_(0, 1) = -data_(0, 1) / det;
            result.data_(1, 0) = -data_(1, 0) / det;
            result.data_(1, 1) = data_(0, 0) / det;
            return result;
        }

        // For larger matrices, use xtensor-blas when available
        throw std::runtime_error(
            "Matrix inverse computation for matrices larger than 2x2 temporarily disabled due to "
            "xtensor-blas compatibility issue");
        // Tensor<T> result(shape_[0], shape_[1]);
        // result.data_ = xt::linalg::inv(data_);
        // return result;
    }

    template<typename T>
    auto Tensor<T>::eigenvalues() const {
        if (data_.dimension() != 2 || shape_[0] != shape_[1]) {
            throw std::invalid_argument("Eigenvalues require square 2D tensor");
        }
        // TODO: Fix xtensor-blas compatibility issue with eigenvalues
        throw std::runtime_error(
            "Eigenvalues computation temporarily disabled due to xtensor-blas compatibility issue");
        // return xt::linalg::eigvals(data_);
    }

    // Static factory methods
    template<typename T>
    Tensor<T> Tensor<T>::zeros(const std::vector<size_t> &shape) {
        size_t total_size = 1;
        for (size_t dim: shape) {
            total_size *= dim;
        }
        std::vector<T> data(total_size, T(0));
        return Tensor<T>(data, shape);
    }


    template<typename T>
    Tensor<T> Tensor<T>::ones(const std::vector<size_t> &shape) {
        size_t total_size = 1;
        for (size_t dim: shape) {
            total_size *= dim;
        }
        std::vector<T> data(total_size, T(1));
        return Tensor<T>(data, shape);
    }


    template<typename T>
    Tensor<T> Tensor<T>::full(const std::vector<size_t> &shape, T value) {
        size_t total_size = 1;
        for (size_t dim: shape) {
            total_size *= dim;
        }
        std::vector<T> data(total_size, value);
        return Tensor<T>(data, shape);
    }

    template<typename T>
    Tensor<T> Tensor<T>::identity(size_t size) {
        std::vector<size_t> shape = {size, size};
        Tensor<T> result = Tensor<T>::zeros(shape);
        result.data_ = xt::eye<T>(size);
        return result;
    }

    template<typename T>
    Tensor<T> Tensor<T>::random(const std::vector<size_t> &shape) {
        Tensor<T> result = Tensor<T>::zeros(shape);
        if constexpr (std::is_integral_v<T>) {
            result.data_ = xt::random::randint<T>(shape, T(0), T(10));
        } else {
            result.data_ = xt::random::rand<T>(shape, T(0), T(1));
        }
        return result;
    }

    template<typename T>
    Tensor<T> Tensor<T>::random(const std::vector<size_t> &shape, T min, T max) {
        Tensor<T> result = Tensor<T>::zeros(shape);
        if constexpr (std::is_integral_v<T>) {
            result.data_ = xt::random::randint<T>(shape, min, max);
        } else {
            result.data_ = xt::random::rand<T>(shape, min, max);
        }
        return result;
    }

    // Non-member functions
    template<typename T>
    std::ostream &operator<<(std::ostream &os, const Tensor<T> &tensor) {
        os << tensor.data();
        return os;
    }

    template<typename T>
    Tensor<T> dot(const Tensor<T> &a, const Tensor<T> &b) {
        return Tensor<T>::from_array(xt::linalg::dot(a.data(), b.data()));
    }

    template<typename T>
    T sum(const Tensor<T> &tensor) {
        return xt::sum(tensor.data())();
    }

    template<typename T>
    T mean(const Tensor<T> &tensor) {
        return xt::mean(tensor.data())();
    }

    // New tensor methods implementations
    template<typename T>
    Tensor<T> Tensor<T>::log() const {
        Tensor<T> result = Tensor<T>::zeros(shape_);
        result.data_ = xt::log(data_);
        return result;
    }

    template<typename T>
    Tensor<T> Tensor<T>::sigmoid() const {
        Tensor<T> result = Tensor<T>::zeros(shape_);
        result.data_ = 1.0 / (1.0 + xt::exp(-data_));
        return result;
    }

    template<typename T>
    Tensor<T> Tensor<T>::tanh() const {
        Tensor<T> result = Tensor<T>::zeros(shape_);
        result.data_ = xt::tanh(data_);
        return result;
    }

    template<typename T>
    Tensor<T> Tensor<T>::relu() const {
        Tensor<T> result = Tensor<T>::zeros(shape_);
        result.data_ = xt::maximum(data_, static_cast<T>(0));
        return result;
    }

    template<typename T>
    Tensor<T> Tensor<T>::sum() const {
        T sum_val = xt::sum(data_)();
        return Tensor<T>(std::vector<T>{sum_val}, std::vector<size_t>{});
    }

    template<typename T>
    Tensor<T> Tensor<T>::sum(const std::vector<int> &axes, bool keepdims) const {
        auto result = data_;
        for (int axis: axes) {
            if (axis >= 0 && axis < static_cast<int>(result.shape().size())) {
                if (keepdims) {
                    result = xt::sum(result, {static_cast<size_t>(axis)}, xt::keep_dims);
                } else {
                    result = xt::sum(result, {static_cast<size_t>(axis)});
                }
            }
        }
        return Tensor<T>::from_array(result);
    }

    template<typename T>
    Tensor<T> Tensor<T>::mean() const {
        T mean_val = xt::mean(data_)();
        return Tensor<T>(std::vector<T>{mean_val}, std::vector<size_t>{});
    }

    template<typename T>
    Tensor<T> Tensor<T>::mean(const std::vector<int> &axes, bool keepdims) const {
        auto result = data_;
        for (int axis: axes) {
            if (axis >= 0 && axis < static_cast<int>(result.shape().size())) {
                if (keepdims) {
                    result = xt::mean(result, {static_cast<size_t>(axis)}, xt::keep_dims);
                } else {
                    result = xt::mean(result, {static_cast<size_t>(axis)});
                }
            }
        }
        return Tensor<T>::from_array(result);
    }

    template<typename T>
    bool Tensor<T>::empty() const {
        return data_.size() == 0;
    }

    template<typename T>
    void Tensor<T>::clear() {
        data_ = xt::xarray<T>();
        shape_.clear();
    }

    template<typename T>
    Tensor<T> Tensor<T>::ones_like(const Tensor<T> &tensor) {
        return Tensor<T>::ones(tensor.shape_);
    }

    template<typename T>
    Tensor<T> Tensor<T>::zeros_like(const Tensor<T> &tensor) {
        return Tensor<T>::zeros(tensor.shape_);
    }

    // Explicit template instantiations
    template class Tensor<float>;
    template class Tensor<double>;
    template class Tensor<int>;


    // Explicit instantiation of friend functions
    template std::ostream &operator<<<float>(std::ostream &, const Tensor<float> &);

    template std::ostream &operator<<<double>(std::ostream &,
                                              const Tensor<double> &);

    template Tensor<float> dot<float>(const Tensor<float> &, const Tensor<float> &);

    template Tensor<double> dot<double>(const Tensor<double> &,
                                        const Tensor<double> &);

    template float sum<float>(const Tensor<float> &);

    template double sum<double>(const Tensor<double> &);

    template float mean<float>(const Tensor<float> &);

    template double mean<double>(const Tensor<double> &);

    // Explicit instantiations for cast function
    template Tensor<float> cast_tensor<bool, float>(const Tensor<bool> &);

    template Tensor<double> cast_tensor<bool, double>(const Tensor<bool> &);

    template Tensor<bool> cast_tensor<float, bool>(const Tensor<float> &);

    template Tensor<bool> cast_tensor<double, bool>(const Tensor<double> &);

    template Tensor<float> Tensor<bool>::cast<float>() const;

    template Tensor<double> Tensor<bool>::cast<double>() const;

    template Tensor<bool> Tensor<float>::cast<bool>() const;

    template Tensor<bool> Tensor<double>::cast<bool>() const;

    // Explicit instantiations for compare_greater function
    template Tensor<bool> compare_greater<float>(const Tensor<float> &, const Tensor<float> &);

    template Tensor<bool> compare_greater<double>(const Tensor<double> &, const Tensor<double> &);

    template Tensor<bool> compare_greater<int>(const Tensor<int> &, const Tensor<int> &);
} // namespace dl
