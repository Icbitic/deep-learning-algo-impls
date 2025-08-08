#include <numeric>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/generators/xrandom.hpp>
#include <xtensor/views/xview.hpp>
#include "utils/tensor.hpp"

namespace utils {
    // Default constructor is defined inline in header

    // N-dimensional constructor
    template<typename T>
    Tensor<T>::Tensor(const std::vector<size_t>& shape) : rows_(shape.size() > 0 ? shape[0] : 0), cols_(shape.size() > 1 ? shape[1] : 1), data_(xt::zeros<T>(shape)) {}

    // N-dimensional constructor with value
    template<typename T>
    Tensor<T>::Tensor(const std::vector<size_t>& shape, T value) : rows_(shape.size() > 0 ? shape[0] : 0), cols_(shape.size() > 1 ? shape[1] : 1), data_(xt::ones<T>(shape) * value) {}

    // 2D constructor (backward compatibility)
    template<typename T>
    Tensor<T>::Tensor(size_t rows, size_t cols) : rows_(rows), cols_(cols), data_(xt::zeros<T>({rows, cols})) {}

    // 2D constructor with value (backward compatibility)
    template<typename T>
    Tensor<T>::Tensor(size_t rows, size_t cols, T value) : rows_(rows), cols_(cols), data_(xt::ones<T>({rows, cols}) * value) {}

    // Constructor from xt::xarray
    template<typename T>
    Tensor<T>::Tensor(const xt::xarray<T>& array) : data_(array) {
        auto shape = array.shape();
        rows_ = shape.size() > 0 ? shape[0] : 0;
        cols_ = shape.size() > 1 ? shape[1] : 1;
    }

    // 2D initializer list constructor (backward compatibility)
    template<typename T>
    Tensor<T>::Tensor(std::initializer_list<std::initializer_list<T>> list) {
        rows_ = list.size();
        cols_ = list.begin()->size();

        std::vector<T> flat_data;
        flat_data.reserve(rows_ * cols_);

        for (const auto &row: list) {
            std::copy(row.begin(), row.end(), std::back_inserter(flat_data));
        }

        data_ = xt::adapt(flat_data, {rows_, cols_});
    }

    // operator() methods are defined as variadic templates in header

    // Element access for 2D using at() method
    template<typename T>
    T &Tensor<T>::at(size_t row, size_t col) {
        return operator()(row, col);
    }

    template<typename T>
    const T &Tensor<T>::at(size_t row, size_t col) const {
        return operator()(row, col);
    }

    // Getters
    // size(), rows(), cols(), and shape() are defined inline in the header

    // data() methods are defined inline in header

    // Arithmetic operations
    template<typename T>
    Tensor<T> Tensor<T>::operator+(const Tensor<T> &other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw std::invalid_argument("Tensor shapes must match for addition");
        }

        Tensor<T> result(rows_, cols_);
        result.data_ = data_ + other.data_;
        return result;
    }

    template<typename T>
    Tensor<T> Tensor<T>::operator-(const Tensor<T> &other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw std::invalid_argument("Tensor shapes must match for subtraction");
        }

        Tensor<T> result(rows_, cols_);
        result.data_ = data_ - other.data_;
        return result;
    }

    template<typename T>
    Tensor<T> Tensor<T>::operator*(const Tensor<T> &other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw std::invalid_argument("Tensor shapes must match for element-wise multiplication");
        }

        Tensor<T> result(rows_, cols_);
        result.data_ = data_ * other.data_;
        return result;
    }

    template<typename T>
    Tensor<T> Tensor<T>::operator*(T scalar) const {
        Tensor<T> result(rows_, cols_);
        result.data_ = data_ * scalar;
        return result;
    }

    template<typename T>
    Tensor<T> Tensor<T>::matmul(const Tensor<T> &other) const {
        if (data_.dimension() != 2 || other.data_.dimension() != 2) {
            throw std::invalid_argument("Matrix multiplication requires 2D tensors");
        }
        if (cols_ != other.rows_) {
            throw std::invalid_argument("Tensor dimensions incompatible for matrix multiplication");
        }

        // TODO: Fix xtensor-blas compatibility issue with matrix multiplication
        throw std::runtime_error("Matrix multiplication temporarily disabled due to xtensor-blas compatibility issue");
        // Tensor<T> result(rows_, other.cols_);
        // result.data_ = xt::linalg::dot(data_, other.data_);
        // return result;
    }

    // Tensor operations
    template<typename T>
    Tensor<T> Tensor<T>::transpose() const {
        if (data_.dimension() != 2) {
            throw std::invalid_argument("Transpose requires 2D tensor");
        }
        
        Tensor<T> result(cols_, rows_);
        result.data_ = xt::transpose(data_);
        return result;
    }

    template<typename T>
    Tensor<T> Tensor<T>::transpose(const std::vector<size_t>& axes) const {
        if (axes.size() != data_.dimension()) {
            throw std::invalid_argument("Number of axes must match tensor dimensions");
        }
        
        std::vector<size_t> current_shape = {rows_, cols_};
        std::vector<size_t> new_shape(axes.size());
        for (size_t i = 0; i < axes.size(); ++i) {
            new_shape[i] = current_shape[axes[i]];
        }
        
        Tensor<T> result(new_shape);
        result.data_ = xt::transpose(data_, axes);
        return result;
    }

    template<typename T>
    Tensor<T> Tensor<T>::reshape(const std::vector<size_t>& new_shape) const {
        size_t old_size = rows_ * cols_;
        size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1UL, std::multiplies<size_t>());
        
        if (old_size != new_size) {
            throw std::invalid_argument("Total size must remain the same for reshape");
        }

        Tensor<T> result(new_shape);
        result.data_ = xt::reshape_view(data_, new_shape);
        return result;
    }

    template<typename T>
    Tensor<T> Tensor<T>::reshape(size_t new_rows, size_t new_cols) const {
        return reshape({new_rows, new_cols});
    }

    template<typename T>
    Tensor<T> Tensor<T>::view(const std::vector<size_t>& new_shape) const {
        size_t old_size = rows_ * cols_;
        size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1UL, std::multiplies<size_t>());
        
        if (old_size != new_size) {
            throw std::invalid_argument("Total size must remain the same for view");
        }

        Tensor<T> result(new_shape);
        result.data_ = xt::reshape_view(data_, new_shape);
        return result;
    }

    template<typename T>
    Tensor<T> Tensor<T>::squeeze(int axis) const {
        std::vector<size_t> current_shape = {rows_, cols_};
        std::vector<size_t> new_shape;
        
        if (axis == -1) {
            // Squeeze all dimensions of size 1
            for (size_t dim : current_shape) {
                if (dim != 1) {
                    new_shape.push_back(dim);
                }
            }
        } else {
            // Squeeze specific axis
            if (axis >= static_cast<int>(current_shape.size()) || current_shape[axis] != 1) {
                throw std::invalid_argument("Cannot squeeze dimension that is not 1");
            }
            for (size_t i = 0; i < current_shape.size(); ++i) {
                if (i != static_cast<size_t>(axis)) {
                    new_shape.push_back(current_shape[i]);
                }
            }
        }
        
        if (new_shape.empty()) {
            new_shape.push_back(1);  // Ensure at least 1D
        }
        
        return view(new_shape);
    }

    template<typename T>
    Tensor<T> Tensor<T>::unsqueeze(size_t axis) const {
        std::vector<size_t> current_shape = {rows_, cols_};
        if (axis > current_shape.size()) {
            throw std::invalid_argument("Axis out of range for unsqueeze");
        }
        
        std::vector<size_t> new_shape = current_shape;
        new_shape.insert(new_shape.begin() + axis, 1);
        
        return view(new_shape);
    }

    template<typename T>
    T Tensor<T>::determinant() const {
        if (data_.dimension() != 2 || rows_ != cols_) {
            throw std::invalid_argument("Determinant requires square 2D tensor");
        }
        // TODO: Fix xtensor-blas compatibility issue with determinant
        throw std::runtime_error("Determinant computation temporarily disabled due to xtensor-blas compatibility issue");
        // return xt::linalg::det(data_);
    }

    template<typename T>
    Tensor<T> Tensor<T>::inverse() const {
        if (data_.dimension() != 2 || rows_ != cols_) {
            throw std::invalid_argument("Inverse requires square 2D tensor");
        }

        // TODO: Fix xtensor-blas compatibility issue with inverse
        throw std::runtime_error("Matrix inverse computation temporarily disabled due to xtensor-blas compatibility issue");
        // Tensor<T> result(rows_, cols_);
        // result.data_ = xt::linalg::inv(data_);
        // return result;
    }

    template<typename T>
    auto Tensor<T>::eigenvalues() const {
        if (data_.dimension() != 2 || rows_ != cols_) {
            throw std::invalid_argument("Eigenvalues require square 2D tensor");
        }
        // TODO: Fix xtensor-blas compatibility issue with eigenvalues
        throw std::runtime_error("Eigenvalues computation temporarily disabled due to xtensor-blas compatibility issue");
        // return xt::linalg::eigvals(data_);
    }

    // Static factory methods
    template<typename T>
    Tensor<T> Tensor<T>::zeros(const std::vector<size_t>& shape) {
        return Tensor<T>(shape);
    }

    template<typename T>
    Tensor<T> Tensor<T>::zeros(size_t rows, size_t cols) {
        return Tensor<T>(rows, cols);
    }

    template<typename T>
    Tensor<T> Tensor<T>::ones(const std::vector<size_t>& shape) {
        Tensor<T> result(shape);
        result.data_ = xt::ones<T>(shape);
        return result;
    }

    template<typename T>
    Tensor<T> Tensor<T>::ones(size_t rows, size_t cols) {
        return ones({rows, cols});
    }

    template<typename T>
    Tensor<T> Tensor<T>::full(const std::vector<size_t>& shape, T value) {
        return Tensor<T>(shape, value);
    }

    template<typename T>
    Tensor<T> Tensor<T>::identity(size_t size) {
        std::vector<size_t> shape = {size, size};
        Tensor<T> result(shape);
        result.data_ = xt::eye<T>(size);
        return result;
    }

    template<typename T>
    Tensor<T> Tensor<T>::random(const std::vector<size_t>& shape) {
        Tensor<T> result(shape);
        if constexpr (std::is_integral_v<T>) {
            result.data_ = xt::random::randint<T>(shape, T(0), T(10));
        } else {
            result.data_ = xt::random::rand<T>(shape, T(0), T(1));
        }
        return result;
    }

    template<typename T>
    Tensor<T> Tensor<T>::random(const std::vector<size_t>& shape, T min, T max) {
        Tensor<T> result(shape);
        if constexpr (std::is_integral_v<T>) {
            result.data_ = xt::random::randint<T>(shape, min, max);
        } else {
            result.data_ = xt::random::rand<T>(shape, min, max);
        }
        return result;
    }

    template<typename T>
    Tensor<T> Tensor<T>::random(size_t rows, size_t cols) {
        std::vector<size_t> shape = {rows, cols};
        Tensor<T> result(shape);
        if constexpr (std::is_integral_v<T>) {
            result.data_ = xt::random::randint<T>(shape, T(0), T(10));
        } else {
            result.data_ = xt::random::rand<T>(shape, T(0), T(1));
        }
        return result;
    }

    template<typename T>
    Tensor<T> Tensor<T>::random(size_t rows, size_t cols, T min, T max) {
        std::vector<size_t> shape = {rows, cols};
        Tensor<T> result(shape);
        if constexpr (std::is_integral_v<T>) {
            result.data_ = xt::random::randint<T>(shape, min, max);
        } else {
            result.data_ = xt::random::rand<T>(shape, min, max);
        }
        return result;
    }

    template<typename T>
    Tensor<T> Tensor<T>::from_array(const xt::xarray<T>& array) {
        return Tensor<T>(array);
    }

    // Non-member functions
    template<typename T>
    std::ostream &operator<<(std::ostream &os, const Tensor<T> &tensor) {
        os << tensor.data();
        return os;
    }

    template<typename T>
    Tensor<T> dot(const Tensor<T> &a, const Tensor<T> &b) {
        if (a.data().dimension() != 2 || b.data().dimension() != 2) {
            throw std::invalid_argument("Dot product requires 2D tensors");
        }
        if (a.cols() != b.rows()) {
            throw std::invalid_argument("Tensor dimensions incompatible for dot product");
        }

        // TODO: Fix xtensor-blas compatibility issue with dot product
        throw std::runtime_error("Dot product computation temporarily disabled due to xtensor-blas compatibility issue");
        // Tensor<T> result(a.rows(), b.cols());
        // result.data() = xt::linalg::dot(a.data(), b.data());
        // return result;
    }

    template<typename T>
    T sum(const Tensor<T> &tensor) {
        return xt::sum(tensor.data())();
    }

    template<typename T>
    T mean(const Tensor<T> &tensor) {
        return xt::mean(tensor.data())();
    }

    // Explicit template instantiations
    template class Tensor<float>;
    template class Tensor<double>;
    template class Tensor<int>;

    // Explicit instantiation of non-member functions
    template std::ostream &operator<< <float>(std::ostream &, const Tensor<float> &);
    template std::ostream &operator<< <double>(std::ostream &, const Tensor<double> &);

    template Tensor<float> dot<float>(const Tensor<float> &, const Tensor<float> &);
    template Tensor<double> dot<double>(const Tensor<double> &, const Tensor<double> &);

    template float sum<float>(const Tensor<float> &);
    template double sum<double>(const Tensor<double> &);

    template float mean<float>(const Tensor<float> &);
    template double mean<double>(const Tensor<double> &);
} // namespace utils
