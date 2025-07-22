#include "utils/matrix.hpp"
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor-blas/xlinalg.hpp>

namespace dl {
    namespace utils {
        template<typename T>
        Matrix<T>::Matrix(size_t rows, size_t cols) : rows_(rows), cols_(cols) {
            data_ = xt::zeros<T>({rows, cols});
        }

        template<typename T>
        Matrix<T>::Matrix(size_t rows, size_t cols, T value) : rows_(rows), cols_(cols) {
            data_ = xt::ones<T>({rows, cols}) * value;
        }

        template<typename T>
        Matrix<T>::Matrix(std::initializer_list<std::initializer_list<T>> list) {
            rows_ = list.size();
            cols_ = list.begin()->size();

            std::vector<T> flat_data;
            for (const auto &row: list) {
                for (const auto &val: row) {
                    flat_data.push_back(val);
                }
            }

            data_ = xt::adapt(flat_data, {rows_, cols_});
        }

        template<typename T>
        T &Matrix<T>::operator()(size_t row, size_t col) {
            if (row >= rows_ || col >= cols_) {
                throw std::out_of_range("Matrix index out of bounds");
            }
            return data_(row, col);
        }

        template<typename T>
        const T &Matrix<T>::operator()(size_t row, size_t col) const {
            if (row >= rows_ || col >= cols_) {
                throw std::out_of_range("Matrix index out of bounds");
            }
            return data_(row, col);
        }

        template<typename T>
        Matrix<T> Matrix<T>::operator+(const Matrix<T> &other) const {
            if (rows_ != other.rows_ || cols_ != other.cols_) {
                throw std::invalid_argument("Matrix dimensions must match for addition");
            }

            Matrix<T> result(rows_, cols_);
            result.data_ = data_ + other.data_;
            return result;
        }

        template<typename T>
        Matrix<T> Matrix<T>::operator-(const Matrix<T> &other) const {
            if (rows_ != other.rows_ || cols_ != other.cols_) {
                throw std::invalid_argument("Matrix dimensions must match for subtraction");
            }

            Matrix<T> result(rows_, cols_);
            result.data_ = data_ - other.data_;
            return result;
        }

        template<typename T>
        Matrix<T> Matrix<T>::operator*(const Matrix<T> &other) const {
            if (cols_ != other.rows_) {
                throw std::invalid_argument("Invalid dimensions for matrix multiplication");
            }

            Matrix<T> result(rows_, other.cols_);
            result.data_ = xt::linalg::dot(data_, other.data_);
            return result;
        }

        template<typename T>
        Matrix<T> Matrix<T>::transpose() const {
            Matrix<T> result(cols_, rows_);
            result.data_ = xt::transpose(data_);
            return result;
        }

        template<typename T>
        Matrix<T> Matrix<T>::reshape(size_t new_rows, size_t new_cols) const {
            if (new_rows * new_cols != rows_ * cols_) {
                throw std::invalid_argument("New dimensions must have same total size");
            }

            Matrix<T> result(new_rows, new_cols);
            result.data_ = xt::reshape_view(data_, {new_rows, new_cols});
            return result;
        }

        template<typename T>
        T Matrix<T>::determinant() const {
            if (rows_ != cols_) {
                throw std::invalid_argument("Determinant only defined for square matrices");
            }

            return xt::linalg::det(data_);
        }

        template<typename T>
        Matrix<T> Matrix<T>::inverse() const {
            if (rows_ != cols_) {
                throw std::invalid_argument("Inverse only defined for square matrices");
            }

            Matrix<T> result(rows_, cols_);
            result.data_ = xt::linalg::inv(data_);
            return result;
        }

        template<typename T>
        Matrix<T> Matrix<T>::zeros(size_t rows, size_t cols) {
            Matrix<T> result(rows, cols);
            result.data_ = xt::zeros<T>({rows, cols});
            return result;
        }

        template<typename T>
        Matrix<T> Matrix<T>::ones(size_t rows, size_t cols) {
            Matrix<T> result(rows, cols);
            result.data_ = xt::ones<T>({rows, cols});
            return result;
        }

        template<typename T>
        Matrix<T> Matrix<T>::identity(size_t size) {
            Matrix<T> result(size, size);
            result.data_ = xt::eye<T>(size);
            return result;
        }

        template<typename T>
        Matrix<T> Matrix<T>::random(size_t rows, size_t cols, T min, T max) {
            Matrix<T> result(rows, cols);
            result.data_ = xt::random::rand<T>({rows, cols}, min, max);
            return result;
        }
        template<typename T>
        std::ostream &operator<<(std::ostream &os, const Matrix<T> &matrix) {
            os << matrix.data();
            return os;
        }
        template<typename T>
        Matrix<T> dot(const Matrix<T> &a, const Matrix<T> &b) {
            if (a.cols() != b.rows()) {
                throw std::invalid_argument("Invalid dimensions for dot product");
            }

            Matrix<T> result(a.rows(), b.cols());
            result.data() = xt::linalg::dot(a.data(), b.data());
            return result;
        }
        template<typename T>
        T sum(const Matrix<T> &matrix) {
            return xt::sum(matrix.data())(0);
        }
        template<typename T>
        T mean(const Matrix<T> &matrix) {
            return xt::mean(matrix.data())(0);
        }

        // Explicit template instantiations
        template class Matrix<float>;
        template class Matrix<double>;

        // Non-member function implementations
        // Explicit instantiation of non-member functions
        template std::ostream &operator<< <float>(std::ostream &, const Matrix<float> &);
        template std::ostream &operator<< <double>(std::ostream &, const Matrix<double> &);

        template Matrix<float> dot<float>(const Matrix<float> &, const Matrix<float> &);
        template Matrix<double> dot<double>(const Matrix<double> &, const Matrix<double> &);

        template float sum<float>(const Matrix<float> &);
        template double sum<double>(const Matrix<double> &);

        template float mean<float>(const Matrix<float> &);
        template double mean<double>(const Matrix<double> &);
    } // namespace utils
} // namespace dl
