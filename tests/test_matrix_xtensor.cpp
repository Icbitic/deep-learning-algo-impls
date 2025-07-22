#include <gtest/gtest.h>
#include "utils/matrix.hpp"

using namespace dl::utils;

TEST(MatrixXTensorTest, Construction) {
    // Test default constructor
    Matrix<double> m1;
    EXPECT_EQ(m1.rows(), 0);
    EXPECT_EQ(m1.cols(), 0);
    
    // Test size constructor
    Matrix<double> m2(3, 4);
    EXPECT_EQ(m2.rows(), 3);
    EXPECT_EQ(m2.cols(), 4);
    
    // Test value constructor
    Matrix<double> m3(2, 2, 5.0);
    EXPECT_EQ(m3(0, 0), 5.0);
    EXPECT_EQ(m3(0, 1), 5.0);
    EXPECT_EQ(m3(1, 0), 5.0);
    EXPECT_EQ(m3(1, 1), 5.0);
    
    // Test initializer list constructor
    Matrix<double> m4 = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };
    EXPECT_EQ(m4.rows(), 2);
    EXPECT_EQ(m4.cols(), 3);
    EXPECT_EQ(m4(0, 0), 1.0);
    EXPECT_EQ(m4(0, 1), 2.0);
    EXPECT_EQ(m4(0, 2), 3.0);
    EXPECT_EQ(m4(1, 0), 4.0);
    EXPECT_EQ(m4(1, 1), 5.0);
    EXPECT_EQ(m4(1, 2), 6.0);
}

TEST(MatrixXTensorTest, ElementAccess) {
    Matrix<double> m(2, 3);
    
    // Test element assignment
    m(0, 0) = 1.0;
    m(0, 1) = 2.0;
    m(0, 2) = 3.0;
    m(1, 0) = 4.0;
    m(1, 1) = 5.0;
    m(1, 2) = 6.0;
    
    // Test element access
    EXPECT_EQ(m(0, 0), 1.0);
    EXPECT_EQ(m(0, 1), 2.0);
    EXPECT_EQ(m(0, 2), 3.0);
    EXPECT_EQ(m(1, 0), 4.0);
    EXPECT_EQ(m(1, 1), 5.0);
    EXPECT_EQ(m(1, 2), 6.0);
    
    // Test out of bounds access
    EXPECT_THROW(m(2, 0), std::out_of_range);
    EXPECT_THROW(m(0, 3), std::out_of_range);
}

TEST(MatrixXTensorTest, ArithmeticOperations) {
    Matrix<double> m1 = {
        {1.0, 2.0},
        {3.0, 4.0}
    };
    
    Matrix<double> m2 = {
        {5.0, 6.0},
        {7.0, 8.0}
    };
    
    // Test addition
    Matrix<double> m3 = m1 + m2;
    EXPECT_EQ(m3(0, 0), 6.0);
    EXPECT_EQ(m3(0, 1), 8.0);
    EXPECT_EQ(m3(1, 0), 10.0);
    EXPECT_EQ(m3(1, 1), 12.0);
    
    // Test subtraction
    Matrix<double> m4 = m2 - m1;
    EXPECT_EQ(m4(0, 0), 4.0);
    EXPECT_EQ(m4(0, 1), 4.0);
    EXPECT_EQ(m4(1, 0), 4.0);
    EXPECT_EQ(m4(1, 1), 4.0);
    
    // Test multiplication
    Matrix<double> m5 = {
        {1.0, 2.0},
        {3.0, 4.0}
    };
    
    Matrix<double> m6 = {
        {5.0, 6.0},
        {7.0, 8.0}
    };
    
    Matrix<double> m7 = m5 * m6;
    EXPECT_EQ(m7(0, 0), 19.0);
    EXPECT_EQ(m7(0, 1), 22.0);
    EXPECT_EQ(m7(1, 0), 43.0);
    EXPECT_EQ(m7(1, 1), 50.0);
}

TEST(MatrixXTensorTest, MatrixOperations) {
    // Test transpose
    Matrix<double> m1 = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };
    
    Matrix<double> m2 = m1.transpose();
    EXPECT_EQ(m2.rows(), 3);
    EXPECT_EQ(m2.cols(), 2);
    EXPECT_EQ(m2(0, 0), 1.0);
    EXPECT_EQ(m2(0, 1), 4.0);
    EXPECT_EQ(m2(1, 0), 2.0);
    EXPECT_EQ(m2(1, 1), 5.0);
    EXPECT_EQ(m2(2, 0), 3.0);
    EXPECT_EQ(m2(2, 1), 6.0);
    
    // Test reshape
    Matrix<double> m3 = m1.reshape(3, 2);
    EXPECT_EQ(m3.rows(), 3);
    EXPECT_EQ(m3.cols(), 2);
    EXPECT_EQ(m3(0, 0), 1.0);
    EXPECT_EQ(m3(0, 1), 2.0);
    EXPECT_EQ(m3(1, 0), 3.0);
    EXPECT_EQ(m3(1, 1), 4.0);
    EXPECT_EQ(m3(2, 0), 5.0);
    EXPECT_EQ(m3(2, 1), 6.0);
    
    // Test determinant
    Matrix<double> m4 = {
        {1.0, 2.0},
        {3.0, 4.0}
    };
    EXPECT_DOUBLE_EQ(m4.determinant(), -2.0);
    
    // Test inverse
    Matrix<double> m5 = m4.inverse();
    EXPECT_DOUBLE_EQ(m5(0, 0), -2.0);
    EXPECT_DOUBLE_EQ(m5(0, 1), 1.0);
    EXPECT_DOUBLE_EQ(m5(1, 0), 1.5);
    EXPECT_DOUBLE_EQ(m5(1, 1), -0.5);
}

TEST(MatrixXTensorTest, FactoryMethods) {
    // Test zeros
    Matrix<double> m1 = Matrix<double>::zeros(2, 3);
    EXPECT_EQ(m1.rows(), 2);
    EXPECT_EQ(m1.cols(), 3);
    EXPECT_EQ(m1(0, 0), 0.0);
    EXPECT_EQ(m1(0, 1), 0.0);
    EXPECT_EQ(m1(0, 2), 0.0);
    EXPECT_EQ(m1(1, 0), 0.0);
    EXPECT_EQ(m1(1, 1), 0.0);
    EXPECT_EQ(m1(1, 2), 0.0);
    
    // Test ones
    Matrix<double> m2 = Matrix<double>::ones(2, 3);
    EXPECT_EQ(m2.rows(), 2);
    EXPECT_EQ(m2.cols(), 3);
    EXPECT_EQ(m2(0, 0), 1.0);
    EXPECT_EQ(m2(0, 1), 1.0);
    EXPECT_EQ(m2(0, 2), 1.0);
    EXPECT_EQ(m2(1, 0), 1.0);
    EXPECT_EQ(m2(1, 1), 1.0);
    EXPECT_EQ(m2(1, 2), 1.0);
    
    // Test identity
    Matrix<double> m3 = Matrix<double>::identity(3);
    EXPECT_EQ(m3.rows(), 3);
    EXPECT_EQ(m3.cols(), 3);
    EXPECT_EQ(m3(0, 0), 1.0);
    EXPECT_EQ(m3(0, 1), 0.0);
    EXPECT_EQ(m3(0, 2), 0.0);
    EXPECT_EQ(m3(1, 0), 0.0);
    EXPECT_EQ(m3(1, 1), 1.0);
    EXPECT_EQ(m3(1, 2), 0.0);
    EXPECT_EQ(m3(2, 0), 0.0);
    EXPECT_EQ(m3(2, 1), 0.0);
    EXPECT_EQ(m3(2, 2), 1.0);
    
    // Test random
    Matrix<double> m4 = Matrix<double>::random(2, 3, 0.0, 1.0);
    EXPECT_EQ(m4.rows(), 2);
    EXPECT_EQ(m4.cols(), 3);
    
    // Check that values are within range
    for (size_t i = 0; i < m4.rows(); ++i) {
        for (size_t j = 0; j < m4.cols(); ++j) {
            EXPECT_GE(m4(i, j), 0.0);
            EXPECT_LE(m4(i, j), 1.0);
        }
    }
}

TEST(MatrixXTensorTest, NonMemberFunctions) {
    Matrix<double> m1 = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };
    
    // Test sum
    EXPECT_DOUBLE_EQ(sum(m1), 21.0);
    
    // Test mean
    EXPECT_DOUBLE_EQ(mean(m1), 3.5);
    
    // Test dot product
    Matrix<double> m2 = {
        {1.0, 2.0},
        {3.0, 4.0},
        {5.0, 6.0}
    };
    
    Matrix<double> m3 = dot(m1, m2);
    EXPECT_EQ(m3.rows(), 2);
    EXPECT_EQ(m3.cols(), 2);
    EXPECT_DOUBLE_EQ(m3(0, 0), 22.0);
    EXPECT_DOUBLE_EQ(m3(0, 1), 28.0);
    EXPECT_DOUBLE_EQ(m3(1, 0), 49.0);
    EXPECT_DOUBLE_EQ(m3(1, 1), 64.0);
}