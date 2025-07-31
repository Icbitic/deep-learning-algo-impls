#pragma once

/**
 * @file ml.hpp
 * @brief Main header file for traditional machine learning algorithms
 * @author Kalenitid
 * @version 1.0.0
 *
 * This header provides access to all traditional machine learning algorithms
 * implemented in the ml namespace, including:
 * - Principal Component Analysis (PCA)
 * - K-Means Clustering
 * - Support Vector Machine (SVM)
 */

#include "kmeans.hpp"
#include "pca.hpp"
#include "svm.hpp"

/**
 * @namespace ml
 * @brief Namespace containing traditional machine learning algorithms
 *
 * This namespace provides implementations of classical machine learning
 * algorithms that are commonly used for data preprocessing, dimensionality
 * reduction, clustering, and classification tasks.
 *
 * The algorithms are designed to work seamlessly with the matrix utilities
 * provided in the utils namespace.
 *
 * @example
 * ```cpp
 * #include "ml/ml.hpp"
 *
 * using namespace ml;
 * using namespace utils;
 *
 * // PCA for dimensionality reduction
 * PCA<double> pca;
 * Matrix<double> data = \/* your data *\/;
 * pca.fit(data);
 * Matrix<double> reduced = pca.transform(data, 2);
 *
 * // K-Means clustering
 * KMeans<double> kmeans(3);
 * std::vector<int> labels = kmeans.fit_predict(data);
 *
 * // SVM classification
 * SVM<double> svm(KernelType::RBF);
 * std::vector<int> targets = \/* your labels *\/;
 * svm.fit(data, targets);
 * std::vector<int> predictions = svm.predict(test_data);
 * ```
 */

namespace ml {
    // All algorithm classes are already defined in their respective headers
    // This file serves as a convenient single include point
}
