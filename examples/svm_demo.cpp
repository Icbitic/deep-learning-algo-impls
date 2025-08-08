/**
 * @file svm_demo.cpp
 * @brief Demonstration of SVM with PyTorch-like automatic differentiation
 * @author Kalenitid
 * @version 1.0.0
 * 
 * This example demonstrates the difference between traditional SVM implementation
 * with static computation and the new autograd-based implementation that uses
 * automatic differentiation for gradient computation.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include "ml/svm.hpp"
#include "utils/tensor.hpp"
#include "utils/autograd.hpp"

using namespace utils;
using namespace ml;

void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "  " << title << std::endl;
    std::cout << std::string(60, '=') << std::endl;
}

void demonstrate_autograd_basics() {
    print_separator("AUTOGRAD BASICS DEMONSTRATION");
    
    std::cout << "\n1. Creating Variables with autograd tracking:" << std::endl;
    
    // Create matrices
    Tensor<double> data1({{1.0, 2.0}, {3.0, 4.0}});
    Tensor<double> data2({{0.5, 1.5}, {2.5, 3.5}});
    
    // Create Variables (PyTorch-like tensors with autograd)
    Variable<double> var1(data1, true);  // requires_grad = true
    Variable<double> var2(data2, true);
    
    std::cout << "Variable 1 (requires_grad=true):" << std::endl;
    std::cout << var1.data() << std::endl;
    
    std::cout << "Variable 2 (requires_grad=true):" << std::endl;
    std::cout << var2.data() << std::endl;
    
    std::cout << "\n2. Forward pass with automatic graph construction:" << std::endl;
    
    // Perform operations (builds computational graph automatically)
    Variable<double> sum_result = var1 + var2;
    Variable<double> product_result = var1 * var2;
    Variable<double> final_result = sum_result.dot(product_result.transpose());
    
    std::cout << "Sum result:" << std::endl;
    std::cout << sum_result.data() << std::endl;
    
    std::cout << "Product result:" << std::endl;
    std::cout << product_result.data() << std::endl;
    
    std::cout << "Final result (sum.dot(product.T)):" << std::endl;
    std::cout << final_result.data() << std::endl;
    
    std::cout << "\n3. Backward pass (automatic gradient computation):" << std::endl;
    
    // Compute gradients automatically
    final_result.backward();
    
    std::cout << "Gradients computed automatically!" << std::endl;
    std::cout << "This is similar to PyTorch's autograd engine." << std::endl;
}

void compare_svm_implementations() {
    print_separator("SVM IMPLEMENTATION COMPARISON");
    
    // Create a linearly separable dataset
    Tensor<double> X({
        {2.0, 3.0}, {3.0, 3.0}, {1.0, 2.0}, {2.0, 1.0},  // Class +1
        {-2.0, -1.0}, {-1.0, -2.0}, {-3.0, -2.0}, {-2.0, -3.0}  // Class -1
    });
    std::vector<int> y = {1, 1, 1, 1, -1, -1, -1, -1};
    
    std::cout << "\nDataset: 8 samples, 2 features, 2 classes (linearly separable)" << std::endl;
    std::cout << "Features (first 4 are class +1, last 4 are class -1):" << std::endl;
    std::cout << X << std::endl;
    
    // ========================================
    // SVM (Automatic Differentiation)
    // ========================================
    std::cout << "\n1. SVM (Automatic Differentiation):" << std::endl;
    std::cout << "   Method: Gradient Descent with Autograd" << std::endl;
    std::cout << "   Gradient computation: Automatic Differentiation" << std::endl;
    
    SVM<double> autograd_svm(KernelType::LINEAR, 1.0, 1.0, 3, 0.0, 1e-4, 500, 0.02);
    
    auto start_autograd = std::chrono::high_resolution_clock::now();
    autograd_svm.fit(X, y);
    auto end_autograd = std::chrono::high_resolution_clock::now();
    auto duration_autograd = std::chrono::duration_cast<std::chrono::microseconds>(end_autograd - start_autograd);
    
    std::vector<int> autograd_predictions = autograd_svm.predict(X);
    
    // Calculate accuracy
    int autograd_correct = 0;
    for (size_t i = 0; i < y.size(); ++i) {
        if (autograd_predictions[i] == y[i]) autograd_correct++;
    }
    double autograd_accuracy = static_cast<double>(autograd_correct) / y.size();
    
    std::cout << "   Training time: " << duration_autograd.count() << " μs" << std::endl;
    std::cout << "   Accuracy: " << std::fixed << std::setprecision(1) << autograd_accuracy * 100 << "%" << std::endl;
    
    // Show training progress
    std::vector<double> loss_history = autograd_svm.loss_history();
    if (loss_history.size() > 0) {
        std::cout << "   Initial loss: " << std::fixed << std::setprecision(4) << loss_history[0] << std::endl;
        std::cout << "   Final loss: " << loss_history.back() << std::endl;
        std::cout << "   Training iterations: " << loss_history.size() << std::endl;
        
        // Show loss reduction
        if (loss_history.size() > 1) {
            double loss_reduction = ((loss_history[0] - loss_history.back()) / loss_history[0]) * 100;
            std::cout << "   Loss reduction: " << std::fixed << std::setprecision(1) << loss_reduction << "%" << std::endl;
        }
    }
    
    // ========================================
    // RESULTS SUMMARY
    // ========================================
    std::cout << "\n2. RESULTS SUMMARY:" << std::endl;
    
    std::cout << "\n   Predictions:" << std::endl;
    std::cout << "   Sample | True | Predicted" << std::endl;
    std::cout << "   -------|------|----------" << std::endl;
    for (size_t i = 0; i < y.size(); ++i) {
        std::cout << "   " << std::setw(6) << i+1 << " | "
                  << std::setw(4) << y[i] << " | "
                  << std::setw(8) << autograd_predictions[i] << std::endl;
    }
    
    std::cout << "\n   Performance:" << std::endl;
    std::cout << "   Accuracy: " << std::fixed << std::setprecision(1) << autograd_accuracy * 100 << "%" << std::endl;
    std::cout << "   Training time: " << duration_autograd.count() << " μs" << std::endl;
}

void demonstrate_kernel_autograd() {
    print_separator("KERNEL METHODS WITH AUTOGRAD");
    
    // Create a non-linearly separable dataset (XOR-like)
    Tensor<double> X_nonlinear({
        {1.0, 1.0}, {-1.0, -1.0},    // Class +1
        {1.0, -1.0}, {-1.0, 1.0}     // Class -1
    });
    std::vector<int> y_nonlinear = {1, 1, -1, -1};
    
    std::cout << "\nNon-linearly separable dataset (XOR-like):" << std::endl;
    std::cout << X_nonlinear << std::endl;
    std::cout << "Labels: [1, 1, -1, -1]" << std::endl;
    
    // Test different kernels with autograd
    std::vector<std::pair<KernelType, std::string>> kernels = {
        {KernelType::LINEAR, "Linear"},
        {KernelType::RBF, "RBF (Radial Basis Function)"},
        {KernelType::POLYNOMIAL, "Polynomial"},
        {KernelType::SIGMOID, "Sigmoid"}
    };
    
    std::cout << "\nTesting different kernels with autograd:" << std::endl;
    
    for (const auto& [kernel_type, kernel_name] : kernels) {
        std::cout << "\n" << kernel_name << " Kernel:" << std::endl;
        
        SVM<double> svm(kernel_type, 1.0, 1.0, 2, 0.0, 1e-4, 200, 0.05);
        
        auto start = std::chrono::high_resolution_clock::now();
        svm.fit(X_nonlinear, y_nonlinear);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::vector<int> predictions = svm.predict(X_nonlinear);
        
        int correct = 0;
        for (size_t i = 0; i < y_nonlinear.size(); ++i) {
            if (predictions[i] == y_nonlinear[i]) correct++;
        }
        double accuracy = static_cast<double>(correct) / y_nonlinear.size();
        
        std::cout << "   Accuracy: " << std::fixed << std::setprecision(1) << accuracy * 100 << "%" << std::endl;
        std::cout << "   Training time: " << duration.count() << " μs" << std::endl;
        
        // Show final loss
        std::vector<double> loss_history = svm.loss_history();
        if (!loss_history.empty()) {
            std::cout << "   Final loss: " << std::fixed << std::setprecision(4) << loss_history.back() << std::endl;
        }
    }
}

int main() {
    std::cout << "SVM with PyTorch-like Automatic Differentiation Demo" << std::endl;
    std::cout << "===================================================" << std::endl;
    
    try {
        // Demonstrate autograd basics
        demonstrate_autograd_basics();
        
        // Compare SVM implementations
        compare_svm_implementations();
        
        // Demonstrate kernel methods with autograd
        demonstrate_kernel_autograd();
        
        print_separator("DEMO COMPLETED SUCCESSFULLY");
        std::cout << "\nThe autograd-based SVM demonstrates how modern automatic" << std::endl;
        std::cout << "differentiation can be applied to traditional ML algorithms," << std::endl;
        std::cout << "providing more flexibility and easier experimentation." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}