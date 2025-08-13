#pragma once

#include <functional>
#include <memory>
#include <vector>
#include "tensor.hpp"

/**
 * @file autograd.hpp
 * @brief Modern automatic differentiation engine for deep learning
 * @author Kalenitid
 * @version 2.0.0
 * 
 * This implementation provides a comprehensive autograd system with:
 * - Dynamic computational graph construction
 * - Topological sorting for efficient backward pass
 * - Memory-efficient gradient computation
 * - Support for complex neural network architectures
 * - Gradient accumulation and checkpointing
 */

namespace utils {
    template<typename T>
    class Variable;

    /**
     * @brief Base class for all operations in the computational graph
     * 
     * This class represents a node in the computational graph that can perform
     * forward and backward computations. Each operation stores references to its
     * input variables and can save intermediate results for gradient computation.
     */
    template<typename T>
    class Function {
    public:
        using VariablePtr = std::shared_ptr<Variable<T> >;
        using TensorVec = std::vector<Tensor<T> >;
        using VariableVec = std::vector<VariablePtr>;

        virtual ~Function() = default;

        /**
         * @brief Forward pass computation
         * @param inputs Input variables
         * @return Output tensor
         */
        virtual Tensor<T> forward(const std::vector<Variable<T> > &inputs) = 0;

        /**
         * @brief Backward pass computation
         * @param grad_output Gradient flowing from the output
         * @return Gradients with respect to each input
         */
        virtual TensorVec backward(const Tensor<T> &grad_output) = 0;

        /**
         * @brief Set input variables for backward pass
         */
        void set_inputs(const VariableVec &inputs);

        /**
         * @brief Get input variables
         */
        const VariableVec &get_inputs() const;

        /**
         * @brief Get the number of inputs this function expects
         */
        virtual size_t num_inputs() const = 0;

        /**
         * @brief Check if this function needs input values for backward pass
         */
        virtual bool needs_input_grad(size_t input_idx) const;

    protected:
        /**
         * @brief Save tensors needed for backward pass
         */
        void save_for_backward(const TensorVec &tensors);

        /**
         * @brief Get saved tensors
         */
        const TensorVec &get_saved_tensors() const;

        TensorVec saved_tensors_;
        VariableVec input_variables_;
    };

    /**
     * @brief Variable class that supports automatic differentiation
     *
     * This class wraps tensors and tracks operations for automatic gradient computation.
     * It maintains the computational graph by storing references to the operations that
     * created it and can trigger backward pass computation.
     */
    template<typename T>
    class Variable : public std::enable_shared_from_this<Variable<T> > {
    public:
        using FunctionPtr = std::shared_ptr<Function<T> >;
        using VariablePtr = std::shared_ptr<Variable<T> >;

        /**
         * @brief Constructor from tensor data
         * @param data The tensor data
         * @param requires_grad Whether to compute gradients for this variable
         */
        explicit Variable(const Tensor<T> &data, bool requires_grad = false);

        /**
         * @brief Constructor with gradient function (for intermediate results)
         * @param data The tensor data
         * @param grad_fn The function that created this variable
         */
        Variable(const Tensor<T> &data, FunctionPtr grad_fn);

        /**
         * @brief Create a shared pointer to this variable
         */
        VariablePtr shared_from_this() const;

        // Getters and setters
        const Tensor<T> &data() const { return data_; }
        Tensor<T> &data() { return data_; }
        const Tensor<T> &grad() const { return grad_; }
        Tensor<T> &grad() { return grad_; }
        bool requires_grad() const { return requires_grad_; }
        FunctionPtr grad_fn() const { return grad_fn_; }

        /**
         * @brief Get the gradient function (if any)
         */
        FunctionPtr get_grad_fn() const { return grad_fn_; }

        /**
         * @brief Set requires_grad flag
         */
        void set_requires_grad(bool requires_grad);

        /**
         * @brief Perform backward pass with topological sorting
         * @param gradient Optional gradient to start with (defaults to ones)
         * @param retain_graph Whether to keep the computational graph after backward
         */
        void backward(const Tensor<T> &gradient = Tensor<T>(), bool retain_graph = false);

        /**
         * @brief Zero the gradients
         */
        void zero_grad();

        /**
         * @brief Detach from computational graph
         * @return A new variable with the same data but no gradient tracking
         */
        std::shared_ptr<Variable<T>> detach() const;

        /**
         * @brief Clone this variable with optional gradient requirement
         */
        std::shared_ptr<Variable<T>> clone(bool requires_grad = true) const;

        // Arithmetic operations
        std::shared_ptr<Variable<T>> operator+(const Variable<T> &other) const;

        std::shared_ptr<Variable<T>> operator-(const Variable<T> &other) const;

        std::shared_ptr<Variable<T>> operator*(const Variable<T> &other) const;

        std::shared_ptr<Variable<T>> operator/(const Variable<T> &other) const;

        // Scalar operations
        std::shared_ptr<Variable<T>> operator+(T scalar) const;
        std::shared_ptr<Variable<T>> operator-(T scalar) const;
        std::shared_ptr<Variable<T>> operator*(T scalar) const;
        std::shared_ptr<Variable<T>> operator/(T scalar) const;

        // Smart pointer operations
        static VariablePtr add(VariablePtr a, VariablePtr b);
        static VariablePtr sub(VariablePtr a, VariablePtr b);
        static VariablePtr mul(VariablePtr a, VariablePtr b);
        static VariablePtr div(VariablePtr a, VariablePtr b);
        static VariablePtr mul(VariablePtr a, T scalar);
        static VariablePtr add(VariablePtr a, T scalar);
        static VariablePtr sub(VariablePtr a, T scalar);
        static VariablePtr div(VariablePtr a, T scalar);

        // Tensor operations
        std::shared_ptr<Variable<T>> matmul(const std::shared_ptr<Variable<T>> &other) const;

        std::shared_ptr<Variable<T>> dot(const std::shared_ptr<Variable<T>> &other) const;

        std::shared_ptr<Variable<T>> transpose() const;

        std::shared_ptr<Variable<T>> transpose(const std::vector<size_t> &axes) const;

        std::shared_ptr<Variable<T>> reshape(const std::vector<size_t> &new_shape) const;

        std::shared_ptr<Variable<T>> view(const std::vector<size_t> &new_shape) const;

        std::shared_ptr<Variable<T>> squeeze(int axis = -1) const;

        std::shared_ptr<Variable<T>> unsqueeze(size_t axis) const;

        // Reduction operations
        std::shared_ptr<Variable<T>> sum() const;

        std::shared_ptr<Variable<T>> sum(const std::vector<int> &axes, bool keepdims = false) const;

        std::shared_ptr<Variable<T>> mean() const;

        std::shared_ptr<Variable<T>> mean(const std::vector<int> &axes, bool keepdims = false) const;

        std::shared_ptr<Variable<T>> max() const;

        std::shared_ptr<Variable<T>> min() const;

        // Activation functions
        std::shared_ptr<Variable<T>> sigmoid() const;

        std::shared_ptr<Variable<T>> tanh() const;

        std::shared_ptr<Variable<T>> relu() const;

        std::shared_ptr<Variable<T>> leaky_relu(T negative_slope = 0.01) const;

        std::shared_ptr<Variable<T>> gelu() const;

        std::shared_ptr<Variable<T>> softmax(int axis = -1) const;

        std::shared_ptr<Variable<T>> log_softmax(int axis = -1) const;

        // Mathematical functions
        std::shared_ptr<Variable<T>> exp() const;

        std::shared_ptr<Variable<T>> log() const;

        std::shared_ptr<Variable<T>> sqrt() const;

        std::shared_ptr<Variable<T>> pow(T exponent) const;

        std::shared_ptr<Variable<T>> abs() const;

        // Element access (variadic for n-dimensional tensors)
        template<typename... Args>
        T &operator()(Args... indices) { return data_(indices...); }

        template<typename... Args>
        const T &operator()(Args... indices) const { return data_(indices...); }

        // Shape and size information
        const std::vector<size_t> &shape() const { return data_.shape(); }
        size_t size() const { return data_.size(); }
        size_t ndim() const { return data_.shape().size(); }

        // Backward compatibility for 2D tensors
        size_t rows() const;

        size_t cols() const;

        /**
         * @brief Create a variable with gradient function and input references
         */
        static std::shared_ptr<Variable<T>> create_with_grad_fn(const Tensor<T> &data,
                                               FunctionPtr grad_fn,
                                               const std::vector<VariablePtr> &inputs);

        /**
         * @brief Check if any input requires gradients
         */
        static bool any_requires_grad(const std::vector<Variable<T> > &variables);

        /**
         * @brief Convert variables to shared pointers
         */
        static std::vector<VariablePtr> to_shared_ptrs(const std::vector<Variable<T> > &variables);

    private:
        Tensor<T> data_;
        Tensor<T> grad_;
        bool requires_grad_;
        FunctionPtr grad_fn_;

        /**
         * @brief Topological sort for backward pass
         */
        static std::vector<VariablePtr> topological_sort(VariablePtr root);
    };

    // Global operator overloads for smart pointers
    template<typename T>
    std::shared_ptr<Variable<T>> operator+(const std::shared_ptr<Variable<T>>& lhs, const std::shared_ptr<Variable<T>>& rhs);

    template<typename T>
    std::shared_ptr<Variable<T>> operator-(const std::shared_ptr<Variable<T>>& lhs, const std::shared_ptr<Variable<T>>& rhs);

    template<typename T>
    std::shared_ptr<Variable<T>> operator*(const std::shared_ptr<Variable<T>>& lhs, const std::shared_ptr<Variable<T>>& rhs);

    template<typename T>
    std::shared_ptr<Variable<T>> operator/(const std::shared_ptr<Variable<T>>& lhs, const std::shared_ptr<Variable<T>>& rhs);

    template<typename T>
    std::shared_ptr<Variable<T>> operator*(const std::shared_ptr<Variable<T>>& lhs, T scalar);

    template<typename T>
    std::shared_ptr<Variable<T>> operator*(T scalar, const std::shared_ptr<Variable<T>>& rhs);

    template<typename T>
    std::shared_ptr<Variable<T>> operator+(const std::shared_ptr<Variable<T>>& lhs, T scalar);

    template<typename T>
    std::shared_ptr<Variable<T>> operator+(T scalar, const std::shared_ptr<Variable<T>>& rhs);

    template<typename T>
    std::shared_ptr<Variable<T>> operator-(const std::shared_ptr<Variable<T>>& lhs, T scalar);

    template<typename T>
    std::shared_ptr<Variable<T>> operator-(T scalar, const std::shared_ptr<Variable<T>>& rhs);

    template<typename T>
    std::shared_ptr<Variable<T>> operator/(const std::shared_ptr<Variable<T>>& lhs, T scalar);

    template<typename T>
    std::shared_ptr<Variable<T>> operator/(T scalar, const std::shared_ptr<Variable<T>>& rhs);

    // ============================================================================
    // OPERATION IMPLEMENTATIONS
    // ============================================================================

    /**
     * @brief Addition operation (element-wise)
     */
    template<typename T>
    class AddFunction : public Function<T> {
    public:
        Tensor<T> forward(const std::vector<Variable<T> > &inputs) override;

        Function<T>::TensorVec backward(const Tensor<T> &grad_output) override;

        size_t num_inputs() const override;
    };

    /**
     * @brief Subtraction operation (element-wise)
     */
    template<typename T>
    class SubFunction : public Function<T> {
    public:
        Tensor<T> forward(const std::vector<Variable<T> > &inputs) override;

        Function<T>::TensorVec backward(const Tensor<T> &grad_output) override;

        size_t num_inputs() const override;
    };

    /**
     * @brief Element-wise multiplication operation
     */
    template<typename T>
    class MulFunction : public Function<T> {
    public:
        Tensor<T> forward(const std::vector<Variable<T> > &inputs) override;

        Function<T>::TensorVec backward(const Tensor<T> &grad_output) override;

        size_t num_inputs() const override;
    };

    /**
     * @brief Division operation (element-wise)
     */
    template<typename T>
    class DivFunction : public Function<T> {
    public:
        Tensor<T> forward(const std::vector<Variable<T> > &inputs) override;

        Function<T>::TensorVec backward(const Tensor<T> &grad_output) override;

        size_t num_inputs() const override;
    };

    /**
     * @brief Matrix multiplication operation
     */
    template<typename T>
    class MatMulFunction : public Function<T> {
    public:
        Tensor<T> forward(const std::vector<Variable<T> > &inputs) override;

        Function<T>::TensorVec backward(const Tensor<T> &grad_output) override;

        size_t num_inputs() const override;
    };

    /**
     * @brief Dot product operation (for compatibility)
     */
    template<typename T>
    class DotFunction : public Function<T> {
    public:
        Tensor<T> forward(const std::vector<Variable<T> > &inputs) override;

        Function<T>::TensorVec backward(const Tensor<T> &grad_output) override;

        size_t num_inputs() const override;
    };

    /**
     * @brief Transpose operation
     */
    template<typename T>
    class TransposeFunction : public Function<T> {
    public:
        Tensor<T> forward(const std::vector<Variable<T> > &inputs) override;

        Function<T>::TensorVec backward(const Tensor<T> &grad_output) override;

        size_t num_inputs() const override;
    };

    /**
     * @brief Transpose operation with specified axes
     */
    template<typename T>
    class TransposeAxesFunction : public Function<T> {
    public:
        explicit TransposeAxesFunction(const std::vector<size_t> &axes);

        Tensor<T> forward(const std::vector<Variable<T> > &inputs) override;

        Function<T>::TensorVec backward(const Tensor<T> &grad_output) override;

        size_t num_inputs() const override;

    private:
        std::vector<size_t> axes_;
        std::vector<size_t> inverse_axes_;
    };

    /**
     * @brief View operation (reshape without copying)
     */
    template<typename T>
    class ViewFunction : public Function<T> {
    public:
        explicit ViewFunction(const std::vector<size_t> &original_shape);

        Tensor<T> forward(const std::vector<Variable<T> > &inputs) override;

        Function<T>::TensorVec backward(const Tensor<T> &grad_output) override;

        size_t num_inputs() const override;

    private:
        std::vector<size_t> original_shape_;
    };

    /**
     * @brief Squeeze operation (remove dimensions of size 1)
     */
    template<typename T>
    class SqueezeFunction : public Function<T> {
    public:
        explicit SqueezeFunction(int axis, const std::vector<size_t> &original_shape);

        Tensor<T> forward(const std::vector<Variable<T> > &inputs) override;

        Function<T>::TensorVec backward(const Tensor<T> &grad_output) override;

        size_t num_inputs() const override;

    private:
        int axis_;
        std::vector<size_t> original_shape_;
    };

    /**
     * @brief Unsqueeze operation (add dimension of size 1)
     */
    template<typename T>
    class UnsqueezeFunction : public Function<T> {
    public:
        explicit UnsqueezeFunction(size_t axis);

        Tensor<T> forward(const std::vector<Variable<T> > &inputs) override;

        Function<T>::TensorVec backward(const Tensor<T> &grad_output) override;

        size_t num_inputs() const override;

    private:
        size_t axis_;
    };

    /**
     * @brief Sum reduction along specified axes
     */
    template<typename T>
    class SumAxesFunction : public Function<T> {
    public:
        explicit SumAxesFunction(const std::vector<int> &axes, bool keepdims);

        Tensor<T> forward(const std::vector<Variable<T> > &inputs) override;

        Function<T>::TensorVec backward(const Tensor<T> &grad_output) override;

        size_t num_inputs() const override;

    private:
        std::vector<int> axes_;
        bool keepdims_;
        std::vector<size_t> input_shape_;
    };

    /**
     * @brief Mean reduction along specified axes
     */
    template<typename T>
    class MeanAxesFunction : public Function<T> {
    public:
        explicit MeanAxesFunction(const std::vector<int> &axes, bool keepdims);

        Tensor<T> forward(const std::vector<Variable<T> > &inputs) override;

        Function<T>::TensorVec backward(const Tensor<T> &grad_output) override;

        size_t num_inputs() const override;

    private:
        std::vector<int> axes_;
        bool keepdims_;
        std::vector<size_t> input_shape_;
        size_t reduction_size_;
    };

    /**
     * @brief Reshape operation
     */
    template<typename T>
    class ReshapeFunction : public Function<T> {
    public:
        explicit ReshapeFunction(const std::vector<size_t> &original_shape);

        Tensor<T> forward(const std::vector<Variable<T> > &inputs) override;

        Function<T>::TensorVec backward(const Tensor<T> &grad_output) override;

        size_t num_inputs() const override;

    private:
        std::vector<size_t> original_shape_;
    };

    /**
     * @brief Sigmoid activation function
     */
    template<typename T>
    class SigmoidFunction : public Function<T> {
    public:
        Tensor<T> forward(const std::vector<Variable<T> > &inputs) override;

        Function<T>::TensorVec backward(const Tensor<T> &grad_output) override;

        size_t num_inputs() const override;
    };

    /**
     * @brief ReLU activation function
     */
    template<typename T>
    class ReLUFunction : public Function<T> {
    public:
        Tensor<T> forward(const std::vector<Variable<T> > &inputs) override;

        Function<T>::TensorVec backward(const Tensor<T> &grad_output) override;

        size_t num_inputs() const override;
    };

    /**
     * @brief Exponential function
     */
    template<typename T>
    class ExpFunction : public Function<T> {
    public:
        Tensor<T> forward(const std::vector<Variable<T> > &inputs) override;

        Function<T>::TensorVec backward(const Tensor<T> &grad_output) override;

        size_t num_inputs() const override;
    };

    /**
     * @brief Sum reduction function
     */
    template<typename T>
    class SumFunction : public Function<T> {
    public:
        Tensor<T> forward(const std::vector<Variable<T> > &inputs) override;

        Function<T>::TensorVec backward(const Tensor<T> &grad_output) override;

        size_t num_inputs() const override;
    };

    /**
     * @brief Mean function for automatic differentiation
     */
    template<typename T>
    class MeanFunction : public Function<T> {
    public:
        Tensor<T> forward(const std::vector<Variable<T> > &inputs) override;

        Function<T>::TensorVec backward(const Tensor<T> &grad_output) override;

        size_t num_inputs() const override;
    };

    /**
     * @brief Tanh function for automatic differentiation
     */
    template<typename T>
    class TanhFunction : public Function<T> {
    public:
        Tensor<T> forward(const std::vector<Variable<T> > &inputs) override;

        Function<T>::TensorVec backward(const Tensor<T> &grad_output) override;

        size_t num_inputs() const override;
    };

    /**
     * @brief Log function for automatic differentiation
     */
    template<typename T>
    class LogFunction : public Function<T> {
    public:
        Tensor<T> forward(const std::vector<Variable<T> > &inputs) override;

        Function<T>::TensorVec backward(const Tensor<T> &grad_output) override;

        size_t num_inputs() const override;
    };


    // Type aliases
    using VariableF = Variable<float>;
    using VariableD = Variable<double>;

    // Helper functions for creating Variables on heap
    
    /**
     * @brief Create a shared pointer to a Variable (similar to std::make_shared)
     * @tparam T The data type (float, double, etc.)
     * @param data The tensor data
     * @param requires_grad Whether gradients should be computed for this variable
     * @return Shared pointer to the created Variable
     */
    template<typename T>
    std::shared_ptr<Variable<T>> make_variable(const Tensor<T>& data, bool requires_grad = false) {
        return std::make_shared<Variable<T>>(data, requires_grad);
    }

    /**
     * @brief Create a shared pointer to a Variable with gradient function
     * @tparam T The data type (float, double, etc.)
     * @param data The tensor data
     * @param grad_fn The gradient function
     * @return Shared pointer to the created Variable
     */
    template<typename T>
    std::shared_ptr<Variable<T>> make_variable(const Tensor<T>& data, std::shared_ptr<Function<T>> grad_fn) {
        return std::make_shared<Variable<T>>(data, grad_fn);
    }

    /**
     * @brief Create a shared pointer to a Variable from scalar value
     * @tparam T The data type (float, double, etc.)
     * @param value The scalar value
     * @param requires_grad Whether gradients should be computed for this variable
     * @return Shared pointer to the created Variable
     */
    template<typename T>
    std::shared_ptr<Variable<T>> make_variable_scalar(T value, bool requires_grad = false) {
        return std::make_shared<Variable<T>>(Tensor<T>(value), requires_grad);
    }

    /**
     * @brief Create a shared pointer to a Variable from shape (zeros)
     * @tparam T The data type (float, double, etc.)
     * @param shape The shape of the tensor
     * @param requires_grad Whether gradients should be computed for this variable
     * @return Shared pointer to the created Variable
     */
    template<typename T>
    std::shared_ptr<Variable<T>> make_variable_zeros(const std::vector<size_t>& shape, bool requires_grad = false) {
        return std::make_shared<Variable<T>>(Tensor<T>::zeros(shape), requires_grad);
    }

    /**
     * @brief Create a shared pointer to a Variable from shape (ones)
     * @tparam T The data type (float, double, etc.)
     * @param shape The shape of the tensor
     * @param requires_grad Whether gradients should be computed for this variable
     * @return Shared pointer to the created Variable
     */
    template<typename T>
    std::shared_ptr<Variable<T>> make_variable_ones(const std::vector<size_t>& shape, bool requires_grad = false) {
        return std::make_shared<Variable<T>>(Tensor<T>::ones(shape), requires_grad);
    }

    // Convenient type aliases for shared pointers
    using VariableFPtr = std::shared_ptr<VariableF>;
    using VariableDPtr = std::shared_ptr<VariableD>;

} // namespace utils
