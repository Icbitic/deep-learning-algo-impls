#include "utils/autograd.hpp"
#include <algorithm>
#include <cmath>
#include <queue>
#include <unordered_set>

namespace dl {
    template<typename T>
    std::vector<typename Variable<T>::VariablePtr> Variable<T>::topological_sort(VariablePtr root) {
        std::vector<VariablePtr> topo_order;
        std::unordered_set<VariablePtr> visited;
        std::unordered_set<VariablePtr> temp_visited;

        std::function<void(const VariablePtr &)> visit = [&](const VariablePtr &var) {
            if (temp_visited.count(var)) {
                throw std::runtime_error("Cycle detected in computational graph");
            }
            if (visited.count(var)) {
                return;
            }

            temp_visited.insert(var);

            if (var->get_grad_fn()) {
                for (const auto &input: var->get_grad_fn()->get_inputs()) {
                    if (input && input->requires_grad()) {
                        visit(input);
                    }
                }
            }
            // Always include variables that require gradients in the topological order,
            // even if they are leaf variables (no grad_fn)

            temp_visited.erase(var);
            visited.insert(var);
            topo_order.push_back(var);
        };

        visit(root);
        std::reverse(topo_order.begin(), topo_order.end());
        return topo_order;
    }

    template<typename T>
    void Variable<T>::backward(const Tensor<T> &gradient, bool retain_graph) {
        if (!requires_grad_) {
            return;
        }

        // Initialize gradient if empty or default-constructed
        Tensor<T> grad;

        // Check if gradient is default-constructed (has default value but size > 0)
        bool is_default_gradient = false;
        if (!gradient.empty() && gradient.size() > 0) {
            // Check if all values are zero (indicating default construction)
            auto grad_data = gradient.data();
            bool all_zero = true;
            for (size_t i = 0; i < grad_data.size(); ++i) {
                if (grad_data.flat(i) != T(0)) {
                    all_zero = false;
                    break;
                }
            }
            is_default_gradient = all_zero;
        }

        if (gradient.empty() || gradient.size() == 0 || is_default_gradient) {
            // For scalar tensors (0-dimensional), create a scalar gradient of 1.0
            if (data_.shape().empty()) {
                grad = Tensor<T>(T(1.0));
            } else {
                grad = Tensor<T>::ones(data_.shape());
            }
        } else {
            grad = gradient;
        }

        // Get topological order for proper gradient computation
        VariablePtr self_ptr;
        try {
            self_ptr = shared_from_this();
        } catch (const std::bad_weak_ptr &) {
            // For stack-allocated Variables, create a temporary shared_ptr
            self_ptr = std::shared_ptr<Variable<T> >(this, [](Variable<T> *) {
            });
        }
        auto topo_order = Variable<T>::topological_sort(self_ptr);

        // Initialize gradients map
        std::unordered_map<VariablePtr, Tensor<T> > gradients;
        gradients[self_ptr] = grad;

        // Backward pass in topological order
        for (const auto &var: topo_order) {
            if (!var->requires_grad() || !gradients.count(var)) {
                continue;
            }

            // Accumulate gradient
            if (var->grad_.empty()) {
                var->grad_ = gradients[var];
            } else {
                // Handle shape mismatch: if existing grad is scalar and new grad has shape,
                // or vice versa, use the one with proper shape
                if (var->grad_.shape().empty() && !gradients[var].shape().empty()) {
                    // Existing grad is scalar, new grad has shape - use new grad
                    var->grad_ = gradients[var];
                } else if (!var->grad_.shape().empty() && gradients[var].shape().empty()) {
                    // Existing grad has shape, new grad is scalar - keep existing
                    // (this shouldn't happen in normal backprop)
                } else {
                    // Both have same dimensionality, normal addition
                    var->grad_ = var->grad_ + gradients[var];
                }
            }

            // Propagate gradients if this variable has a gradient function
            if (var->get_grad_fn()) {
                auto input_grads = var->get_grad_fn()->backward(gradients[var]);
                const auto &input_vars = var->get_grad_fn()->get_inputs();

                // Handle gradient accumulation properly for repeated variables
                std::unordered_map<VariablePtr, Tensor<T> > local_grads;

                for (size_t i = 0; i < input_grads.size() && i < input_vars.size(); ++i) {
                    if (input_vars[i] && input_vars[i]->requires_grad()) {
                        if (local_grads.count(input_vars[i])) {
                            // Same variable appears multiple times, accumulate locally first
                            local_grads[input_vars[i]] = local_grads[input_vars[i]] + input_grads[i];
                        } else {
                            local_grads[input_vars[i]] = input_grads[i];
                        }
                    }
                }

                // Now add the accumulated local gradients to the global gradients map
                for (const auto &[input_var, local_grad]: local_grads) {
                    if (gradients.count(input_var)) {
                        gradients[input_var] = gradients[input_var] + local_grad;
                    } else {
                        gradients[input_var] = local_grad;
                    }
                }
            }
        }

        // Clear computational graph if not retaining
        if (!retain_graph) {
            for (const auto &var: topo_order) {
                var->grad_fn_.reset();
            }
        }
    }

    template<typename T>
    std::shared_ptr<Variable<T> > Variable<T>::operator+(const Variable<T> &other) const {
        auto add_fn = std::make_shared<AddFunction<T> >();
        Tensor<T> result = add_fn->forward({*this, other});

        if (requires_grad_ || other.requires_grad_) {
            // Create shared_ptr that stores references to original Variables for gradient accumulation
            auto self_ptr = std::shared_ptr<Variable<T> >(const_cast<Variable<T> *>(this), [](Variable<T> *) {
            });
            auto other_ptr = std::shared_ptr<Variable<T> >(const_cast<Variable<T> *>(&other), [](Variable<T> *) {
            });
            auto result_var = Variable<T>::create_with_grad_fn(result, add_fn, {self_ptr, other_ptr});
            return result_var;
        }
        return std::make_shared<Variable<T> >(result, false);
    }

    template<typename T>
    std::shared_ptr<Variable<T> > Variable<T>::operator-(const Variable<T> &other) const {
        auto sub_fn = std::make_shared<SubFunction<T> >();
        Tensor<T> result = sub_fn->forward({*this, other});

        if (requires_grad_ || other.requires_grad_) {
            auto self_ptr = shared_from_this();
            auto other_ptr = std::make_shared<Variable<T> >(other);
            auto result_var = Variable<T>::create_with_grad_fn(result, sub_fn, {self_ptr, other_ptr});
            return result_var;
        }
        return std::make_shared<Variable<T> >(result, false);
    }

    template<typename T>
    std::shared_ptr<Variable<T> > Variable<T>::operator*(const Variable<T> &other) const {
        auto mul_fn = std::make_shared<MulFunction<T> >();
        Tensor<T> result = mul_fn->forward({*this, other});

        if (requires_grad_ || other.requires_grad_) {
            auto self_ptr = shared_from_this();
            auto other_ptr = std::shared_ptr<Variable<T> >(const_cast<Variable<T> *>(&other), [](Variable<T> *) {
            });
            auto result_var = Variable<T>::create_with_grad_fn(result, mul_fn, {self_ptr, other_ptr});
            return result_var;
        }
        return std::make_shared<Variable<T> >(result, false);
    }

    template<typename T>
    std::shared_ptr<Variable<T> > Variable<T>::operator/(const Variable<T> &other) const {
        auto div_fn = std::make_shared<DivFunction<T> >();
        Tensor<T> result = div_fn->forward({*this, other});

        if (requires_grad_ || other.requires_grad_) {
            auto self_ptr = shared_from_this();
            auto other_ptr = std::shared_ptr<Variable<T> >(const_cast<Variable<T> *>(&other), [](Variable<T> *) {
            });
            auto result_var = Variable<T>::create_with_grad_fn(result, div_fn, {self_ptr, other_ptr});
            return result_var;
        }
        return std::make_shared<Variable<T> >(result, false);
    }

    template<typename T>
    std::shared_ptr<Variable<T> > Variable<T>::operator+(T scalar) const {
        auto scalar_fn = std::make_shared<AddFunction<T> >();
        Tensor<T> scalar_tensor = Tensor<T>::full(data_.shape(), scalar);
        Variable<T> scalar_var(scalar_tensor, false);
        Tensor<T> result = scalar_fn->forward({*this, scalar_var});

        if (requires_grad_) {
            auto self_ptr = shared_from_this();
            auto scalar_ptr = std::make_shared<Variable<T> >(scalar_var);
            auto result_var = Variable<T>::create_with_grad_fn(result, scalar_fn, {self_ptr, scalar_ptr});
            return result_var;
        }
        return std::make_shared<Variable<T> >(result, false);
    }

    template<typename T>
    std::shared_ptr<Variable<T> > Variable<T>::operator-(T scalar) const {
        auto scalar_fn = std::make_shared<SubFunction<T> >();
        Tensor<T> scalar_tensor = Tensor<T>::full(data_.shape(), scalar);
        Variable<T> scalar_var(scalar_tensor, false);
        Tensor<T> result = scalar_fn->forward({*this, scalar_var});

        if (requires_grad_) {
            auto self_ptr = shared_from_this();
            auto scalar_ptr = std::make_shared<Variable<T> >(scalar_var);
            auto result_var = Variable<T>::create_with_grad_fn(result, scalar_fn, {self_ptr, scalar_ptr});
            return result_var;
        }
        return std::make_shared<Variable<T> >(result, false);
    }

    template<typename T>
    std::shared_ptr<Variable<T> > Variable<T>::operator*(T scalar) const {
        auto scalar_fn = std::make_shared<MulFunction<T> >();
        Tensor<T> scalar_tensor = Tensor<T>::full(data_.shape(), scalar);
        Variable<T> scalar_var(scalar_tensor, false);
        Tensor<T> result = scalar_fn->forward({*this, scalar_var});

        if (requires_grad_) {
            auto self_ptr = shared_from_this();
            auto scalar_ptr = std::make_shared<Variable<T> >(scalar_var);
            auto result_var = Variable<T>::create_with_grad_fn(result, scalar_fn, {self_ptr, scalar_ptr});
            return result_var;
        }
        return std::make_shared<Variable<T> >(result, false);
    }

    template<typename T>
    std::shared_ptr<Variable<T> > Variable<T>::operator/(T scalar) const {
        auto scalar_fn = std::make_shared<DivFunction<T> >();
        Tensor<T> scalar_tensor = Tensor<T>::full(data_.shape(), scalar);
        Variable<T> scalar_var(scalar_tensor, false);
        Tensor<T> result = scalar_fn->forward({*this, scalar_var});

        if (requires_grad_) {
            auto self_ptr = shared_from_this();
            auto scalar_ptr = std::make_shared<Variable<T> >(scalar_var);
            auto result_var = Variable<T>::create_with_grad_fn(result, scalar_fn, {self_ptr, scalar_ptr});
            return result_var;
        }
        return std::make_shared<Variable<T> >(result, false);
    }

    // Smart pointer operations
    template<typename T>
    Variable<T>::VariablePtr Variable<T>::add(VariablePtr a, VariablePtr b) {
        auto add_fn = std::make_shared<AddFunction<T> >();
        Tensor<T> result = add_fn->forward({*a, *b});

        if (a->requires_grad() || b->requires_grad()) {
            auto result_var = Variable<T>::create_with_grad_fn(result, add_fn, {a, b});
            return result_var;
        }
        return std::make_shared<Variable<T> >(result, false);
    }

    template<typename T>
    Variable<T>::VariablePtr Variable<T>::sub(VariablePtr a, VariablePtr b) {
        auto sub_fn = std::make_shared<SubFunction<T> >();
        Tensor<T> result = sub_fn->forward({*a, *b});

        if (a->requires_grad() || b->requires_grad()) {
            auto result_var = Variable<T>::create_with_grad_fn(result, sub_fn, {a, b});
            return result_var;
        }
        return std::make_shared<Variable<T> >(result, false);
    }

    template<typename T>
    Variable<T>::VariablePtr Variable<T>::mul(VariablePtr a, VariablePtr b) {
        auto mul_fn = std::make_shared<MulFunction<T> >();
        Tensor<T> result = mul_fn->forward({*a, *b});

        if (a->requires_grad() || b->requires_grad()) {
            auto result_var = Variable<T>::create_with_grad_fn(result, mul_fn, {a, b});
            return result_var;
        }
        return std::make_shared<Variable<T> >(result, false);
    }

    template<typename T>
    Variable<T>::VariablePtr Variable<T>::div(VariablePtr a, VariablePtr b) {
        auto div_fn = std::make_shared<DivFunction<T> >();
        Tensor<T> result = div_fn->forward({*a, *b});

        if (a->requires_grad() || b->requires_grad()) {
            auto result_var = Variable<T>::create_with_grad_fn(result, div_fn, {a, b});
            return result_var;
        }
        return std::make_shared<Variable<T> >(result, false);
    }

    template<typename T>
    Variable<T>::VariablePtr Variable<T>::mul(VariablePtr a, T scalar) {
        auto scalar_mul_fn = std::make_shared<ScalarMulFunction<T> >(scalar);
        Tensor<T> result = scalar_mul_fn->forward({*a});

        if (a->requires_grad()) {
            auto result_var = Variable<T>::create_with_grad_fn(result, scalar_mul_fn, {a});
            return result_var;
        }
        return std::make_shared<Variable<T> >(result, false);
    }

    template<typename T>
    Variable<T>::VariablePtr Variable<T>::add(VariablePtr a, T scalar) {
        auto scalar_add_fn = std::make_shared<ScalarAddFunction<T> >(scalar);
        Tensor<T> result = scalar_add_fn->forward({*a});

        if (a->requires_grad()) {
            auto result_var = Variable<T>::create_with_grad_fn(result, scalar_add_fn, {a});
            return result_var;
        }
        return std::make_shared<Variable<T> >(result, false);
    }

    template<typename T>
    Variable<T>::VariablePtr Variable<T>::sub(VariablePtr a, T scalar) {
        auto scalar_sub_fn = std::make_shared<ScalarSubFunction<T> >(scalar);
        Tensor<T> result = scalar_sub_fn->forward({*a});

        if (a->requires_grad()) {
            auto result_var = Variable<T>::create_with_grad_fn(result, scalar_sub_fn, {a});
            return result_var;
        }
        return std::make_shared<Variable<T> >(result, false);
    }

    template<typename T>
    Variable<T>::VariablePtr Variable<T>::div(VariablePtr a, T scalar) {
        auto scalar_div_fn = std::make_shared<ScalarDivFunction<T> >(scalar);
        Tensor<T> result = scalar_div_fn->forward({*a});

        if (a->requires_grad()) {
            auto result_var = Variable<T>::create_with_grad_fn(result, scalar_div_fn, {a});
            return result_var;
        }
        return std::make_shared<Variable<T> >(result, false);
    }

    template<typename T>
    std::shared_ptr<Variable<T> > Variable<T>::matmul(const std::shared_ptr<Variable<T> > &other) const {
        auto matmul_fn = std::make_shared<MatMulFunction<T> >();
        Tensor<T> result = matmul_fn->forward({*this, *other});

        if (requires_grad_ || other->requires_grad_) {
            auto self_ptr = shared_from_this();
            auto result_var = Variable<T>::create_with_grad_fn(result, matmul_fn, {self_ptr, other});
            return result_var;
        }
        return std::make_shared<Variable<T> >(result, false);
    }

    template<typename T>
    std::shared_ptr<Variable<T> > Variable<T>::dot(const std::shared_ptr<Variable<T> > &other) const {
        auto dot_fn = std::make_shared<DotFunction<T> >();
        Tensor<T> result = dot_fn->forward({*this, *other});

        if (requires_grad_ || other->requires_grad_) {
            auto self_ptr = shared_from_this();
            return Variable<T>::create_with_grad_fn(result, dot_fn, {self_ptr, other});
        }
        return std::make_shared<Variable<T> >(result, false);
    }

    template<typename T>
    std::shared_ptr<Variable<T> > Variable<T>::transpose() const {
        auto transpose_fn = std::make_shared<TransposeFunction<T> >();
        Tensor<T> result = transpose_fn->forward({*this});

        if (requires_grad_) {
            auto self_ptr = shared_from_this();
            return Variable<T>::create_with_grad_fn(result, transpose_fn, {self_ptr});
        }
        return std::make_shared<Variable<T> >(result, false);
    }

    template<typename T>
    std::shared_ptr<Variable<T> > Variable<T>::transpose(const std::vector<size_t> &axes) const {
        auto transpose_axes_fn = std::make_shared<TransposeAxesFunction<T> >(axes);
        Tensor<T> result = transpose_axes_fn->forward({*this});

        if (requires_grad_) {
            auto self_ptr = shared_from_this();
            return Variable<T>::create_with_grad_fn(result, transpose_axes_fn, {self_ptr});
        }
        return std::make_shared<Variable<T> >(result, false);
    }

    template<typename T>
    std::shared_ptr<Variable<T> > Variable<T>::view(const std::vector<size_t> &new_shape) const {
        auto view_fn = std::make_shared<ViewFunction<T> >(data_.shape());
        Tensor<T> result = view_fn->forward({*this});

        if (requires_grad_) {
            auto self_ptr = shared_from_this();
            return Variable<T>::create_with_grad_fn(result, view_fn, {self_ptr});
        }
        return std::make_shared<Variable<T> >(result, false);
    }

    template<typename T>
    std::shared_ptr<Variable<T> > Variable<T>::squeeze(int axis) const {
        auto squeeze_fn = std::make_shared<SqueezeFunction<T> >(axis, data_.shape());
        Tensor<T> result = squeeze_fn->forward({*this});

        if (requires_grad_) {
            auto self_ptr = shared_from_this();
            return Variable<T>::create_with_grad_fn(result, squeeze_fn, {self_ptr});
        }
        return std::make_shared<Variable<T> >(result, false);
    }

    template<typename T>
    std::shared_ptr<Variable<T> > Variable<T>::unsqueeze(size_t axis) const {
        auto unsqueeze_fn = std::make_shared<UnsqueezeFunction<T> >(axis);
        Tensor<T> result = unsqueeze_fn->forward({*this});

        if (requires_grad_) {
            auto self_ptr = shared_from_this();
            return Variable<T>::create_with_grad_fn(result, unsqueeze_fn, {self_ptr});
        }
        return std::make_shared<Variable<T> >(result, false);
    }

    template<typename T>
    std::shared_ptr<Variable<T> > Variable<T>::sum(const std::vector<int> &axes, bool keepdims) const {
        auto sum_axes_fn = std::make_shared<SumAxesFunction<T> >(axes, keepdims);
        Tensor<T> result = sum_axes_fn->forward({*this});

        if (requires_grad_) {
            auto self_ptr = shared_from_this();
            return Variable<T>::create_with_grad_fn(result, sum_axes_fn, {self_ptr});
        }
        return std::make_shared<Variable<T> >(result, false);
    }

    template<typename T>
    std::shared_ptr<Variable<T> > Variable<T>::mean(const std::vector<int> &axes, bool keepdims) const {
        auto mean_axes_fn = std::make_shared<MeanAxesFunction<T> >(axes, keepdims);
        Tensor<T> result = mean_axes_fn->forward({*this});

        if (requires_grad_) {
            auto self_ptr = shared_from_this();
            return Variable<T>::create_with_grad_fn(result, mean_axes_fn, {self_ptr});
        }
        return std::make_shared<Variable<T> >(result, false);
    }

    template<typename T>
    std::shared_ptr<Variable<T> > Variable<T>::sum() const {
        auto sum_fn = std::make_shared<SumFunction<T> >();
        Tensor<T> result = sum_fn->forward({*this});

        if (requires_grad_) {
            auto self_ptr = shared_from_this();
            auto result_var = Variable<T>::create_with_grad_fn(result, sum_fn, {self_ptr});
            return result_var;
        }
        return std::make_shared<Variable<T> >(result, false);
    }

    template<typename T>
    std::shared_ptr<Variable<T> > Variable<T>::mean() const {
        auto mean_fn = std::make_shared<MeanFunction<T> >();
        Tensor<T> result = mean_fn->forward({*this});

        if (requires_grad_) {
            auto self_ptr = shared_from_this();
            auto result_var = Variable<T>::create_with_grad_fn(result, mean_fn, {self_ptr});
            return result_var;
        }
        return std::make_shared<Variable<T> >(result, false);
    }

    template<typename T>
    std::shared_ptr<Variable<T> > Variable<T>::sigmoid() const {
        auto sigmoid_fn = std::make_shared<SigmoidFunction<T> >();
        Tensor<T> result = sigmoid_fn->forward({*this});

        if (requires_grad_) {
            auto self_ptr = shared_from_this();
            auto result_var = Variable<T>::create_with_grad_fn(result, sigmoid_fn, {self_ptr});
            return result_var;
        }
        return std::make_shared<Variable<T> >(result, false);
    }

    template<typename T>
    std::shared_ptr<Variable<T> > Variable<T>::tanh() const {
        auto tanh_fn = std::make_shared<TanhFunction<T> >();
        Tensor<T> result = tanh_fn->forward({*this});

        if (requires_grad_) {
            auto self_ptr = shared_from_this();
            auto result_var = Variable<T>::create_with_grad_fn(result, tanh_fn, {self_ptr});
            return result_var;
        }
        return std::make_shared<Variable<T> >(result, false);
    }

    template<typename T>
    std::shared_ptr<Variable<T> > Variable<T>::relu() const {
        auto relu_fn = std::make_shared<ReLUFunction<T> >();
        Tensor<T> result = relu_fn->forward({*this});

        if (requires_grad_) {
            auto self_ptr = shared_from_this();
            auto result_var = Variable<T>::create_with_grad_fn(result, relu_fn, {self_ptr});
            return result_var;
        }
        return std::make_shared<Variable<T> >(result, false);
    }

    template<typename T>
    std::shared_ptr<Variable<T> > Variable<T>::exp() const {
        auto exp_fn = std::make_shared<ExpFunction<T> >();
        Tensor<T> result = exp_fn->forward({*this});

        if (requires_grad_) {
            auto self_ptr = shared_from_this();
            return Variable<T>::create_with_grad_fn(result, exp_fn, {self_ptr});
        }
        return std::make_shared<Variable<T> >(result, false);
    }

    template<typename T>
    std::shared_ptr<Variable<T> > Variable<T>::log() const {
        auto log_fn = std::make_shared<LogFunction<T> >();
        Tensor<T> result = log_fn->forward({*this});

        if (requires_grad_) {
            auto self_ptr = shared_from_this();
            return Variable<T>::create_with_grad_fn(result, log_fn, {self_ptr});
        }
        return std::make_shared<Variable<T> >(result, false);
    }

    // Function class method implementations
    template<typename T>
    void Function<T>::set_inputs(const std::vector<std::shared_ptr<Variable<T> > > &inputs) {
        input_variables_ = inputs;
    }

    template<typename T>
    const std::vector<std::shared_ptr<Variable<T> > > &Function<T>::get_inputs() const {
        return input_variables_;
    }

    template<typename T>
    bool Function<T>::needs_input_grad(size_t index) const {
        return index < input_variables_.size() && input_variables_[index] && input_variables_[index]->requires_grad();
    }

    template<typename T>
    void Function<T>::save_for_backward(const std::vector<Tensor<T> > &tensors) {
        saved_tensors_ = tensors;
    }

    template<typename T>
    const std::vector<Tensor<T> > &Function<T>::get_saved_tensors() const {
        return saved_tensors_;
    }

    // Variable class method implementations
    template<typename T>
    Variable<T>::Variable(const Tensor<T> &data, bool requires_grad)
        : data_(data), grad_(Tensor<T>()), requires_grad_(requires_grad), grad_fn_(nullptr) {
    }

    template<typename T>
    Variable<T>::Variable(const Tensor<T> &data, Variable<T>::FunctionPtr grad_fn)
        : data_(data), grad_(Tensor<T>()), requires_grad_(true), grad_fn_(grad_fn) {
    }

    template<typename T>
    Variable<T>::VariablePtr Variable<T>::shared_from_this() const {
        return std::const_pointer_cast<Variable<T> >(
            this->std::enable_shared_from_this<Variable<T> >::shared_from_this());
    }

    template<typename T>
    std::shared_ptr<Variable<T> > Variable<T>::create_with_grad_fn(const Tensor<T> &data,
                                                                   FunctionPtr grad_fn,
                                                                   const std::vector<VariablePtr> &inputs) {
        auto var = std::make_shared<Variable<T> >(data, grad_fn);
        if (grad_fn) {
            grad_fn->set_inputs(inputs);
        }
        return var;
    }

    template<typename T>
    void Variable<T>::set_requires_grad(bool requires_grad) {
        requires_grad_ = requires_grad;
    }

    template<typename T>
    void Variable<T>::zero_grad() {
        grad_ = Tensor<T>();
    }

    template<typename T>
    std::shared_ptr<Variable<T> > Variable<T>::detach() const {
        return std::make_shared<Variable<T> >(data_, false);
    }

    template<typename T>
    std::shared_ptr<Variable<T> > Variable<T>::clone(bool requires_grad) const {
        auto cloned = std::make_shared<Variable<T> >(data_, requires_grad);
        if (!grad_.empty()) {
            cloned->grad_ = grad_;
        }
        return cloned;
    }

    template<typename T>
    size_t Variable<T>::rows() const {
        auto s = data_.shape();
        return s.size() >= 2 ? s[0] : (s.size() == 1 ? s[0] : 1);
    }

    template<typename T>
    size_t Variable<T>::cols() const {
        auto s = data_.shape();
        return s.size() >= 2 ? s[1] : 1;
    }

    // Function implementations
    template<typename T>
    Tensor<T> AddFunction<T>::forward(const std::vector<Variable<T> > &inputs) {
        return inputs[0].data() + inputs[1].data();
    }

    template<typename T>
    Function<T>::TensorVec AddFunction<T>::backward(const Tensor<T> &grad_output) {
        return {grad_output, grad_output};
    }

    template<typename T>
    size_t AddFunction<T>::num_inputs() const {
        return 2;
    }

    template<typename T>
    Tensor<T> SubFunction<T>::forward(const std::vector<Variable<T> > &inputs) {
        return inputs[0].data() - inputs[1].data();
    }

    template<typename T>
    Function<T>::TensorVec SubFunction<T>::backward(const Tensor<T> &grad_output) {
        return {grad_output, -grad_output};
    }

    template<typename T>
    size_t SubFunction<T>::num_inputs() const {
        return 2;
    }

    template<typename T>
    Tensor<T> MulFunction<T>::forward(const std::vector<Variable<T> > &inputs) {
        this->save_for_backward({inputs[0].data(), inputs[1].data()});
        return inputs[0].data() * inputs[1].data();
    }

    template<typename T>
    Function<T>::TensorVec MulFunction<T>::backward(const Tensor<T> &grad_output) {
        auto saved = this->get_saved_tensors();
        auto grad0 = grad_output * saved[1];
        auto grad1 = grad_output * saved[0];
        return {grad0, grad1};
    }

    template<typename T>
    size_t MulFunction<T>::num_inputs() const {
        return 2;
    }

    template<typename T>
    Tensor<T> DivFunction<T>::forward(const std::vector<Variable<T> > &inputs) {
        this->save_for_backward({inputs[0].data(), inputs[1].data()});
        return inputs[0].data() / inputs[1].data();
    }

    template<typename T>
    Function<T>::TensorVec DivFunction<T>::backward(const Tensor<T> &grad_output) {
        auto saved = this->get_saved_tensors();
        auto grad_a = grad_output / saved[1];
        auto grad_b = -grad_output * saved[0] / (saved[1] * saved[1]);
        return {grad_a, grad_b};
    }

    template<typename T>
    size_t DivFunction<T>::num_inputs() const {
        return 2;
    }

    template<typename T>
    Tensor<T> MatMulFunction<T>::forward(const std::vector<Variable<T> > &inputs) {
        this->save_for_backward({inputs[0].data(), inputs[1].data()});
        return inputs[0].data().matmul(inputs[1].data());
    }

    template<typename T>
    Function<T>::TensorVec MatMulFunction<T>::backward(const Tensor<T> &grad_output) {
        auto saved = this->get_saved_tensors();
        auto grad_a = grad_output.matmul(saved[1].transpose());
        auto grad_b = saved[0].transpose().matmul(grad_output);
        return {grad_a, grad_b};
    }

    template<typename T>
    size_t MatMulFunction<T>::num_inputs() const {
        return 2;
    }

    template<typename T>
    Tensor<T> DotFunction<T>::forward(const std::vector<Variable<T> > &inputs) {
        this->save_for_backward({inputs[0].data(), inputs[1].data()});
        return dot(inputs[0].data(), inputs[1].data());
    }

    template<typename T>
    Function<T>::TensorVec DotFunction<T>::backward(const Tensor<T> &grad_output) {
        auto saved = this->get_saved_tensors();
        // Extract scalar value from grad_output
        T grad_scalar;
        if (grad_output.empty()) {
            // If grad_output is empty, use 1.0 as default gradient
            grad_scalar = static_cast<T>(1.0);
        } else if (grad_output.shape().size() == 0) {
            // 0-dimensional tensor (scalar) - access without indices
            grad_scalar = grad_output.scalar();
        } else {
            // Extract scalar value from grad_output (which is a scalar from dot product)
            grad_scalar = grad_output(0);
        }
        return {saved[1] * grad_scalar, saved[0] * grad_scalar};
    }

    template<typename T>
    size_t DotFunction<T>::num_inputs() const {
        return 2;
    }

    template<typename T>
    Tensor<T> TransposeFunction<T>::forward(const std::vector<Variable<T> > &inputs) {
        return inputs[0].data().transpose();
    }

    template<typename T>
    Function<T>::TensorVec TransposeFunction<T>::backward(const Tensor<T> &grad_output) {
        return {grad_output.transpose()};
    }

    template<typename T>
    size_t TransposeFunction<T>::num_inputs() const {
        return 1;
    }

    template<typename T>
    ReshapeFunction<T>::ReshapeFunction(const std::vector<size_t> &original_shape)
        : original_shape_(original_shape) {
    }

    template<typename T>
    Tensor<T> ReshapeFunction<T>::forward(const std::vector<Variable<T> > &inputs) {
        return inputs[0].data(); // Reshape logic would be implemented in Tensor class
    }

    template<typename T>
    Function<T>::TensorVec ReshapeFunction<T>::backward(const Tensor<T> &grad_output) {
        return {grad_output}; // Reshape back to original shape
    }

    template<typename T>
    size_t ReshapeFunction<T>::num_inputs() const {
        return 1;
    }

    template<typename T>
    Tensor<T> SigmoidFunction<T>::forward(const std::vector<Variable<T> > &inputs) {
        auto result = inputs[0].data().sigmoid();
        this->save_for_backward({result});
        return result;
    }

    template<typename T>
    Function<T>::TensorVec SigmoidFunction<T>::backward(const Tensor<T> &grad_output) {
        auto saved = this->get_saved_tensors();
        auto sigmoid_out = saved[0];
        return {grad_output * sigmoid_out * (Tensor<T>::ones_like(sigmoid_out) - sigmoid_out)};
    }

    template<typename T>
    size_t SigmoidFunction<T>::num_inputs() const {
        return 1;
    }

    template<typename T>
    Tensor<T> ReLUFunction<T>::forward(const std::vector<Variable<T> > &inputs) {
        this->save_for_backward({inputs[0].data()});
        return inputs[0].data().relu();
    }

    template<typename T>
    Function<T>::TensorVec ReLUFunction<T>::backward(const Tensor<T> &grad_output) {
        auto saved = this->get_saved_tensors();
        auto mask = saved[0] > Tensor<T>::zeros(saved[0].shape());
        auto mask_casted = mask.template cast<T>();
        return {grad_output * mask_casted};
    }

    template<typename T>
    size_t ReLUFunction<T>::num_inputs() const {
        return 1;
    }

    template<typename T>
    Tensor<T> ExpFunction<T>::forward(const std::vector<Variable<T> > &inputs) {
        auto result = inputs[0].data().exp();
        this->save_for_backward({result});
        return result;
    }

    template<typename T>
    Function<T>::TensorVec ExpFunction<T>::backward(const Tensor<T> &grad_output) {
        auto saved = this->get_saved_tensors();
        return {grad_output * saved[0]};
    }

    template<typename T>
    size_t ExpFunction<T>::num_inputs() const {
        return 1;
    }

    template<typename T>
    Tensor<T> SumFunction<T>::forward(const std::vector<Variable<T> > &inputs) {
        this->save_for_backward({inputs[0].data()});
        return inputs[0].data().sum();
    }

    template<typename T>
    Function<T>::TensorVec SumFunction<T>::backward(const Tensor<T> &grad_output) {
        auto saved = this->get_saved_tensors();
        std::cout << "SumFunction backward: grad_output=" << grad_output << ", saved.size()=" << saved.size() <<
                std::endl;
        if (!saved.empty()) {
            std::cout << "saved[0]=" << saved[0] << std::endl;
        }
        // Extract scalar value from grad_output
        T grad_scalar;
        if (grad_output.empty()) {
            // If grad_output is empty, use 1.0 as default gradient
            grad_scalar = static_cast<T>(1.0);
            std::cout << "Using default grad_scalar=1.0" << std::endl;
        } else if (grad_output.shape().size() == 0) {
            // 0-dimensional tensor (scalar) - access without indices
            grad_scalar = grad_output.scalar();
            std::cout << "Scalar grad_scalar=" << grad_scalar << std::endl;
        } else {
            // Extract scalar value from grad_output (which is a 1x1 tensor from sum)
            grad_scalar = grad_output(0);
            std::cout << "Indexed grad_scalar=" << grad_scalar << std::endl;
        }

        auto ones_tensor = Tensor<T>::ones_like(saved[0]);
        std::cout << "ones_like tensor=" << ones_tensor << std::endl;
        auto result = ones_tensor * grad_scalar;
        std::cout << "Final gradient=" << result << std::endl;
        return {result};
    }

    template<typename T>
    size_t SumFunction<T>::num_inputs() const {
        return 1;
    }

    template<typename T>
    Tensor<T> MeanFunction<T>::forward(const std::vector<Variable<T> > &inputs) {
        this->save_for_backward({inputs[0].data()});
        return inputs[0].data().mean();
    }

    template<typename T>
    Function<T>::TensorVec MeanFunction<T>::backward(const Tensor<T> &grad_output) {
        auto saved = this->get_saved_tensors();
        auto size = saved[0].size();
        // Extract scalar value from grad_output
        T grad_scalar;
        if (grad_output.empty()) {
            // If grad_output is empty, use 1.0 as default gradient
            grad_scalar = static_cast<T>(1.0);
        } else if (grad_output.shape().size() == 0) {
            // 0-dimensional tensor (scalar) - access without indices
            grad_scalar = grad_output.scalar();
        } else {
            // Extract scalar value from grad_output (which is a 1x1 tensor from mean)
            grad_scalar = grad_output(0);
        }
        return {Tensor<T>::ones_like(saved[0]) * grad_scalar * (T(1) / T(size))};
    }

    template<typename T>
    size_t MeanFunction<T>::num_inputs() const {
        return 1;
    }

    template<typename T>
    Tensor<T> TanhFunction<T>::forward(const std::vector<Variable<T> > &inputs) {
        auto result = inputs[0].data().tanh();
        this->save_for_backward({result});
        return result;
    }

    template<typename T>
    Function<T>::TensorVec TanhFunction<T>::backward(const Tensor<T> &grad_output) {
        auto saved = this->get_saved_tensors();
        auto tanh_out = saved[0];
        return {grad_output * (Tensor<T>::ones_like(tanh_out) - tanh_out * tanh_out)};
    }

    template<typename T>
    size_t TanhFunction<T>::num_inputs() const {
        return 1;
    }

    template<typename T>
    Tensor<T> LogFunction<T>::forward(const std::vector<Variable<T> > &inputs) {
        this->save_for_backward({inputs[0].data()});
        return inputs[0].data().log();
    }

    template<typename T>
    Function<T>::TensorVec LogFunction<T>::backward(const Tensor<T> &grad_output) {
        auto saved = this->get_saved_tensors();
        return {grad_output / saved[0]};
    }

    template<typename T>
    size_t LogFunction<T>::num_inputs() const {
        return 1;
    }

    // TransposeAxesFunction implementation
    template<typename T>
    TransposeAxesFunction<T>::TransposeAxesFunction(const std::vector<size_t> &axes)
        : axes_(axes) {
    }

    template<typename T>
    Tensor<T> TransposeAxesFunction<T>::forward(const std::vector<Variable<T> > &inputs) {
        return inputs[0].data().transpose(axes_);
    }

    template<typename T>
    Function<T>::TensorVec TransposeAxesFunction<T>::backward(const Tensor<T> &grad_output) {
        // Create inverse permutation
        std::vector<size_t> inverse_axes(axes_.size());
        for (size_t i = 0; i < axes_.size(); ++i) {
            inverse_axes[axes_[i]] = i;
        }
        return {grad_output.transpose(inverse_axes)};
    }

    template<typename T>
    size_t TransposeAxesFunction<T>::num_inputs() const {
        return 1;
    }

    // ViewFunction implementation
    template<typename T>
    ViewFunction<T>::ViewFunction(const std::vector<size_t> &original_shape)
        : original_shape_(original_shape) {
    }

    template<typename T>
    Tensor<T> ViewFunction<T>::forward(const std::vector<Variable<T> > &inputs) {
        this->save_for_backward({inputs[0].data()});
        return inputs[0].data();
    }

    template<typename T>
    Function<T>::TensorVec ViewFunction<T>::backward(const Tensor<T> &grad_output) {
        return {grad_output.reshape(original_shape_)};
    }

    template<typename T>
    size_t ViewFunction<T>::num_inputs() const {
        return 1;
    }

    // SqueezeFunction implementation
    template<typename T>
    SqueezeFunction<T>::SqueezeFunction(int axis, const std::vector<size_t> &original_shape)
        : axis_(axis), original_shape_(original_shape) {
    }

    template<typename T>
    Tensor<T> SqueezeFunction<T>::forward(const std::vector<Variable<T> > &inputs) {
        return inputs[0].data().squeeze(axis_);
    }

    template<typename T>
    Function<T>::TensorVec SqueezeFunction<T>::backward(const Tensor<T> &grad_output) {
        return {grad_output.reshape(original_shape_)};
    }

    template<typename T>
    size_t SqueezeFunction<T>::num_inputs() const {
        return 1;
    }

    // UnsqueezeFunction implementation
    template<typename T>
    UnsqueezeFunction<T>::UnsqueezeFunction(size_t axis)
        : axis_(axis) {
    }

    template<typename T>
    Tensor<T> UnsqueezeFunction<T>::forward(const std::vector<Variable<T> > &inputs) {
        this->save_for_backward({inputs[0].data()});
        return inputs[0].data().unsqueeze(axis_);
    }

    template<typename T>
    Function<T>::TensorVec UnsqueezeFunction<T>::backward(const Tensor<T> &grad_output) {
        return {grad_output.squeeze(static_cast<int>(axis_))};
    }

    template<typename T>
    size_t UnsqueezeFunction<T>::num_inputs() const {
        return 1;
    }

    // SumAxesFunction implementation
    template<typename T>
    SumAxesFunction<T>::SumAxesFunction(const std::vector<int> &axes, bool keepdims)
        : axes_(axes), keepdims_(keepdims) {
    }

    template<typename T>
    Tensor<T> SumAxesFunction<T>::forward(const std::vector<Variable<T> > &inputs) {
        this->save_for_backward({inputs[0].data()});
        return inputs[0].data().sum(axes_, keepdims_);
    }

    template<typename T>
    Function<T>::TensorVec SumAxesFunction<T>::backward(const Tensor<T> &grad_output) {
        auto saved = this->get_saved_tensors();
        auto grad = grad_output;

        // If keepdims is false, we need to unsqueeze the gradient
        if (!keepdims_) {
            for (int axis: axes_) {
                if (axis >= 0) {
                    grad = grad.unsqueeze(static_cast<size_t>(axis));
                }
            }
        }

        // Broadcast gradient to match input shape
        return {Tensor<T>::ones_like(saved[0]) * grad};
    }

    template<typename T>
    size_t SumAxesFunction<T>::num_inputs() const {
        return 1;
    }

    // MeanAxesFunction implementation
    template<typename T>
    MeanAxesFunction<T>::MeanAxesFunction(const std::vector<int> &axes, bool keepdims)
        : axes_(axes), keepdims_(keepdims) {
    }

    template<typename T>
    Tensor<T> MeanAxesFunction<T>::forward(const std::vector<Variable<T> > &inputs) {
        this->save_for_backward({inputs[0].data()});
        return inputs[0].data().mean(axes_, keepdims_);
    }

    template<typename T>
    Function<T>::TensorVec MeanAxesFunction<T>::backward(const Tensor<T> &grad_output) {
        auto saved = this->get_saved_tensors();
        auto grad = grad_output;

        // Calculate the number of elements being averaged
        size_t num_elements = 1;
        for (int axis: axes_) {
            if (axis >= 0 && axis < static_cast<int>(saved[0].shape().size())) {
                num_elements *= saved[0].shape()[axis];
            }
        }

        // If keepdims is false, we need to unsqueeze the gradient
        if (!keepdims_) {
            for (int axis: axes_) {
                if (axis >= 0) {
                    grad = grad.unsqueeze(static_cast<size_t>(axis));
                }
            }
        }

        // Broadcast gradient to match input shape and divide by number of elements
        return {Tensor<T>::ones_like(saved[0]) * grad * (T(1) / T(num_elements))};
    }

    template<typename T>
    size_t MeanAxesFunction<T>::num_inputs() const {
        return 1;
    }

    // ============================================================================
    // SCALAR OPERATION IMPLEMENTATIONS
    // ============================================================================

    template<typename T>
    ScalarAddFunction<T>::ScalarAddFunction(T scalar) : scalar_(scalar) {
    }

    template<typename T>
    Tensor<T> ScalarAddFunction<T>::forward(const std::vector<Variable<T> > &inputs) {
        return inputs[0].data() + Tensor<T>::full(inputs[0].data().shape(), scalar_);
    }

    template<typename T>
    Function<T>::TensorVec ScalarAddFunction<T>::backward(const Tensor<T> &grad_output) {
        return {grad_output};
    }

    template<typename T>
    size_t ScalarAddFunction<T>::num_inputs() const {
        return 1;
    }

    template<typename T>
    ScalarMulFunction<T>::ScalarMulFunction(T scalar) : scalar_(scalar) {
    }

    template<typename T>
    Tensor<T> ScalarMulFunction<T>::forward(const std::vector<Variable<T> > &inputs) {
        return inputs[0].data() * scalar_;
    }

    template<typename T>
    Function<T>::TensorVec ScalarMulFunction<T>::backward(const Tensor<T> &grad_output) {
        return {grad_output * scalar_};
    }

    template<typename T>
    size_t ScalarMulFunction<T>::num_inputs() const {
        return 1;
    }

    template<typename T>
    ScalarSubFunction<T>::ScalarSubFunction(T scalar) : scalar_(scalar) {
    }

    template<typename T>
    Tensor<T> ScalarSubFunction<T>::forward(const std::vector<Variable<T> > &inputs) {
        return inputs[0].data() + Tensor<T>::full(inputs[0].data().shape(), -scalar_);
    }

    template<typename T>
    Function<T>::TensorVec ScalarSubFunction<T>::backward(const Tensor<T> &grad_output) {
        return {grad_output};
    }

    template<typename T>
    size_t ScalarSubFunction<T>::num_inputs() const {
        return 1;
    }

    template<typename T>
    ScalarDivFunction<T>::ScalarDivFunction(T scalar) : scalar_(scalar) {
    }

    template<typename T>
    Tensor<T> ScalarDivFunction<T>::forward(const std::vector<Variable<T> > &inputs) {
        return inputs[0].data() * (T(1) / scalar_);
    }

    template<typename T>
    Function<T>::TensorVec ScalarDivFunction<T>::backward(const Tensor<T> &grad_output) {
        return {grad_output * (T(1) / scalar_)};
    }

    template<typename T>
    size_t ScalarDivFunction<T>::num_inputs() const {
        return 1;
    }

    template<typename T>
    ReverseScalarSubFunction<T>::ReverseScalarSubFunction(T scalar) : scalar_(scalar) {
    }

    template<typename T>
    Tensor<T> ReverseScalarSubFunction<T>::forward(const std::vector<Variable<T> > &inputs) {
        return Tensor<T>::full(inputs[0].data().shape(), scalar_) - inputs[0].data();
    }

    template<typename T>
    Function<T>::TensorVec ReverseScalarSubFunction<T>::backward(const Tensor<T> &grad_output) {
        return {-grad_output};
    }

    template<typename T>
    size_t ReverseScalarSubFunction<T>::num_inputs() const {
        return 1;
    }

    template<typename T>
    ReverseScalarDivFunction<T>::ReverseScalarDivFunction(T scalar) : scalar_(scalar) {
    }

    template<typename T>
    Tensor<T> ReverseScalarDivFunction<T>::forward(const std::vector<Variable<T> > &inputs) {
        return Tensor<T>::full(inputs[0].data().shape(), scalar_) / inputs[0].data();
    }

    template<typename T>
    Function<T>::TensorVec ReverseScalarDivFunction<T>::backward(const Tensor<T> &grad_output) {
        const auto &inputs = this->get_inputs();
        const Tensor<T> &input_data = inputs[0]->data();
        auto input_squared = input_data * input_data;
        auto ones_tensor = Tensor<T>::ones(input_squared.shape());
        return {grad_output * (-scalar_) * (ones_tensor / input_squared)};
    }

    template<typename T>
    size_t ReverseScalarDivFunction<T>::num_inputs() const {
        return 1;
    }

    // Global operator overloads for smart pointers
    template<typename T>
    std::shared_ptr<Variable<T> > operator+(const std::shared_ptr<Variable<T> > &lhs,
                                            const std::shared_ptr<Variable<T> > &rhs) {
        return Variable<T>::add(lhs, rhs);
    }

    template<typename T>
    std::shared_ptr<Variable<T> > operator-(const std::shared_ptr<Variable<T> > &lhs,
                                            const std::shared_ptr<Variable<T> > &rhs) {
        return Variable<T>::sub(lhs, rhs);
    }

    template<typename T>
    std::shared_ptr<Variable<T> > operator*(const std::shared_ptr<Variable<T> > &lhs,
                                            const std::shared_ptr<Variable<T> > &rhs) {
        return Variable<T>::mul(lhs, rhs);
    }

    template<typename T>
    std::shared_ptr<Variable<T> > operator/(const std::shared_ptr<Variable<T> > &lhs,
                                            const std::shared_ptr<Variable<T> > &rhs) {
        return Variable<T>::div(lhs, rhs);
    }

    template<typename T>
    std::shared_ptr<Variable<T> > operator*(const std::shared_ptr<Variable<T> > &lhs, T scalar) {
        return Variable<T>::mul(lhs, scalar);
    }

    template<typename T>
    std::shared_ptr<Variable<T> > operator*(T scalar, const std::shared_ptr<Variable<T> > &rhs) {
        return Variable<T>::mul(rhs, scalar);
    }

    template<typename T>
    std::shared_ptr<Variable<T> > operator+(const std::shared_ptr<Variable<T> > &lhs, T scalar) {
        return Variable<T>::add(lhs, scalar);
    }

    template<typename T>
    std::shared_ptr<Variable<T> > operator+(T scalar, const std::shared_ptr<Variable<T> > &rhs) {
        return Variable<T>::add(rhs, scalar);
    }

    template<typename T>
    std::shared_ptr<Variable<T> > operator-(const std::shared_ptr<Variable<T> > &lhs, T scalar) {
        return Variable<T>::sub(lhs, scalar);
    }

    template<typename T>
    std::shared_ptr<Variable<T> > operator-(T scalar, const std::shared_ptr<Variable<T> > &rhs) {
        auto reverse_scalar_sub_fn = std::make_shared<ReverseScalarSubFunction<T> >(scalar);
        Tensor<T> result = reverse_scalar_sub_fn->forward({*rhs});

        if (rhs->requires_grad()) {
            auto result_var = Variable<T>::create_with_grad_fn(result, reverse_scalar_sub_fn, {rhs});
            return result_var;
        }
        return std::make_shared<Variable<T> >(result, false);
    }

    template<typename T>
    std::shared_ptr<Variable<T> > operator/(const std::shared_ptr<Variable<T> > &lhs, T scalar) {
        return Variable<T>::div(lhs, scalar);
    }

    template<typename T>
    std::shared_ptr<Variable<T> > operator/(T scalar, const std::shared_ptr<Variable<T> > &rhs) {
        auto reverse_scalar_div_fn = std::make_shared<ReverseScalarDivFunction<T> >(scalar);
        Tensor<T> result = reverse_scalar_div_fn->forward({*rhs});

        if (rhs->requires_grad()) {
            auto result_var = Variable<T>::create_with_grad_fn(result, reverse_scalar_div_fn, {rhs});
            return result_var;
        }
        return std::make_shared<Variable<T> >(result, false);
    }

    // Explicit template instantiations
    template class Variable<float>;
    template class Variable<double>;

    // Explicit instantiations for global operators
    template std::shared_ptr<Variable<float> > operator+(const std::shared_ptr<Variable<float> > &lhs,
                                                         const std::shared_ptr<Variable<float> > &rhs);

    template std::shared_ptr<Variable<double> > operator+(const std::shared_ptr<Variable<double> > &lhs,
                                                          const std::shared_ptr<Variable<double> > &rhs);

    template std::shared_ptr<Variable<float> > operator-(const std::shared_ptr<Variable<float> > &lhs,
                                                         const std::shared_ptr<Variable<float> > &rhs);

    template std::shared_ptr<Variable<double> > operator-(const std::shared_ptr<Variable<double> > &lhs,
                                                          const std::shared_ptr<Variable<double> > &rhs);

    template std::shared_ptr<Variable<float> > operator*(const std::shared_ptr<Variable<float> > &lhs,
                                                         const std::shared_ptr<Variable<float> > &rhs);

    template std::shared_ptr<Variable<double> > operator*(const std::shared_ptr<Variable<double> > &lhs,
                                                          const std::shared_ptr<Variable<double> > &rhs);

    template std::shared_ptr<Variable<float> > operator/(const std::shared_ptr<Variable<float> > &lhs,
                                                         const std::shared_ptr<Variable<float> > &rhs);

    template std::shared_ptr<Variable<double> > operator/(const std::shared_ptr<Variable<double> > &lhs,
                                                          const std::shared_ptr<Variable<double> > &rhs);

    template std::shared_ptr<Variable<float> > operator*(const std::shared_ptr<Variable<float> > &lhs, float scalar);

    template std::shared_ptr<Variable<double> > operator*(const std::shared_ptr<Variable<double> > &lhs, double scalar);

    template std::shared_ptr<Variable<float> > operator*(float scalar, const std::shared_ptr<Variable<float> > &rhs);

    template std::shared_ptr<Variable<double> > operator*(double scalar, const std::shared_ptr<Variable<double> > &rhs);

    template std::shared_ptr<Variable<float> > operator+(const std::shared_ptr<Variable<float> > &lhs, float scalar);

    template std::shared_ptr<Variable<double> > operator+(const std::shared_ptr<Variable<double> > &lhs, double scalar);

    template std::shared_ptr<Variable<float> > operator+(float scalar, const std::shared_ptr<Variable<float> > &rhs);

    template std::shared_ptr<Variable<double> > operator+(double scalar, const std::shared_ptr<Variable<double> > &rhs);

    template std::shared_ptr<Variable<float> > operator-(const std::shared_ptr<Variable<float> > &lhs, float scalar);

    template std::shared_ptr<Variable<double> > operator-(const std::shared_ptr<Variable<double> > &lhs, double scalar);

    template std::shared_ptr<Variable<float> > operator-(float scalar, const std::shared_ptr<Variable<float> > &rhs);

    template std::shared_ptr<Variable<double> > operator-(double scalar, const std::shared_ptr<Variable<double> > &rhs);

    template std::shared_ptr<Variable<float> > operator/(const std::shared_ptr<Variable<float> > &lhs, float scalar);

    template std::shared_ptr<Variable<double> > operator/(const std::shared_ptr<Variable<double> > &lhs, double scalar);

    template std::shared_ptr<Variable<float> > operator/(float scalar, const std::shared_ptr<Variable<float> > &rhs);

    template std::shared_ptr<Variable<double> > operator/(double scalar, const std::shared_ptr<Variable<double> > &rhs);

    template class Function<float>;
    template class Function<double>;
    template class AddFunction<float>;
    template class AddFunction<double>;
    template class SubFunction<float>;
    template class SubFunction<double>;
    template class MulFunction<float>;
    template class MulFunction<double>;
    template class DivFunction<float>;
    template class DivFunction<double>;
    template class DotFunction<float>;
    template class DotFunction<double>;
    template class MatMulFunction<float>;
    template class MatMulFunction<double>;
    template class TransposeFunction<float>;
    template class TransposeFunction<double>;
    template class ReshapeFunction<float>;
    template class ReshapeFunction<double>;
    template class SigmoidFunction<float>;
    template class SigmoidFunction<double>;
    template class ReLUFunction<float>;
    template class ReLUFunction<double>;
    template class ExpFunction<float>;
    template class ExpFunction<double>;
    template class SumFunction<float>;
    template class SumFunction<double>;
    template class MeanFunction<float>;
    template class MeanFunction<double>;
    template class TanhFunction<float>;
    template class TanhFunction<double>;
    template class LogFunction<float>;
    template class LogFunction<double>;
    template class TransposeAxesFunction<float>;
    template class TransposeAxesFunction<double>;
    template class ViewFunction<float>;
    template class ViewFunction<double>;
    template class SqueezeFunction<float>;
    template class SqueezeFunction<double>;
    template class UnsqueezeFunction<float>;
    template class UnsqueezeFunction<double>;
    template class SumAxesFunction<float>;
    template class SumAxesFunction<double>;
    template class MeanAxesFunction<float>;
    template class MeanAxesFunction<double>;
    template class ScalarAddFunction<float>;
    template class ScalarAddFunction<double>;
    template class ScalarMulFunction<float>;
    template class ScalarMulFunction<double>;
    template class ScalarSubFunction<float>;
    template class ScalarSubFunction<double>;
    template class ScalarDivFunction<float>;
    template class ScalarDivFunction<double>;
    template class ReverseScalarSubFunction<float>;
    template class ReverseScalarSubFunction<double>;
    template class ReverseScalarDivFunction<float>;
    template class ReverseScalarDivFunction<double>;

    // Explicit function template instantiations
    // topological_sort is automatically instantiated when used in backward()

    // Helper function template instantiations
    template std::shared_ptr<Variable<float> > make_variable(const Tensor<float> &data, bool requires_grad);

    template std::shared_ptr<Variable<double> > make_variable(const Tensor<double> &data, bool requires_grad);

    template std::shared_ptr<Variable<float> > make_variable(const Tensor<float> &data,
                                                             std::shared_ptr<Function<float> > grad_fn);

    template std::shared_ptr<Variable<double> > make_variable(const Tensor<double> &data,
                                                              std::shared_ptr<Function<double> > grad_fn);

    template std::shared_ptr<Variable<float> > make_variable_scalar(float value, bool requires_grad);

    template std::shared_ptr<Variable<double> > make_variable_scalar(double value, bool requires_grad);

    template std::shared_ptr<Variable<float> >
    make_variable_zeros(const std::vector<size_t> &shape, bool requires_grad);

    template std::shared_ptr<Variable<double> > make_variable_zeros(const std::vector<size_t> &shape,
                                                                    bool requires_grad);

    template std::shared_ptr<Variable<float> > make_variable_ones(const std::vector<size_t> &shape, bool requires_grad);

    template std::shared_ptr<Variable<double> >
    make_variable_ones(const std::vector<size_t> &shape, bool requires_grad);
} // namespace dl
