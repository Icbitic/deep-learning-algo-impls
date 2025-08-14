# Variable Helper Functions

This document describes the shorthand helper functions for creating `Variable` objects on the heap, making it more convenient to work with the autograd system.

## Overview

The helper functions provide a convenient way to create `Variable` objects using `std::shared_ptr`, similar to `std::make_shared`. This eliminates the need for manual memory management and provides a cleaner API.

## Available Helper Functions

### 1. `make_variable_scalar<T>(value, requires_grad)`

Creates a scalar variable with the specified value.

```cpp
// Create scalar variables
auto x = utils::make_variable_scalar(2.5, true);   // double, requires_grad=true
auto y = utils::make_variable_scalar(3.0f, false); // float, requires_grad=false
```

### 2. `make_variable_zeros<T>(shape, requires_grad)`

Creates a variable filled with zeros of the specified shape.

```cpp
// Create zero-filled tensors
auto zeros_2x3 = utils::make_variable_zeros<double>({2, 3}, true);
auto zeros_vector = utils::make_variable_zeros<float>({5}, false);
```

### 3. `make_variable_ones<T>(shape, requires_grad)`

Creates a variable filled with ones of the specified shape.

```cpp
// Create ones-filled tensors
auto ones_3x3 = utils::make_variable_ones<double>({3, 3}, true);
auto ones_vector = utils::make_variable_ones<float>({4}, false);
```

### 4. `make_variable<T>(tensor, requires_grad)`

Creates a variable from an existing tensor.

```cpp
// Create from existing tensor
auto tensor_data = TensorD({1.0, 2.0, 3.0, 4.0}, {2, 2});
auto var_from_tensor = utils::make_variable(tensor_data, true);
```

## Type Aliases

For convenience, the following type aliases are provided:

- `VariableDPtr` - `std::shared_ptr<Variable<double>>`
- `VariableFPtr` - `std::shared_ptr<Variable<float>>`

```cpp
// Using type aliases
VariableDPtr double_var = utils::make_variable_scalar(10.0, true);
VariableFPtr float_var = utils::make_variable_scalar(5.0f, false);
```

## Usage in Autograd Computations

When using variables created with helper functions in computations, remember to dereference the shared pointers:

```cpp
auto a = utils::make_variable_scalar(4.0, true);
auto b = utils::make_variable_scalar(6.0, true);

// Dereference shared pointers for operations
auto result = (*a) * (*b);
result.backward();

// Access gradients through the computation graph
auto grad_fn = result.get_grad_fn();
if (grad_fn) {
    auto inputs = grad_fn->get_inputs();
    if (inputs.size() >= 2) {
        std::cout << "da/dx = " << inputs[0]->grad().scalar() << std::endl;
        std::cout << "db/dx = " << inputs[1]->grad().scalar() << std::endl;
    }
}
```

## Benefits

1. **Memory Safety**: Automatic memory management with `std::shared_ptr`
2. **Cleaner API**: No need for manual `new` and `delete`
3. **Type Safety**: Template-based functions ensure type consistency
4. **Convenience**: One-line creation of common variable types
5. **Consistency**: Similar to standard library patterns like `std::make_shared`

## Example Program

See `examples/variable_helpers_demo.cpp` for a complete demonstration of all helper functions.

To build and run the demo:

```bash
cmake -B build
cmake --build build --target variable_helpers_demo
./build/variable_helpers_demo
```