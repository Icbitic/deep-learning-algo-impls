cmake_minimum_required(VERSION 3.31)
project(deep_learning_algo_impls)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable testing
enable_testing()

# Find required packages
find_package(GTest REQUIRED)
find_package(Threads REQUIRED)

# Find xtensor packages
find_package(xtensor REQUIRED)
find_package(xtensor-blas QUIET)

# Set xtensor include directories explicitly
include_directories(${xtensor_INCLUDE_DIRS})
if(xtensor-blas_FOUND)
    include_directories(${xtensor-blas_INCLUDE_DIRS})
endif()

# Include directories
include_directories(include)
include_directories(/opt/homebrew/Cellar/xtensor/0.27.0/include)
include_directories(/usr/local/include)

# Collect source files
file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "include/*.hpp")

# Create library for deep learning algorithms
add_library(dl_algorithms ${SOURCES} ${HEADERS})
target_include_directories(dl_algorithms PUBLIC include)
target_link_libraries(dl_algorithms PUBLIC xtensor)
if(xtensor-blas_FOUND)
    target_link_libraries(dl_algorithms PUBLIC xtensor-blas)
endif()

# Main executable
add_executable(deep_learning_algo_impls main.cpp)
target_link_libraries(deep_learning_algo_impls dl_algorithms)

# Test executable
file(GLOB_RECURSE TEST_SOURCES "tests/*.cpp")
add_executable(run_tests ${TEST_SOURCES})
target_link_libraries(run_tests PRIVATE dl_algorithms GTest::gtest GTest::gtest_main Threads::Threads)
target_include_directories(run_tests PRIVATE include)

# Demo executable for SVM
add_executable(svm_demo examples/svm_demo.cpp)
target_link_libraries(svm_demo dl_algorithms)
target_include_directories(svm_demo PRIVATE include)

# Demo executable for Autograd Neural Network
add_executable(neural_network_demo demos/neural_network_demo.cpp)
target_link_libraries(neural_network_demo dl_algorithms)
target_include_directories(neural_network_demo PRIVATE include)

# Register tests with CTest
add_test(NAME unit_tests COMMAND run_tests)

# Compiler flags for better debugging and optimization
target_compile_options(dl_algorithms PRIVATE -Wall -Wextra -O2)
target_compile_options(deep_learning_algo_impls PRIVATE -Wall -Wextra -O2)
target_compile_options(run_tests PRIVATE -Wall -Wextra -O2)
target_compile_options(svm_demo PRIVATE -Wall -Wextra -O2)
target_compile_options(neural_network_demo PRIVATE -Wall -Wextra -O2)
