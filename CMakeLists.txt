cmake_minimum_required(VERSION 3.31)
project(deep_learning_algo_impls)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable testing
enable_testing()

# Find required packages
find_package(GTest REQUIRED)
find_package(Threads REQUIRED)

# ---- Fetch xtensor ----
include(FetchContent)
include(FetchContent)

FetchContent_Declare(
        xtensor
        GIT_REPOSITORY https://github.com/xtensor-stack/xtensor.git
        GIT_TAG        0.24.1
        PATCH_COMMAND ${CMAKE_COMMAND} -E echo 'cmake_policy(VERSION 3.5)' >> CMakeLists.txt
)

FetchContent_MakeAvailable(xtensor)

# Optionally fetch xtensor-blas
FetchContent_Declare(
        xtensor_blas
        GIT_REPOSITORY https://github.com/xtensor-stack/xtensor-blas.git
        GIT_TAG        master
)
FetchContent_MakeAvailable(xtensor_blas)

# Include project headers
include_directories(include)

# Collect source files
file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "include/*.hpp")

# Create library for deep learning algorithms
add_library(dl_algorithms ${SOURCES} ${HEADERS})
target_include_directories(dl_algorithms PUBLIC include)
target_link_libraries(dl_algorithms PUBLIC xtensor xtensor-blas)

# Main executable
add_executable(deep_learning_algo_impls main.cpp)
target_link_libraries(deep_learning_algo_impls dl_algorithms)

# Test executable
file(GLOB_RECURSE TEST_SOURCES "tests/*.cpp")
add_executable(run_tests ${TEST_SOURCES})
target_link_libraries(run_tests PRIVATE dl_algorithms GTest::gtest GTest::gtest_main Threads::Threads)
target_include_directories(run_tests PRIVATE include)

# Register tests with CTest
add_test(NAME unit_tests COMMAND run_tests)

# Compiler flags for better debugging and optimization
target_compile_options(dl_algorithms PRIVATE -Wall -Wextra -O2)
target_compile_options(deep_learning_algo_impls PRIVATE -Wall -Wextra -O2)
target_compile_options(run_tests PRIVATE -Wall -Wextra -O2)
