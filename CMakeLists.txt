cmake_minimum_required(VERSION 3.18)
project(cusolve LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==========================================
# 1. CPU Backend & Demo
# ==========================================
add_library(cusolve_cpu src/cpu/jacobi.cpp)
target_include_directories(cusolve_cpu PUBLIC include)

add_executable(demo_jacobi examples/cpu/demo_jacobi.cpp)
target_link_libraries(demo_jacobi PRIVATE cusolve_cpu)

# ==========================================
# 2. CUDA Backend & Demo
# ==========================================
option(BUILD_CUDA "Build CUDA backend" OFF)

if(BUILD_CUDA)
    enable_language(CUDA)
    
    add_library(cusolve_cuda src/cuda/jacobi.cu)
    target_include_directories(cusolve_cuda PUBLIC include)
    set_target_properties(cusolve_cuda PROPERTIES
        CUDA_ARCHITECTURES "61" # 61 for GTX 10-series/MX230. Change to "75" for Google Colab T4
        CUDA_STANDARD 17
    )

    add_executable(demo_jacobi_cuda examples/cuda/demo_jacobi_cuda.cpp)
    target_link_libraries(demo_jacobi_cuda PRIVATE cusolve_cuda)
endif()

# ==========================================
# 3. Unit Testing (Catch2)
# ==========================================
enable_testing()

# Fetch Catch2 automatically if not installed
include(FetchContent)
FetchContent_Declare(
    Catch2 
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git 
    GIT_TAG v3.5.0
)
FetchContent_MakeAvailable(Catch2)

# CPU Test Target
add_executable(tests_cpu tests/test_jacobi.cpp)
target_link_libraries(tests_cpu PRIVATE cusolve_cpu Catch2::Catch2WithMain)
add_test(NAME CPUTest COMMAND tests_cpu)

# CUDA Test Target (Only built if BUILD_CUDA=ON)
if(BUILD_CUDA)
    add_executable(tests_cuda tests/test_jacobi_cuda.cpp)
    target_link_libraries(tests_cuda PRIVATE cusolve_cuda Catch2::Catch2WithMain)
    add_test(NAME CUDATest COMMAND tests_cuda)
endif()