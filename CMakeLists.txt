cmake_minimum_required(VERSION 3.18)
project(cusolve LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==========================================
# 1. CPU Backend & Demo
# ==========================================
add_library(cusolve_cpu src/cpu/jacobi.cpp)
target_include_directories(cusolve_cpu PUBLIC include)
set_target_properties(cusolve_cpu PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_executable(demo_jacobi examples/cpu/demo_jacobi.cpp)
target_link_libraries(demo_jacobi PRIVATE cusolve_cpu)

# ==========================================
# 2. CUDA Backend & Demo
# ==========================================
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CUSOLVE_HAS_CUDA TRUE)
    message(STATUS "CUDA compiler found -> building GPU backend")

    add_library(cusolve_cuda src/cuda/jacobi.cu)
    target_include_directories(cusolve_cuda PUBLIC include)
    set_target_properties(cusolve_cuda PROPERTIES
        CUDA_ARCHITECTURES "61"
        CUDA_STANDARD 17
        POSITION_INDEPENDENT_CODE ON
    )
    add_executable(demo_jacobi_cuda examples/cuda/demo_jacobi_cuda.cpp)
    target_link_libraries(demo_jacobi_cuda PRIVATE cusolve_cuda)
else()
    set(CUSOLVE_HAS_CUDA FALSE)
    message(STATUS "No CUDA compiler found -> CPU-only build")
endif()
# ==========================================
# 3. Unit Testing (Catch2)
# ==========================================
enable_testing()
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.5.0
)
FetchContent_MakeAvailable(Catch2)

add_executable(tests_cpu tests/test_jacobi.cpp)
target_link_libraries(tests_cpu PRIVATE cusolve_cpu Catch2::Catch2WithMain)
add_test(NAME CPUTest COMMAND tests_cpu)

if(CUSOLVE_HAS_CUDA)
    add_executable(tests_cuda tests/test_jacobi_cuda.cpp)
    target_link_libraries(tests_cuda PRIVATE cusolve_cuda Catch2::Catch2WithMain)
    add_test(NAME CUDATest COMMAND tests_cuda)
endif()

# ==========================================
# 4. Python Bindings
# ==========================================
find_package(pybind11 CONFIG REQUIRED)
file(GLOB BINDING_SOURCES CONFIGURE_DEPENDS src/bindings/*.cpp)
pybind11_add_module(_cusolve ${BINDING_SOURCES})
target_link_libraries(_cusolve PRIVATE cusolve_cpu)

if(CUSOLVE_HAS_CUDA)
    target_link_libraries(_cusolve PRIVATE cusolve_cuda)
    target_compile_definitions(_cusolve PRIVATE CUSOLVE_HAS_CUDA)
endif()

install(TARGETS _cusolve DESTINATION cusolve)