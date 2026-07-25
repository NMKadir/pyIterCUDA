cmake_minimum_required(VERSION 3.18)
project(cusolve LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(cusolve_cpu src/cpu/jacobi.cpp)
target_include_directories(cusolve_cpu PUBLIC include)

add_executable(demo_jacobi examples/demo_jacobi.cpp)
target_link_libraries(demo_jacobi PRIVATE cusolve_cpu)

# 3. Unit Testing with Catch2
include(FetchContent)
FetchContent_Declare(
    Catch2 
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git 
    GIT_TAG v3.5.0
)
FetchContent_MakeAvailable(Catch2)

add_executable(tests tests/test_jacobi.cpp)
target_link_libraries(tests PRIVATE cusolve_cpu Catch2::Catch2WithMain)
