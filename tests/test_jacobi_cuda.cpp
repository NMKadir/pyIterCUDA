#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "cusolve/jacobi_cuda.hpp"

TEST_CASE("CUDA Jacobi converges on diagonally dominant system") {
    int n = 3;
    
    // Flattened 3x3 matrix (row-major format)
    std::vector<double> A_flat = {
        4,  1, -1,
        2,  7,  1,
        1, -3, 12
    };
    std::vector<double> b = {3, 19, 31};

    // Call the CUDA solver
    auto result = cusolve::jacobi_solve_cuda(A_flat, b, n);

    REQUIRE(result.converged);
    REQUIRE(result.x[0] == Catch::Approx(1.0).margin(1e-4));
    REQUIRE(result.x[1] == Catch::Approx(2.0).margin(1e-4));
    REQUIRE(result.x[2] == Catch::Approx(3.0).margin(1e-4));
}