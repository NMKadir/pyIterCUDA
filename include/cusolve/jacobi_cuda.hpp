#pragma once
#include<vector>
#include "cusolve/jacobi.hpp"
namespace cusolve{

    struct SolverResult;

    SolverResult jacobi_solve_cuda(
            const std::vector<double>& A_flat, //row-major, n*n
            const std::vector<double>& b,
            int n,
            double tol = 1e-8,
            int max_iter = 1000
    );
    
} //namespace cusolve