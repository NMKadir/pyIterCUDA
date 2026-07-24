#pragma once
#include<vector>

namespace cusolve{

    struct SolverResult{
        std::vector<double> x;
        int iterations;
        double residual_norm;
        bool converged;
    };

    SolverResult jacobi_solve(
        const std::vector<std::vector<double>>& A,
        const std::vector<double>& b,
        double tol = 1e-8,
        int max_iter = 1000
    );
} //namespace cusolve