#include "cusolve/jacobi.hpp"
#include <cmath>

namespace cusolve {

    SolverResult jacobi_solve(
        const std::vector<std::vector<double>>& A,
        const std::vector<double>& b,
        double tol,
        int max_iter)
        {
            int n = b.size();
            std::vector<double> x_old(n, 0.0), x_new(n, 0.0);

            for(int it=0; it<max_iter; ++it){
                for(int i=0; i<n; ++i){
                    double sigma = 0.0;
                    for(int j=0; j<n; j++)
                        if(j!=i) sigma += A[i][j] * x_old[j];
                    x_new[i] = (b[i] - sigma) / A[i][i];
                }

                double residual = 0.0;
                for(int i=0; i<n; ++i){
                    double r_i = b[i];
                    for(int j=0; j<n; ++j) r_i -= A[i][j] * x_new[j];
                    residual += r_i * r_i;
                }
                residual = std::sqrt(residual);
                x_old = x_new;
                if(residual < tol){
                    return {x_new, it+1, residual, true};
                }
            }
            return {x_new, max_iter, 0.0, false};
        }
} //namespace cusolve
