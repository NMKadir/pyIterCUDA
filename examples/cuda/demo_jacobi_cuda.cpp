#include "cusolve/jacobi_cuda.hpp"
#include<iostream>

int main(){
    int n=3;
    std::vector<double> A_flat = {
        4,1,-1,
        2,7,1,
        1,-3,12
    };
    std::vector<double> b = {3,19,31};
    auto result = cusolve::jacobi_solve_cuda(A_flat, b, n);

    std::cout << "Converged: " << result.converged << "\n";
    std::cout << "Iterations: " << result.iterations << "\n";

    for(double xi : result.x) std::cout << xi << " ";
    std::cout << "\n";
}