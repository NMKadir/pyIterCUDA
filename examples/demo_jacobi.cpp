#include "cusolve/jacobi.hpp"
#include <iostream>

int main(){

    std::vector<std::vector<double>> A = {
        {10, -1, 2},
        {-1, 11, -1},
        {2, -1, 10}
    };
    std::vector<double> b = {9,8,22};

    auto result = cusolve::jacobi_solve(A,b);

    std::cout << "Converged: " << result.converged << "\n";
    std::cout << "Iterations: " << result.iterations << "\n";
    for(double xi : result.x) std::cout << xi << " ";
    std::cout << "\n";
}