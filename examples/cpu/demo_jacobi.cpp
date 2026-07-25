#include "cusolve/jacobi.hpp"
#include <iostream>

int main(){

    std::vector<std::vector<double>> A = {
        {4, 1, -1},
        {2, 7, 1},
        {1, -3, 12}
    };
    std::vector<double> b = {3,19,31};

    auto result = cusolve::jacobi_solve(A,b);

    std::cout << "Converged: " << result.converged << "\n";
    std::cout << "Iterations: " << result.iterations << "\n";
    for(double xi : result.x) std::cout << xi << " ";
    std::cout << "\n";
}