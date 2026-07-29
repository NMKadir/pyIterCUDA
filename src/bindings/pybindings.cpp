#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "cusolve/jacobi.hpp"
#include "cusolve/jacobi_cuda.hpp"

namespace py = pybind11;

namespace cusolve {

// Convert a numpy 2D array (n x n) into the flat row-major vector<double>
// your C++/CUDA functions expect.
static std::vector<double> flatten(py::array_t<double, py::array::c_style | py::array::forcecast> A) {
    auto buf = A.request();
    if (buf.ndim != 2 || buf.shape[0] != buf.shape[1])
        throw std::runtime_error("A must be a square 2D array");
    double* ptr = static_cast<double*>(buf.ptr);
    return std::vector<double>(ptr, ptr + buf.shape[0] * buf.shape[1]);
}

} // namespace cusolve

PYBIND11_MODULE(_cusolve, m) {
    m.doc() = "CUDA iterative solver library — Python bindings";

    py::class_<cusolve::SolverResult>(m, "SolverResult")
        .def_readonly("x", &cusolve::SolverResult::x)
        .def_readonly("iterations", &cusolve::SolverResult::iterations)
        .def_readonly("residual_norm", &cusolve::SolverResult::residual_norm)
        .def_readonly("converged", &cusolve::SolverResult::converged);

    m.def("jacobi_cpu",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> A,
           std::vector<double> b, double tol, int max_iter) {
            int n = static_cast<int>(b.size());
            auto buf = A.request();
            // rebuild vector<vector<double>> for the existing CPU API
            std::vector<std::vector<double>> A_2d(n, std::vector<double>(n));
            double* ptr = static_cast<double*>(buf.ptr);
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    A_2d[i][j] = ptr[i * n + j];
            return cusolve::jacobi_solve(A_2d, b, tol, max_iter);
        },
        py::arg("A"), py::arg("b"), py::arg("tol") = 1e-8, py::arg("max_iter") = 1000,
        "Solve Ax=b using Jacobi iteration on CPU");

#ifdef CUSOLVE_HAS_CUDA
    m.def("jacobi_gpu",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> A,
           std::vector<double> b, double tol, int max_iter) {
            int n = static_cast<int>(b.size());
            auto A_flat = cusolve::flatten(A);
            return cusolve::jacobi_solve_cuda(A_flat, b, n, tol, max_iter);
        },
        py::arg("A"), py::arg("b"), py::arg("tol") = 1e-8, py::arg("max_iter") = 1000,
        "Solve Ax=b using Jacobi iteration on GPU (CUDA)");
#endif
}