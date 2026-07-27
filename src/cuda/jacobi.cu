#include "cusolve/jacobi.hpp"
#include "cusolve/jacobi_cuda.hpp"
#include<cuda_runtime.h>
#include<cmath>
#include<vector>
#include<stdexcept>

#define CUDA_CHECK(call) \
    do{ \
        cudaError_t err = call; \
        if (err != cudaSuccess) \
            throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err)); \
    } while(0)

namespace cusolve{
    __global__ void jacobi_kernel(const double *A, const double *b, const double *x_old, double *x_new, int n){
        int i = blockIdx.x*blockDim.x+threadIdx.x;

        if(i>=n) return;

        double sigma = 0.0;
        for(int j=0; j<n; ++j){
            if(j!=i) sigma += A[i*n+j] * x_old[j];
        }
        x_new[i] = (b[i] - sigma) / A[i*n+i];
    }

    __global__ void residual_sq_kernel(const double *A, const double *b, const double *x, double *partial, int n){
        int i = blockIdx.x*blockDim.x+threadIdx.x;

        if(i>=n) return;

        double r_i = b[i];
        for(int j=0; j<n; ++j) r_i -= A[i*n+j] * x[j];
        partial[i] = r_i*r_i;
    }

    SolverResult jacobi_solve_cuda(
        const std::vector<double>& A_flat,
            const std::vector<double>& b,
            int n,
            double tol,
            int max_iter 
    )
    {
        double *d_A, *d_b, *d_x_old, *d_x_new, *d_partial;
        size_t bytes = n*sizeof(double);
        CUDA_CHECK(cudaMalloc((void **)&d_A, n*bytes));
        CUDA_CHECK(cudaMalloc((void **)&d_b, bytes));
        CUDA_CHECK(cudaMalloc((void **)&d_x_old, bytes));
        CUDA_CHECK(cudaMalloc((void **)&d_x_new, bytes));
        CUDA_CHECK(cudaMalloc((void **)&d_partial, bytes));

        CUDA_CHECK(cudaMemcpy(d_A, A_flat.data(), n*bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_x_old, 0, bytes));

        int block_size = 256;
        int grid_size = (n+block_size-1)/block_size;

        std::vector<double> partial(n);
        int iter=0;
        double residual = -1.0;
        bool converged = false;

        for(iter=0; iter<max_iter; ++iter){
            jacobi_kernel<<<grid_size, block_size>>>(d_A, d_b, d_x_old, d_x_new, n);
            CUDA_CHECK(cudaGetLastError());

            residual_sq_kernel<<<grid_size, block_size>>>(d_A, d_b, d_x_new, d_partial, n);
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaMemcpy(partial.data(), d_partial, bytes, cudaMemcpyDeviceToHost));
            double sum = 0.0;
            for(double v:partial) sum+=v;
            residual = std::sqrt(sum);
            std::swap(d_x_old, d_x_new);

            if(residual<tol){converged = true; ++iter; break; }
        }

        std::vector<double> x(n);
        CUDA_CHECK(cudaMemcpy(x.data(), d_x_old, bytes, cudaMemcpyDeviceToHost));
        cudaFree(d_A);
        cudaFree(d_b);
        cudaFree(d_x_old);
        cudaFree(d_x_new);
        cudaFree(d_partial);

        return SolverResult{x, converged ? iter : max_iter, residual, converged};
    }
} //namespcae cusolve