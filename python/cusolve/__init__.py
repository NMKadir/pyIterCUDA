"""cusolve: CUDA-accelerated iterative solver library."""
import numpy as np
from ._cusolve import jacobi_cpu

try:
    from ._cusolve import jacobi_gpu
    _HAS_GPU = True
except ImportError:
    _HAS_GPU = False


def jacobi(A, b, tol=1e-8, max_iter=1000, backend="cpu"):
    """
    Solve Ax = b using Jacobi iteration.

    Parameters
    ----------
    A : array_like, shape (n, n)
    b : array_like, shape (n,)
    backend : "cpu" or "gpu"

    Returns
    -------
    SolverResult with .x, .iterations, .residual_norm, .converged
    """
    A = np.ascontiguousarray(A, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64).tolist()

    if backend == "gpu":
        if not _HAS_GPU:
            raise RuntimeError("GPU backend not available — built without CUDA")
        return jacobi_gpu(A, b, tol, max_iter)
    elif backend == "cpu":
        return jacobi_cpu(A, b, tol, max_iter)
    else:
        raise ValueError(f"Unknown backend: {backend}")


__all__ = ["jacobi"]