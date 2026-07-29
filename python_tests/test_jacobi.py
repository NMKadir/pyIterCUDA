import numpy as np
import pytest
import cusolve


@pytest.fixture
def diagonally_dominant_system():
    A = np.array([[4, 1, -1], [2, 7, 1], [1, -3, 12]], dtype=np.float64)
    b = np.array([3, 19, 31], dtype=np.float64)
    expected_x = np.array([1.0, 2.0, 3.0])
    return A, b, expected_x


def test_jacobi_cpu_converges(diagonally_dominant_system):
    A, b, expected_x = diagonally_dominant_system
    result = cusolve.jacobi(A, b, backend="cpu")
    assert result.converged
    np.testing.assert_allclose(result.x, expected_x, atol=1e-4)


@pytest.mark.skipif(not cusolve._HAS_GPU, reason="built without CUDA")
def test_jacobi_gpu_converges(diagonally_dominant_system):
    A, b, expected_x = diagonally_dominant_system
    result = cusolve.jacobi(A, b, backend="gpu")
    assert result.converged
    np.testing.assert_allclose(result.x, expected_x, atol=1e-4)


@pytest.mark.skipif(not cusolve._HAS_GPU, reason="built without CUDA")
def test_cpu_gpu_agree(diagonally_dominant_system):
    A, b, _ = diagonally_dominant_system
    r_cpu = cusolve.jacobi(A, b, backend="cpu")
    r_gpu = cusolve.jacobi(A, b, backend="gpu")
    np.testing.assert_allclose(r_cpu.x, r_gpu.x, atol=1e-6)


def test_non_diagonally_dominant_may_not_converge():
    # sanity check: a badly conditioned system shouldn't silently "converge" to garbage
    A = np.array([[1, 2], [3, 1]], dtype=np.float64)
    b = np.array([5, 10], dtype=np.float64)
    result = cusolve.jacobi(A, b, max_iter=50, backend="cpu")
    assert not result.converged