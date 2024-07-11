import time
import numpy as np
from numpy.linalg import inv
from numpy.typing import NDArray
from contextlib import contextmanager
from scipy.linalg import lu_factor, lu_solve

@contextmanager
def timing(label: str):
    start = time.time()
    yield
    end = time.time()
    print(f"{label}: {end - start:.6f} sec")

def LU_decomposition(A: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    n = A.shape[0]
    L = np.zeros_like(A)
    U = np.zeros_like(A)
    
    for i in range(n):
        L[i, i] = 1
        for j in range(i, n):
            U[i, j] = A[i, j] - L[i, :i] @ U[:i, j]
        for j in range(i+1, n):
            L[j, i] = (A[j, i] - L[j, :i] @ U[:i, i]) / U[i, i]
    return L, U

def forward_substitution(L: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    n = L.shape[0]
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - L[i, :i] @ y[:i]
    return y

def backward_substitution(U: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - U[i, i+1:] @ x[i+1:]) / U[i, i]
    return x

def LU_solve(A: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    L, U = LU_decomposition(A)
    y = forward_substitution(L, b)
    x = backward_substitution(U, y)
    return x

def LU_solve_scipy(A: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    lu, piv = lu_factor(A)
    return lu_solve((lu, piv), b)

if __name__ == "__main__":
    np.random.seed(0)
    size = 100
    
    A = np.random.rand(size, size) * 10
    b = np.random.rand(size) * 10
    
    with timing("LU\t"):
        solution = LU_solve(A, b)
        print("LU\t:", solution[:5])
    
    with timing("LU_scipy\t"):
        solution = LU_solve_scipy(A, b)
        print("LU_scipy\t:", solution[:5])
    
    with timing("Numpy\t"):
        solution = np.linalg.solve(A, b)
        print("Numpy\t:", solution[:5])
    
    with timing("Exact\t"):
        print("Exact\t:", (inv(A) @ b)[:5])
