import numpy as np
from scipy.sparse import diags, kron, eye

L = 20
n = 8
h = L / n

def laplacian_matrix(n, h):
    diagonals = [-2 * np.ones(n), np.ones(n-1), np.ones(n-1)]
    offsets = [0, 1, -1]
    D2 = diags(diagonals, offsets, shape=(n, n)).toarray()
    D2[0, -1] = 1
    D2[-1, 0] = 1
    return D2 / h**2


def first_derivative_matrix(n, h):
    diagonals = [np.ones(n-1), -np.ones(n-1)]
    offsets = [1, -1]
    D = diags(diagonals, offsets, shape=(n, n)).toarray()
    D[0, -1] = -1
    D[-1, 0] = 1
    return D / (2 * h)


D2x = laplacian_matrix(n, h)
D2y = laplacian_matrix(n, h)
A = kron(D2x, eye(n)) + kron(eye(n), D2y)
A1 = A.toarray()


Dx = first_derivative_matrix(n, h)
B = kron(Dx, eye(n))
A2 = B.toarray()


Dy = first_derivative_matrix(n, h)
C = kron(eye(n), Dy)
A3 = C.toarray()


# Output the matrices
print("Matrix A1 (Second Derivative):")
print(A1)
print("\nMatrix A2 (First Derivative d/dx):")
print(A2)
print("\nMatrix A3 (First Derivative d/dy):")
print(A3)


