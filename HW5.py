import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import bicgstab, gmres, splu
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
A = A.toarray()


Dx = first_derivative_matrix(n, h)
B = kron(Dx, eye(n))
B = B.toarray()


Dy = first_derivative_matrix(n, h)
C = kron(eye(n), Dy)
C = C.toarray()

# Domain and grid parameters
Lx, Ly = 20, 20  # Domain size
Nx, Ny = 64, 64  # Number of grid points
dx, dy = Lx / Nx, Ly / Ny  # Grid spacing
x = np.linspace(-Lx / 2, Lx / 2, Nx)
y = np.linspace(-Ly / 2, Ly / 2, Ny)
X, Y = np.meshgrid(x, y)

# Initial condition: Gaussian vorticity
omega_0 = np.exp(-1 * X**2 - Y**2 / 20)

# Parameters
nu = 0.001  # Diffusion coefficient
t_span = (0, 4)  # Time span
t_eval = np.arange(0, 4.5, 0.5)  # Evaluation times

# Wavenumbers for FFT
kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
kx[0] = ky[0] = 1e-6  # Avoid division by zero
KX, KY = np.meshgrid(kx, ky)
k2 = KX**2 + KY**2



########################

