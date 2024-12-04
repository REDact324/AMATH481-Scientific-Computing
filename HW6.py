import numpy as np
from scipy.fftpack import fft2, ifft2
from scipy.integrate import solve_ivp

N = 64
L = 20
dx = L/N
dy = L/N
m = 1
beta = 1
D1 = 0.1
D2 = D1

L = 20
n = 64
beta = 1
D1, D2 = 0.1, 0.1
t_span = np.arange (0, 4.5, 0.5)

x = np.linspace (-L / 2, L / 2, n, endpoint=False)
y = np.linspace (-L / 2, L / 2, n, endpoint=False)
dx = L / n
dy = L / n
X, Y = np.meshgrid (x, y)

m = 1
angle = np.angle (X + 1j * Y)
radius = np.sqrt (X ** 2 + Y ** 2)
U0 = np.tanh (radius) * np.cos (m * angle - radius)
V0 = np.tanh (radius) * np.sin (m * angle - radius)


def lambda_A (U, V):
    A2 = np.abs (U) ** 2 + np.abs (V) ** 2
    return 1 - A2


def omega_A (U, V):
    A2 = np.abs (U) ** 2 + np.abs (V) ** 2
    return -beta * A2


def rhs (t, Z):

    U_hat = Z[:n ** 2].reshape ((n, n))
    V_hat = Z[n ** 2:].reshape ((n, n))

    U = ifft2 (U_hat)
    V = ifft2 (V_hat)

    dUdt_reaction = lambda_A (U, V) * U - omega_A (U, V) * V
    dVdt_reaction = omega_A (U, V) * U + lambda_A (U, V) * V

    dUdt_reaction_hat = fft2 (dUdt_reaction)
    dVdt_reaction_hat = fft2 (dVdt_reaction)

    kx = 2 * np.pi * np.fft.fftfreq (U.shape[1], d=dx)
    ky = 2 * np.pi * np.fft.fftfreq (U.shape[0], d=dy)
    KX, KY = np.meshgrid (kx, ky)
    laplacian_operator = -(KX ** 2 + KY ** 2)

    dUdt_diffusion_hat = D1 * laplacian_operator * U_hat
    dVdt_diffusion_hat = D2 * laplacian_operator * V_hat

    dUdt_hat = dUdt_reaction_hat + dUdt_diffusion_hat
    dVdt_hat = dVdt_reaction_hat + dVdt_diffusion_hat

    rhs = np.concatenate ([dUdt_hat.flatten (), dVdt_hat.flatten ()])
    return rhs

Z0 = np.concatenate ([fft2 (U0).flatten (), fft2 (V0).flatten ()])

sol = solve_ivp (rhs, (t_span[0], t_span[-1]), Z0, t_eval=t_span, method='RK45')

A1 = sol.y

print (A1.shape)
print (A1)

# Chebyshev differentiation matrix
def cheb (N):
    if N == 0:
        D = 0.;
        x = 1.
    else:
        n = np.arange (0, N + 1)
        x = np.cos (np.pi * n / N).reshape (N + 1, 1)
        c = (np.hstack (([2.], np.ones (N - 1), [2.])) * (-1) ** n).reshape (N + 1, 1)
        X = np.tile (x, (1, N + 1))
        dX = X - X.T
        D = np.dot (c, 1. / c.T) / (dX + np.eye (N + 1))
        D -= np.diag (np.sum(D.T, axis=0))
    return D, x.reshape (N + 1)


# Generate Chebyshev grid and differentiation matrix

D1 = D2 = 0.1
beta = 1
L = 20

n = 30
n2 = (n + 1) ** 2
D, x = cheb (n)
D[n, :] = 0
D[0, :] = 0
Dxx = np.dot (D, D) / ((L / 2) ** 2)
y = x

I = np.eye (len (Dxx))
L = np.kron (I, Dxx) + np.kron (Dxx, I)

X, Y = np.meshgrid (x, y)
X = X * 10
Y = Y * 10

t_span = np.arange (0, 4.5, 0.5)

m = 1
angle = np.angle (X + 1j * Y)
radius = np.sqrt (X ** 2 + Y ** 2)
U0 = np.tanh (radius) * np.cos (m * angle - radius)
V0 = np.tanh (radius) * np.sin (m * angle - radius)


# Reaction terms
def lambda_A (U, V):
    A2 = U ** 2 + V ** 2
    return 1 - A2


def omega_A (U, V):
    A2 = U ** 2 + V ** 2
    return -beta * A2


def rhs (t, uv_t):
    n_rhs = n + 1

    ut, vt = uv_t[:n_rhs ** 2], uv_t[n_rhs ** 2:]

    dUdt = (lambda_A (ut, vt) * ut - omega_A (ut, vt) * vt) + D1 * (L @ ut)
    dVdt = (omega_A (ut, vt) * ut + lambda_A (ut, vt) * vt) + D2 * (L @ vt)

    return np.concatenate ([dUdt, dVdt])


# Initial conditions
Z0 = np.concatenate ([U0.reshape (n2), V0.reshape (n2)])

# Solve the system
sol = solve_ivp (rhs, (t_span[0], t_span[-1]), Z0, t_eval=t_span, method='RK45')

A2 = sol.y

print (A2.shape)
print (A2)
