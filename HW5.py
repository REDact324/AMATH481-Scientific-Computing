import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2
from scipy.integrate import solve_ivp
from scipy.linalg import lu, solve_triangular
from scipy.sparse.linalg import bicgstab, gmres
from scipy.sparse import spdiags, csr_matrix
import time


dx = 20/64
dy = 20/64

m = 64
n = m * m
e0 = np.zeros((n, 1))
e1 = np.ones((n, 1))
e2 = np.copy(e1)
e4 = np.copy(e0)

for j in range(1, m+1):
    e2[m * j - 1] = 0
    e4[m * j - 1] = 1

e3 = np.zeros_like(e2)
e3[1:n] = e2[0:n-1]
e3[0] = e2[n-1]
e5 = np.zeros_like(e4)
e5[1:n] = e4[0:n-1]
e5[0] = e4[n-1]

diagonals = [e1.flatten(), e1.flatten(), e5.flatten(),
e2.flatten(), -4 * e1.flatten(), e3.flatten(),
e4.flatten(), e1.flatten(), e1.flatten()]
offsets = [-(n - m), -m, -m+1, -1, 0, 1, m-1, m, (n - m)]
A = (1 / (dx ** 2)) * spdiags(diagonals, offsets, n, n).toarray()
A[0, 0] = 2 / (dx ** 2)

# MATRIX B
diagonals = [e1.flatten(), -e1.flatten(), e1.flatten(), -e1.flatten()]
offsets = [(-n + m), -m, m, n - m]
B = spdiags(diagonals, offsets, n, n,).toarray()
B = B / (2 * dx)

# MATRIX C
for i in range (1, n):
    e1[i] = e4[i - 1]

diagonals = [e1.flatten(), -e2.flatten(), e3.flatten(), -e4.flatten()]
offsets = [(-m + 1), -1, 1, m - 1]
C = spdiags(diagonals, offsets, n, n,).toarray()
C = C / (2 * dy)

##############################

tspan = np.arange(0, 4.5, 0.5)
nu = 0.001
Lx, Ly = 20, 20
nx, ny = 64, 64
N = nx * ny

x2 = np.linspace(-Lx / 2, Lx / 2, nx + 1)
x = x2[:nx]
y2 = np.linspace(-Ly / 2, Ly / 2, ny + 1)
y = y2[:ny]
X, Y = np.meshgrid(x, y)
w0 = np.exp (-X ** 2 - Y ** 2 / 20).flatten()  # 初始漩涡场，同时转成一维向量

kx = (2 * np.pi / Lx) * np.concatenate((np.arange(0, nx / 2), np.arange(-nx / 2, 0)))
kx[0] = 1e-6
ky = (2 * np.pi / Ly) * np.concatenate((np.arange(0, ny / 2), np.arange(-ny / 2, 0)))
ky[0] = 1e-6
Kx, Ky = np.meshgrid(kx, ky)
K = Kx ** 2 + Ky ** 2

################## a #################

def spc_rhs(t, wt2, nx, ny, N, Kx, Ky, K, nu):
    w = wt2.reshape((nx, ny))
    wt = fft2(w)

    # 求解psi
    psit = - wt / K
    psi = np.real(ifft2 (psit)).reshape(N)

    rhs = (
            nu * A.dot(wt2)
            + (B.dot(wt2)) * (C.dot(psi))
            - (B.dot(psi)) * (C.dot(wt2))
    )

    return rhs

start_time = time.time()

sol = solve_ivp(lambda t, w: spc_rhs(t, w, nx, ny, N, Kx, Ky, K, nu), (tspan[0], tspan[-1]), w0, t_eval=tspan, method='RK45')

end_time = time.time()
print(f"Elapsed time for FFT: {(end_time - start_time):.2f} seconds")

A1 = sol.y

# Movie time :P
plot = A1.copy()
n = int(np.sqrt(plot.shape[0]))

fig, ax = plt.subplots(figsize=(6, 6))
cax = ax.imshow (plot[:, 0].reshape((n, n)), extent=[-10, 10, -10, 10], cmap='jet')
fig.colorbar (cax, ax=ax, label='V')
ax.set_title ('FFT')
ax.set_xlabel ('x')
ax.set_ylabel ('y')

def update (frame):
    ax.set_title(f'V Field at t = {frame * 0.5:.2f}')
    cax.set_data(plot[:, frame].reshape ((n, n)))
    return cax,


anim = FuncAnimation(fig, update, frames=plot.shape[1], blit=True)
plt.show(anim)
# anim.save ('/Users/dongyueqi/Documents/Undergraduate/4Y/Fall/AMATH481/AMATH481-Homework/HW5-movie/FFT.gif', writer='imagemagick', fps=2)

############### b1 ################

def ab_rhs(t, w, A, B, C, nu):
    psi = np.linalg.solve(A, w)

    rhs = (
            nu * A.dot (w)
            + (B.dot (w)) * (C.dot (psi))
            - (B.dot (psi)) * (C.dot (w))
    )
    return rhs


start_time = time.time()

sol = solve_ivp(lambda t, w: ab_rhs(t, w, A, B, C, nu), (tspan[0], tspan[-1]), w0, t_eval=tspan, method='RK45')

end_time = time.time ()
print (f"Elapsed time for A/b: {(end_time - start_time):.2f} seconds")

A2 = sol.y

# Movie time :P
plot = A2.copy()
n = int(np.sqrt(plot.shape[0]))

fig, ax = plt.subplots(figsize=(6, 6))
cax = ax.imshow (plot[:, 0].reshape((n, n)), extent=[-10, 10, -10, 10], cmap='jet')
fig.colorbar (cax, ax=ax, label='V')
ax.set_title ('A/b')
ax.set_xlabel ('x')
ax.set_ylabel ('y')

def update (frame):
    ax.set_title(f'V Field at t = {frame * 0.5:.2f}')
    cax.set_data(plot[:, frame].reshape ((n, n)))
    return cax,


anim = FuncAnimation(fig, update, frames=plot.shape[1], blit=True)
plt.show(anim)
# anim.save ('/Users/dongyueqi/Documents/Undergraduate/4Y/Fall/AMATH481/AMATH481-Homework/HW5-movie/A_b.gif', writer='imagemagick', fps=2)

############## b2 ##############

start_time = time.time ()

P, L, U = lu(A)


def lu_rhs(t, w, A, B, C, nu, L, U, P):
    Pw = np.dot(P, w)
    y = solve_triangular(L, Pw, lower=True)
    psi = solve_triangular(U, y, lower=False)

    rhs = (
            nu * A.dot(w)  # nu * A * w
            + (B.dot(w)) * (C.dot(psi))
            - (B.dot(psi)) * (C.dot(w))
    )

    return rhs


sol = solve_ivp (lambda t, w: lu_rhs(t, w, A, B, C, nu, L, U, P), (tspan[0], tspan[-1]), w0, t_eval=tspan, method='RK45')

end_time = time.time ()
print (f"Elapsed time for LU: {(end_time - start_time):.2f} seconds")

A3 = sol.y

# Movie time :P
plot = A3.copy()
n = int(np.sqrt(plot.shape[0]))

fig, ax = plt.subplots(figsize=(6, 6))
cax = ax.imshow (plot[:, 0].reshape((n, n)), extent=[-10, 10, -10, 10], cmap='jet')
fig.colorbar (cax, ax=ax, label='V')
ax.set_title ('LU')
ax.set_xlabel ('x')
ax.set_ylabel ('y')

def update (frame):
    ax.set_title(f'V Field at t = {frame * 0.5:.2f}')
    cax.set_data(plot[:, frame].reshape ((n, n)))
    return cax,


anim = FuncAnimation(fig, update, frames=plot.shape[1], blit=True)
plt.show(anim)
# anim.save ('/Users/dongyueqi/Documents/Undergraduate/4Y/Fall/AMATH481/AMATH481-Homework/HW5-movie/LU.gif', writer='imagemagick', fps=2)

############## BICGSTAB ##############

A_csr = csr_matrix(A)


def bicgstab_rhs (t, w, A, B, C, nu):
    psi, info = bicgstab(A_csr, w, atol=1e-8, maxiter=1000)

    rhs = (
            nu * A.dot(w)  # nu * A * w
            + (B.dot(w)) * (C.dot(psi))
            - (B.dot(psi)) * (C.dot(w))
    )

    return rhs


start_time = time.time ()

sol = solve_ivp(lambda t, w: bicgstab_rhs(t, w, A, B, C, nu), (tspan[0], tspan[-1]), w0, t_eval=tspan, method='RK45')

end_time = time.time ()
print (f"Elapsed time for BICGSTAB: {(end_time - start_time):.2f} seconds")

BGB = sol.y

# Movie time :P
plot = BGB.copy()
n = int(np.sqrt(plot.shape[0]))

fig, ax = plt.subplots(figsize=(6, 6))
cax = ax.imshow (plot[:, 0].reshape((n, n)), extent=[-10, 10, -10, 10], cmap='jet')
fig.colorbar (cax, ax=ax, label='V')
ax.set_title ('LU')
ax.set_xlabel ('x')
ax.set_ylabel ('y')

def update (frame):
    ax.set_title(f'V Field at t = {frame * 0.5:.2f}')
    cax.set_data(plot[:, frame].reshape ((n, n)))
    return cax,


anim = FuncAnimation(fig, update, frames=plot.shape[1], blit=True)
plt.show(anim)
# anim.save ('/Users/dongyueqi/Documents/Undergraduate/4Y/Fall/AMATH481/AMATH481-Homework/HW5-movie/BICGSTAB.gif', writer='imagemagick', fps=2)

################ GMRES ################

def gmres_rhs(t, w, A, B, C, nu):
    psi, info = gmres(A_csr, w, atol=1e-8, restart=50, maxiter=1000)

    rhs = (
            nu * A.dot(w)
            + (B.dot(w)) * (C.dot(psi))
            - (B.dot(psi)) * (C.dot(w))
    )

    return rhs


start_time = time.time()

sol = solve_ivp(lambda t, w: gmres_rhs(t, w, A, B, C, nu), (tspan[0], tspan[-1]), w0, t_eval=tspan, method='RK45')

end_time = time.time()
print(f"Elapsed time for GMRES: {(end_time - start_time):.2f} seconds")

GMRES = sol.y

# Movie time :P
plot = GMRES.copy()
n = int(np.sqrt(plot.shape[0]))

fig, ax = plt.subplots(figsize=(6, 6))
cax = ax.imshow (plot[:, 0].reshape((n, n)), extent=[-10, 10, -10, 10], cmap='jet')
fig.colorbar (cax, ax=ax, label='V')
ax.set_title ('LU')
ax.set_xlabel ('x')
ax.set_ylabel ('y')

def update (frame):
    ax.set_title(f'V Field at t = {frame * 0.5:.2f}')
    cax.set_data(plot[:, frame].reshape ((n, n)))
    return cax,


anim = FuncAnimation(fig, update, frames=plot.shape[1], blit=True)
plt.show(anim)
# anim.save ('/Users/dongyueqi/Documents/Undergraduate/4Y/Fall/AMATH481/AMATH481-Homework/HW5-movie/GMRES.gif', writer='imagemagick', fps=2)
