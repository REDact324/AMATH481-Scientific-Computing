import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.fftpack import fft2, ifft2
from scipy.integrate import solve_ivp
from scipy.linalg import lu, solve_triangular
from scipy.sparse.linalg import bicgstab, gmres
from scipy.sparse import spdiags, csr_matrix
import time

colors = ["#0D0D0D", "#B33527", "#F9B72E", "#E4E2E3", "#66CEC8", "#274966"]
cmap = LinearSegmentedColormap.from_list('mycmap', colors)

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

# ################## a #################

# def spc_rhs(t, wt2, nx, ny, N, Kx, Ky, K, nu):
#     w = wt2.reshape((nx, ny))
#     wt = fft2(w)

#     # 求解psi
#     psit = - wt / K
#     psi = np.real(ifft2 (psit)).reshape(N)

#     rhs = (
#             nu * A.dot(wt2)
#             + (B.dot(wt2)) * (C.dot(psi))
#             - (B.dot(psi)) * (C.dot(wt2))
#     )

#     return rhs

# start_time = time.time()

# sol = solve_ivp(lambda t, w: spc_rhs(t, w, nx, ny, N, Kx, Ky, K, nu), (tspan[0], tspan[-1]), w0, t_eval=tspan, method='RK45')

# end_time = time.time()
# print(f"Elapsed time for FFT: {(end_time - start_time):.2f} seconds")

# A1 = sol.y

# # Movie time :P
# plot = A1.copy()
# n = int(np.sqrt(plot.shape[0]))

# fig, ax = plt.subplots(figsize=(6, 6))
# cax = ax.imshow (plot[:, 0].reshape((n, n)), extent=[-10, 10, -10, 10], cmap=cmap)
# fig.colorbar (cax, ax=ax, label='V')
# ax.set_title ('FFT')
# ax.set_xlabel ('x')
# ax.set_ylabel ('y')

# def update (frame):
#     ax.set_title(f'V Field at t = {frame * 0.5:.2f}')
#     cax.set_data(plot[:, frame].reshape ((n, n)))
#     return cax,


# anim = FuncAnimation(fig, update, frames=plot.shape[1], blit=True)
# plt.show(anim)
# anim.save ('./HW5-movie/FFT.gif', writer='imagemagick', fps=2)

# ############### b1 ################

# def ab_rhs(t, w, A, B, C, nu):
#     psi = np.linalg.solve(A, w)

#     rhs = (
#             nu * A.dot (w)
#             + (B.dot (w)) * (C.dot (psi))
#             - (B.dot (psi)) * (C.dot (w))
#     )
#     return rhs


# start_time = time.time()

# sol = solve_ivp(lambda t, w: ab_rhs(t, w, A, B, C, nu), (tspan[0], tspan[-1]), w0, t_eval=tspan, method='RK45')

# end_time = time.time ()
# print (f"Elapsed time for A/b: {(end_time - start_time):.2f} seconds")

# A2 = sol.y

# # Movie time :P
# plot = A2.copy()
# n = int(np.sqrt(plot.shape[0]))

# fig, ax = plt.subplots(figsize=(6, 6))
# cax = ax.imshow (plot[:, 0].reshape((n, n)), extent=[-10, 10, -10, 10], cmap=cmap)
# fig.colorbar (cax, ax=ax, label='V')
# ax.set_title ('A/b')
# ax.set_xlabel ('x')
# ax.set_ylabel ('y')

# def update (frame):
#     ax.set_title(f'V Field at t = {frame * 0.5:.2f}')
#     cax.set_data(plot[:, frame].reshape ((n, n)))
#     return cax,


# anim = FuncAnimation(fig, update, frames=plot.shape[1], blit=True)
# plt.show(anim)
# anim.save ('./HW5-movie/A_b.gif', writer='imagemagick', fps=2)

# ############## b2 ##############

# start_time = time.time ()

# P, L, U = lu(A)


# def lu_rhs(t, w, A, B, C, nu, L, U, P):
#     Pw = np.dot(P, w)
#     y = solve_triangular(L, Pw, lower=True)
#     psi = solve_triangular(U, y, lower=False)

#     rhs = (
#             nu * A.dot(w)  # nu * A * w
#             + (B.dot(w)) * (C.dot(psi))
#             - (B.dot(psi)) * (C.dot(w))
#     )

#     return rhs


# sol = solve_ivp (lambda t, w: lu_rhs(t, w, A, B, C, nu, L, U, P), (tspan[0], tspan[-1]), w0, t_eval=tspan, method='RK45')

# end_time = time.time ()
# print (f"Elapsed time for LU: {(end_time - start_time):.2f} seconds")

# A3 = sol.y

# # Movie time :P
# plot = A3.copy()
# n = int(np.sqrt(plot.shape[0]))

# fig, ax = plt.subplots(figsize=(6, 6))
# cax = ax.imshow (plot[:, 0].reshape((n, n)), extent=[-10, 10, -10, 10], cmap=cmap)
# fig.colorbar (cax, ax=ax, label='V')
# ax.set_title ('LU')
# ax.set_xlabel ('x')
# ax.set_ylabel ('y')

# def update (frame):
#     ax.set_title(f'V Field at t = {frame * 0.5:.2f}')
#     cax.set_data(plot[:, frame].reshape ((n, n)))
#     return cax,


# anim = FuncAnimation(fig, update, frames=plot.shape[1], blit=True)
# plt.show(anim)
# anim.save ('./HW5-movie/LU.gif', writer='imagemagick', fps=2)

# ############## BICGSTAB ##############

# A_csr = csr_matrix(A)


# def bicgstab_rhs (t, w, A, B, C, nu):
#     psi, info = bicgstab(A_csr, w, atol=1e-8, maxiter=1000)

#     rhs = (
#             nu * A.dot(w)  # nu * A * w
#             + (B.dot(w)) * (C.dot(psi))
#             - (B.dot(psi)) * (C.dot(w))
#     )

#     return rhs


# start_time = time.time ()

# sol = solve_ivp(lambda t, w: bicgstab_rhs(t, w, A, B, C, nu), (tspan[0], tspan[-1]), w0, t_eval=tspan, method='RK45')

# end_time = time.time ()
# print (f"Elapsed time for BICGSTAB: {(end_time - start_time):.2f} seconds")

# BGB = sol.y

# # Movie time :P
# plot = BGB.copy()
# n = int(np.sqrt(plot.shape[0]))

# fig, ax = plt.subplots(figsize=(6, 6))
# cax = ax.imshow (plot[:, 0].reshape((n, n)), extent=[-10, 10, -10, 10], cmap=cmap)
# fig.colorbar (cax, ax=ax, label='V')
# ax.set_title ('LU')
# ax.set_xlabel ('x')
# ax.set_ylabel ('y')

# def update (frame):
#     ax.set_title(f'V Field at t = {frame * 0.5:.2f}')
#     cax.set_data(plot[:, frame].reshape ((n, n)))
#     return cax,


# anim = FuncAnimation(fig, update, frames=plot.shape[1], blit=True)
# plt.show(anim)
# anim.save ('./HW5-movie/BICGSTAB.gif', writer='imagemagick', fps=2)

# ################ GMRES ################

# def gmres_rhs(t, w, A, B, C, nu):
#     psi, info = gmres(A_csr, w, atol=1e-8, restart=50, maxiter=1000)

#     rhs = (
#             nu * A.dot(w)
#             + (B.dot(w)) * (C.dot(psi))
#             - (B.dot(psi)) * (C.dot(w))
#     )

#     return rhs


# start_time = time.time()

# sol = solve_ivp(lambda t, w: gmres_rhs(t, w, A, B, C, nu), (tspan[0], tspan[-1]), w0, t_eval=tspan, method='RK45')

# end_time = time.time()
# print(f"Elapsed time for GMRES: {(end_time - start_time):.2f} seconds")

# GMRES = sol.y

# # Movie time :P
# plot = GMRES.copy()
# n = int(np.sqrt(plot.shape[0]))

# fig, ax = plt.subplots(figsize=(6, 6))
# cax = ax.imshow (plot[:, 0].reshape((n, n)), extent=[-10, 10, -10, 10], cmap=cmap)
# fig.colorbar (cax, ax=ax, label='V')
# ax.set_title ('LU')
# ax.set_xlabel ('x')
# ax.set_ylabel ('y')

# def update (frame):
#     ax.set_title(f'V Field at t = {frame * 0.5:.2f}')
#     cax.set_data(plot[:, frame].reshape ((n, n)))
#     return cax,


# anim = FuncAnimation(fig, update, frames=plot.shape[1], blit=True)
# plt.show(anim)
# anim.save ('./HW5-movie/GMRES.gif', writer='imagemagick', fps=2)

############## c #############

def change_initial(save_file_name, omega_equ):
    nu = 0.001
    tspan = np.linspace(0, 4, 9)

    Lx, Ly = 20, 20
    nx, ny = 64, 64
    N2 = nx * ny

    x2 = np.linspace(-Lx/2, Lx/2, nx + 1); x2 = x2[:nx]
    y2 = np.linspace(-Ly/2, Ly/2, ny + 1); y2 = y2[:ny]
    X, Y = np.meshgrid(x2, y2)

    w = omega_equ
    w0 = w.reshape(N2)

    # Define spectral k values
    kx = ((2 * np.pi) / Lx) * np.concatenate((np.arange(0, nx/2), np.arange(-nx/2, 0)))
    ky = ((2 * np.pi) / Ly) * np.concatenate((np.arange(0, ny/2), np.arange(-ny/2, 0)))
    kx[0] = 1e-6; ky[0] = 1e-6
    KX, KY = np.meshgrid(kx, ky)
    K = KX ** 2 + KY ** 2


    start_time = time.time()
    # Using FFT
    def fft_rhs(t, w2):
        w = w2.reshape((nx, ny))
        wt = fft2(w)
        psi_t = - wt / K
        psi = np.real(ifft2(psi_t)).reshape(N2)
        rhs = nu * np.dot(A, w2) + (np.dot(B, w2)) * (np.dot(C, psi)) - (np.dot(B, psi)) * (np.dot(C, w2))
        return rhs

    wsol2 = solve_ivp(fft_rhs, [tspan[0], tspan[-1]], w0, t_eval=tspan, method='RK45')
    wsol = wsol2.y

    end_time = time.time()
    print(f"Elapsed time in using FFT: {(end_time - start_time):.2f} seconds")


    plot = wsol.copy()
    n = int(np.sqrt(plot.shape[0]))

    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.imshow (plot[:, 0].reshape((n, n)), extent=[-10, 10, -10, 10], cmap=cmap)
    fig.colorbar (cax, ax=ax, label='V')
    ax.set_title ('FFT')
    ax.set_xlabel ('x')
    ax.set_ylabel ('y')

    def update (frame):
        ax.set_title(f'V Field at t = {frame * 0.5:.2f}')
        cax.set_data(plot[:, frame].reshape ((n, n)))
        return cax,

    anim = FuncAnimation(fig=fig, func=update, frames=len(tspan), interval=200, blit=True)
    plt.show(anim)
    # anim.save ('./HW5-movie/'+save_file_name, writer='imagemagick', fps=2)
    plt.close()

# Two oppositely “charged” Gaussian vorticies next to each other, i.e. one with positive amplitude, the other with negative amplitude.
change_initial('FFT(Oppositely_“charged”).gif', 1 * np.exp(-(X-5) ** 2 - ((Y-5) ** 2)/20) - 1 * np.exp(-(X+5) ** 2 - ((Y+5) ** 2)/20))

# Two same “charged” Gaussian vorticies next to each other.
change_initial('FFT(Same_“charged”).gif', 1 * np.exp(-(X-5) ** 2 - ((Y-5) ** 2)/20) + 1 * np.exp(-(X+5) ** 2 - ((Y+5) ** 2)/20))

# Two pairs of oppositely “charged” vorticies which can be made to collide with each other.
change_initial('FFT(Opposite_collide).gif', 1 * np.exp(-(X-5) ** 2 - ((Y-5) ** 2)/20) - 1 * np.exp(-(X+5) ** 2 - ((Y+5) ** 2)/20)
                                            - 1 * np.exp(-(X+8) ** 2 - ((Y-8) ** 2)/20) + 1 * np.exp(-(X-8) ** 2 - ((Y+8) ** 2)/20))


# A random assortment (in position, strength, charge, ellipticity, etc.) of vorticies on the periodic domain. Try 10-15 vorticies and watch what happens.
# Random assortment of vortices
# np.random.seed(324)  # For reproducibility
num_vortices = 15
w4 = np.zeros_like(X)

for _ in range(num_vortices):
    x0 = np.random.uniform(-4, 4)             # Random x position
    y0 = np.random.uniform(-4, 4)             # Random y position
    amplitude = np.random.uniform(-3, 3)      # Random strength/charge (-3 to 3)
    ellipticity = np.random.uniform(0.3, 2)   # Random ellipticity (0.3 to 2)
    w = 1 * np.exp(-(X-x0) ** 2 - ((Y-y0) ** 2)/20)

change_initial('FFT(Random).gif', w)
