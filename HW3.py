import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

####### a ########

def bvpfunc(x, y, k, eps):
    f1 = y[1]
    f2 = (k * x**2 - eps) * y[0]
    dydt = np.array([f1, f2])
    return dydt

# Manual trapezoidal integration function
def manual_trapz(y_values, x_values):
    # Calculate the integral using the trapezoidal rule
    integral = 0.0
    for i in range(len(x_values) - 1):
        h = x_values[i+1] - x_values[i]  # Calculate the width of the trapezoid
        integral += 0.5 * h * (y_values[i] + y_values[i+1])  # Area of the trapezoid
    return integral

# Parameters
eps_start = 0
eps = eps_start
xspan = np.arange(-4, 4.1, 0.1)
tol = 1e-4
k = 1

eig_val1 = []
eig_vec1 = []

for mode in range(1, 6):
    eps = eps_start
    deps = 0.1

    for i in range(1000):
        # Solve the ODE
        sol = solve_ivp(lambda x, y: bvpfunc(x, y, k, eps), [xspan[0], xspan[-1]], [1, np.sqrt(16 - eps)], t_eval=xspan)
        x = sol.t
        y = sol.y

        # Boundary condition check
        temp = y[1, -1] + np.sqrt(16 - eps) * y[0, -1]

        if abs(temp) < tol:
            eig_val1.append(eps)
            # Calculate the norm using the custom trapezoidal integration function
            integral = manual_trapz(y[0, :]**2, x)
            y_norm = y[0, :] / np.sqrt(integral)  # Normalize y using the computed integral
            eig_vec1.append(np.abs(y_norm))
            break

        if (-1)**(mode + 1) * temp > 0:
            eps += deps
        else:
            eps -= deps / 2
            deps /= 2

    eps_start = eps + 0.1

# Extracting eigenvectors and eigenvalues
A1 = np.array(eig_vec1).T
A1 = A1.tolist()
A2 = eig_val1


####### b #######

# Set parameters
dx = 0.1
xspan = np.arange(-4, 4 + dx, dx)
length = len(xspan)

# Initialize matrix A
A = np.zeros((79, 79))
A[0, 0] = 2 / 3 + dx ** 2 * xspan[1] ** 2
A[0, 1] = -2 / 3
for i in range(1, 78):
    A[i, i - 1] = -1
    A[i, i] = 2 + dx ** 2 * xspan[i + 1] ** 2
    A[i, i + 1] = -1
A[78, 77] = -2 / 3
A[78, 78] = 2 / 3 + dx ** 2 * xspan[length - 2] ** 2

# Calculate eigenvalues and eigenvectors
eig_vals, eig_vecs = np.linalg.eig(A)
eig_val2_sorted_indices = np.argsort(eig_vals / dx / dx)
eig_val2 = np.sort(eig_vals / dx / dx)[:5]

# Extract the corresponding eigenvectors
eig_vec2 = []
for i in range(5):
    ind = eig_val2_sorted_indices[i]
    temp = eig_vecs[:, ind]
    vec = np.concatenate((
        [(4 * temp[0] - 1 * temp[1]) / (3 + 2 * dx * np.sqrt(16 - eig_val2[i]))],
        temp,
        [(4 * temp[78] - 1 * temp[77]) / (3 + 2 * dx * np.sqrt(16 - eig_val2[i]))]
    ))
    eig_vec2.append(vec)

# Normalize the eigenvectors
eig_vec2 = np.array(eig_vec2).T
eig_vec2 = np.abs(eig_vec2 / np.sqrt(np.trapz(eig_vec2 ** 2, xspan, axis=0)))

A3 = eig_vec2.tolist()
A4 = eig_val2.tolist()

# ###### c #######

# L = 2
# gamma = 0.05
# xspan = np.arange(-L, L + 0.1, 0.1)

# A = 0.1
# eps_start = 0

# k = 1
# tol = 1e-4

# eig_val3 = []
# eig_vec3 = []

# def bvpfunc2(x, y, k, gamma, eps):
#     f1 = y[1]
#     f2 = (gamma * (abs(y[0])**2) + k * x**2 - eps) * y[0]
#     dydx = [f1, f2]
#     return dydx

# for modes in range(1, 3):
#     eps = eps_start
#     deps = 0.1

#     for i in range(1000):
#         yinit = [A, A * np.sqrt(k * L**2 - eps)]
#         sol = solve_ivp(lambda x, y: bvpfunc2(x, y, k, gamma, eps), [xspan[0], xspan[-1]], yinit, t_eval=xspan, method='RK45')
#         y = sol.y.T
#         y1 = y[:, 0]
#         y2 = y[:, 1]

#         # calculate norm
#         ynorm = np.trapz(y1**2, xspan)

#         if abs(ynorm - 1) < tol:
#             break
#         else:
#             A = A / np.sqrt(ynorm)

#         # eps shooting
#         temp = y[1, -1] + np.sqrt(L**2 - eps) * y[-1, 0]

#         if abs(temp) < tol:
#             break

#         if (-1)**(modes + 1) * temp > 0:
#             eps += deps
#         else:
#             eps -= deps / 2
#             deps /= 2

#     eig_val3.append(eps)
#     eig_vec3.append(y1)
#     eps_start = eps + 0.1

# A5 = np.array(eig_vec3).T
# A5 = A5.tolist()
# A6 = eig_val3

# gamma = -0.05

# eig_val4 = []
# eig_vec4 = []

# for modes in range(1, 3):
#     eps = eps_start
#     deps = 0.1

#     for i in range(1000):
#         yinit = [A, A * np.sqrt(k * L**2 - eps)]
#         sol = solve_ivp(lambda x, y: bvpfunc2(x, y, k, gamma, eps), [xspan[0], xspan[-1]], yinit, t_eval=xspan, method='RK45')
#         y = sol.y.T
#         y1 = y[:, 0]
#         y2 = y[:, 1]

#         # Calculate norm
#         ynorm = np.trapz(y1**2, xspan)

#         if abs(ynorm - 1) < tol:
#             break
#         else:
#             A = A / np.sqrt(ynorm)

#         # Epsilon shooting
#         temp = y[-1, 1] + np.sqrt(L**2 - eps) * y[-1, 0]

#         if abs(temp) < tol:
#             break

#         if (-1)**(modes + 1) * temp > 0:
#             eps += deps
#         else:
#             eps -= deps / 2
#             deps /= 2

#     eig_val4.append(eps)
#     eig_vec4.append(y1)
#     eps_start = eps + 0.1

# A7 = np.array(eig_vec3).T
# A7 = A7.tolist()
# A8 = eig_val3

####### d #######

# Set parameters
# L = 2
# gamma = 0
# x_span = [-L, L]

# A = 1
# E = 1
# k = 1
# TOL_values = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

# def hw1_rhs_a(x, y, E):
#     f1 = y[1]
#     f2 = (k * x**2 - E) * y[0]
#     dydx = [f1, f2]
#     return dydx

# average_step_sizes_rk45 = []
# average_step_sizes_rk23 = []

# for TOL in TOL_values:
#     options = {'rtol': TOL, 'atol': TOL}
#     y0 = [A, A * np.sqrt(k * L**2 - E)]

#     # Solve with RK45
#     sol_rk45 = solve_ivp(lambda x, y: hw1_rhs_a(x, y, E), x_span, y0, method='RK45', args=(E,), **options)
#     step_sizes_rk45 = np.diff(sol_rk45.t)
#     average_step_sizes_rk45.append(np.mean(step_sizes_rk45))

#     # Solve with RK23
#     sol_rk23 = solve_ivp(lambda x, y: hw1_rhs_a(x, y, E), x_span, y0, method='RK23', args=(E,), **options)
#     step_sizes_rk23 = np.diff(sol_rk23.t)
#     average_step_sizes_rk23.append(np.mean(step_sizes_rk23))

# # Plot average step-size vs tolerance on a log-log scale
# plt.figure()
# plt.loglog(average_step_sizes_rk45, TOL_values, label='RK45', marker='o')
# plt.loglog(average_step_sizes_rk23, TOL_values, label='RK23', marker='x')
# plt.xlabel('Average Step Size')
# plt.ylabel('Tolerance')
# plt.legend()
# plt.title('Convergence Study: Average Step Size vs Tolerance')
# plt.grid(True)
# plt.show()

# # Calculate the slopes using polyfit
# slope_rk45 = np.polyfit(np.log(average_step_sizes_rk45), np.log(TOL_values), 1)[0]
# slope_rk23 = np.polyfit(np.log(average_step_sizes_rk23), np.log(TOL_values), 1)[0]

# # Solve with Radau and BDF methods
# average_step_sizes_radau = []
# average_step_sizes_bdf = []

# for TOL in TOL_values:
#     options = {'rtol': TOL, 'atol': TOL}

#     # Solve with Radau
#     sol_radau = solve_ivp(lambda x, y: hw1_rhs_a(x, y, E), x_span, y0, method='Radau', args=(E,), **options)
#     step_sizes_radau = np.diff(sol_radau.t)
#     average_step_sizes_radau.append(np.mean(step_sizes_radau))

#     # Solve with BDF
#     sol_bdf = solve_ivp(lambda x, y: hw1_rhs_a(x, y, E), x_span, y0, method='BDF', args=(E,), **options)
#     step_sizes_bdf = np.diff(sol_bdf.t)
#     average_step_sizes_bdf.append(np.mean(step_sizes_bdf))

# # Calculate the slopes using polyfit
# slope_radau = np.polyfit(np.log(average_step_sizes_radau), np.log(TOL_values), 1)[0]
# slope_bdf = np.polyfit(np.log(average_step_sizes_bdf), np.log(TOL_values), 1)[0]

# # Save the slopes as a 4x1 vector
# A9 = np.array([slope_rk45, slope_rk23, slope_radau, slope_bdf])

# # Output slopes
# print("A9 (slopes of RK45, RK23, Radau, BDF):", A9)

####### e #########

A1_ = np.array(A1)
A2_ = np.array(A2)

xspan = np.arange(-4, 4.1, 0.1)

# Manual Hermite polynomial function
def hermite_polynomial(n, x):
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    else:
        H_n_minus_two = np.ones_like(x)
        H_n_minus_one = 2 * x
        for i in range(2, n + 1):
            H_n = 2 * x * H_n_minus_one - 2 * (i - 1) * H_n_minus_two
            H_n_minus_two = H_n_minus_one
            H_n_minus_one = H_n
        return H_n

# Exact Solutions
exact_eigenvalues = [1, 3, 5, 7, 9]  # Harmonic oscillator eigenvalues (2n + 1)
exact_eigenfunctions = [
    hermite_polynomial(n, xspan) * np.exp(-xspan**2 / 2) for n in range(5)
]

# Normalizing exact eigenfunctions
for i in range(5):
    norm = np.trapz(exact_eigenfunctions[i]**2, xspan)
    exact_eigenfunctions[i] = exact_eigenfunctions[i] / np.sqrt(norm)

exact_eigenfunctions = np.array(exact_eigenfunctions).T

# Calculating Errors
# Eigenfunction Errors (A10)
eigenfunction_errors = []
for n in range(5):
    diff = np.abs(A1_[:, n]) - np.abs(exact_eigenfunctions[:, n])
    error = np.sqrt(np.trapz(diff**2, xspan))
    eigenfunction_errors.append(error)
A10 = eigenfunction_errors

# Eigenvalue Errors (A11)
eigenvalue_errors = [
    100 * (np.abs(A2_[n] - exact_eigenvalues[n]) / exact_eigenvalues[n]) for n in range(5)
]
A11 = eigenvalue_errors

# Print Results
print("A10 (Eigenfunction Errors):", A10)
print("A11 (Eigenvalue Errors):", A11)

