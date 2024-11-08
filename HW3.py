import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import eigs
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

# ####### b #######

L=4
xspan = np.arange(-L, L + 0.1, 0.1)
n = len(xspan)
dx=xspan[1]-xspan[0]
A = np.zeros((n-2, n-2))

for j in range(n - 2):
    A[j, j] = -2 - (dx ** 2) * x[j + 1] ** 2
    if j < n - 3:
        A[j + 1, j] = 1
        A[j, j + 1] = 1

A[0, 0] = A[0, 0] + 4 / 3
A[0, 1] = A[0, 1] - 1 / 3
A[-1, -1] = A[-1, -1] + 4 / 3
A[-1, -2] = A[-1, -2] - 1 / 3

# Calculate eigenvalues and eigenvectors
eigvals, eigvecs = eigs(-A, k=5, which='SM')

# Normalize eigenvectors
vec = np.vstack([4/3 * eigvecs[0, :] -1/3 * eigvecs[1, :], eigvecs,
                4/3 * eigvecs[-1, :] - 1/3 * eigvecs[-2, :]])

ysol_b = np.zeros((n, 5))
esol_b = np.zeros(5)

for j in range(5):
    norm = np.sqrt(np.trapz(vec[:, j] ** 2, xspan))
    ysol_b[:, j] = np.abs(vec[:, j] / norm)

esol_b = np.sort(eigvals[:5]) / dx ** 2
# Sort eigenvalues and eigenvectors
A3=ysol_b
A4=esol_b

print(A2)
print(A4)
print(A1)
print(A3)

###### c #######

L = 2
gamma = 0.05
xspan = np.arange(-L, L + 0.1, 0.1)

A = 0.1
eps_start = 0

k = 1
tol = 1e-4

# Arrays to store eigenvalues and eigenvectors
eig_vals = []
eig_vecs = []

# Function for the boundary value problem
def bvpfunc2(x, y, k, gamma, eps):
    f1 = y[1]
    f2 = (gamma * (abs(y[0])**2) + k * x**2 - eps) * y[0]
    dydx = np.array([f1, f2])
    return dydx

for modes in range(1, 3):
    eps = eps_start
    deps = 0.1

    for i in range(1000):
        yinit = [A, A * np.sqrt(k * L ** 2 - eps)]

        # Solving the ODE
        sol = solve_ivp(lambda x, y: bvpfunc2(x, y, k, gamma, eps), [xspan[0], xspan[-1]], yinit, t_eval=xspan, rtol=1e-12, atol=1e-12)
        y1 = sol.y[0]
        y2 = sol.y[1]

        # Calculating the norm
        ynorm = np.trapz(y1 ** 2, xspan)

        if abs(ynorm - 1) < tol:
            break
        else:
            A = A / np.sqrt(ynorm)

        # Eps shooting
        temp = y2[-1] + np.sqrt(L ** 2 - eps) * y1[-1]

        if abs(temp) < tol:
            break

        if (-1) ** (modes + 1) * temp > 0:
            eps = eps + deps
        else:
            eps = eps - deps / 2
            deps = deps / 2

    eig_vals.append(eps)
    eig_vecs.append(np.abs(y1))
    eps_start = eps + 0.1

A5 = np.array(np.abs(eig_vecs)).T.tolist()
A6 = eig_vals

# Parameters
L = 2
gamma = -0.05
xspan = np.arange(-L, L + 0.1, 0.1)

A = 0.1
eps_start = 0

k = 1
tol = 1e-4

# Arrays to store eigenvalues and eigenvectors
eig_vals = []
eig_vecs = []

for modes in range(1, 3):
    eps = eps_start
    deps = 0.1

    for i in range(1000):
        yinit = [A, A * np.sqrt(k * L ** 2 - eps)]

        # Solving the ODE
        sol = solve_ivp(lambda x, y: bvpfunc2(x, y, k, gamma, eps), [xspan[0], xspan[-1]], yinit, t_eval=xspan, rtol=1e-12, atol=1e-12)
        y1 = sol.y[0]
        y2 = sol.y[1]

        # Calculating the norm
        ynorm = np.trapz(y1 ** 2, xspan)

        if abs(ynorm - 1) < tol:
            break
        else:
            A = A / np.sqrt(ynorm)

        # Eps shooting
        temp = y2[-1] + np.sqrt(L ** 2 - eps) * y1[-1]

        if abs(temp) < tol:
            break

        if (-1) ** (modes + 1) * temp > 0:
            eps = eps + deps
        else:
            eps = eps - deps / 2
            deps = deps / 2

    eig_vals.append(eps)
    eig_vecs.append(np.abs(y1))
    eps_start = eps + 0.1

A7 = np.array(np.abs(eig_vecs)).T.tolist()
A8 = eig_vals

print(A5, A7)

###### d #######

# Set parameters
L = 2
gamma = 0
x_span = [-L, L]

A = 1
E = 1
k = 1
TOL_values = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

def hw1_rhs_a(x, y, E):
    f1 = y[1]
    f2 = (k * x**2 - E) * y[0]
    dydx = [f1, f2]
    return dydx

average_step_sizes_rk45 = []
average_step_sizes_rk23 = []

for TOL in TOL_values:
    options = {'rtol': TOL, 'atol': TOL}
    y0 = [A, A * np.sqrt(k * L**2 - E)]

    # Solve with RK45
    sol_rk45 = solve_ivp(lambda x, y: hw1_rhs_a(x, y, E), x_span, y0, method='RK45', **options)
    step_sizes_rk45 = np.diff(sol_rk45.t)
    average_step_sizes_rk45.append(np.mean(step_sizes_rk45))

    # Solve with RK23
    sol_rk23 = solve_ivp(lambda x, y: hw1_rhs_a(x, y, E), x_span, y0, method='RK23', **options)
    step_sizes_rk23 = np.diff(sol_rk23.t)
    average_step_sizes_rk23.append(np.mean(step_sizes_rk23))

# Plot average step-size vs tolerance on a log-log scale
plt.figure()
plt.loglog(average_step_sizes_rk45, TOL_values, label='RK45', marker='o')
plt.loglog(average_step_sizes_rk23, TOL_values, label='RK23', marker='x')
plt.xlabel('Average Step Size')
plt.ylabel('Tolerance')
plt.legend()
plt.title('Convergence Study: Average Step Size vs Tolerance')
plt.grid(True)
plt.show()

# Calculate the slopes using polyfit
slope_rk45 = np.polyfit(np.log(average_step_sizes_rk45), np.log(TOL_values), 1)[0]
slope_rk23 = np.polyfit(np.log(average_step_sizes_rk23), np.log(TOL_values), 1)[0]

# Solve with Radau and BDF methods
average_step_sizes_radau = []
average_step_sizes_bdf = []

for TOL in TOL_values:
    options = {'rtol': TOL, 'atol': TOL}

    # Solve with Radau
    sol_radau = solve_ivp(lambda x, y: hw1_rhs_a(x, y, E), x_span, y0, method='Radau', **options)
    step_sizes_radau = np.diff(sol_radau.t)
    average_step_sizes_radau.append(np.mean(step_sizes_radau))

    # Solve with BDF
    sol_bdf = solve_ivp(lambda x, y: hw1_rhs_a(x, y, E), x_span, y0, method='BDF', **options)
    step_sizes_bdf = np.diff(sol_bdf.t)
    average_step_sizes_bdf.append(np.mean(step_sizes_bdf))

# Calculate the slopes using polyfit
slope_radau = np.polyfit(np.log(average_step_sizes_radau), np.log(TOL_values), 1)[0]
slope_bdf = np.polyfit(np.log(average_step_sizes_bdf), np.log(TOL_values), 1)[0]

# Save the slopes as a 4x1 vector
A9 = np.array([slope_rk45, slope_rk23, slope_radau, slope_bdf])

####### e #########

A1_ = np.array(A1)
A2_ = np.array(A2)
A3_ = np.array(A3)
A4_ = np.array(A4)

L = 4

def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result
x=  np.linspace(-L, L,81)

# Initialize h array
h = np.array([np.ones_like(x), 2*x, 4*x**2 - 2, 8*x**3 - 12*x, 16*x**4 - 48*x**2 + 12])
# Initialize phi
phi = np.zeros((len(x), 5))
for j in range(5):
    phi[:, j] = (np.exp(-x**2/2) * h[j, :] / np.sqrt(2**j * np.sqrt(np.pi) * factorial(j))).T

# Initialize error arrays
epsi_a = np.zeros(5)
epsi_b = np.zeros(5)
er_a = np.zeros(5)
er_b = np.zeros(5)

# Calculate errors
for j in range(5):
    epsi_a[j] = np.trapz(((np.abs(A1_[:, j]) - np.abs(phi[:, j]))**2),x)
    epsi_b[j] = np.trapz(((np.abs(A3_[:, j]) - np.abs(phi[:, j]))**2),x)
    er_a[j] = 100 * np.abs(A2_[j] - (2 * j + 1)) / (2 * j + 1)
    er_b[j] = 100 * np.abs(A4_[j] - (2 * j + 1)) / (2 * j + 1)

# Store results
A10 = epsi_a
A12 = epsi_b
A11 = er_a
A13 = er_b

# Output the results
print("Eigenfunction Errors (A10):", A10)
print("Eigenvalue Errors (A11):", A11)
print("Eigenfunction Errors (A12):", A12)
print("Eigenvalue Errors (A13):", A13)
