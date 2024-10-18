import numpy as np
from scipy.integrate import solve_ivp

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
