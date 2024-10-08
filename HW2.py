import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

L = 4
xspan = np.arange(-L, L+0.1, 0.1)
K = 1
n_eigenfunctions = 5 # number of eignfunctions

eigenvalues = []
eigenfunctions = []

# Define the differential equation system
def schrodinger_equation(x, y, eps):
    """
    Returns the system of equations representing the Schrodinger equation.
    y[0] = ϕ, dy[1]/dx = d2ϕ/dx2
    eps = eigenvalue parameter (epsilon_n)
    """
    return np.vstack([y[1], (K * x**2 - eps) * y[0]])

def boundary_conditions(ya, yb, eps):
    """
    Boundary conditions: We want the wave function to vanish at both ends (ya, yb).
    """
    return np.array([ya[0], ya[1] - 0.1,yb[0]])

# Initial guess for the eigenfunctions
def initial_guess(x):
    """
    Provide an initial guess for the shooting method.
    """
    return np.vstack([np.exp(-x**2), -2*x*np.exp(-x**2)])

for n in range(n_eigenfunctions):
    # Initial guess for the eigenvalue (epsilon_n)
    eps_guess = n * 2 + 1  # Rough estimate for eigenvalues (for n-th eigenstate)

    # Solve boundary value problem for each eigenvalue
    solution = solve_bvp(schrodinger_equation,
                         boundary_conditions, xspan, initial_guess(xspan), p=[eps_guess])

    # Store the results
    eigenvalues.append(solution.p[0])  # eps (eigenvalue)
    eigenfunctions.append(solution.sol(xspan)[0])  # ϕn (eigenfunction)

# Convert to numpy arrays for easier manipulation
eigenvalues = np.array(eigenvalues)
eigenfunctions = np.array(eigenfunctions)

# Normalize the eigenfunctions
for i in range(len(eigenfunctions)):
    eigenfunctions[i] /= np.sqrt(np.trapz(np.abs(eigenfunctions[i])**2, xspan))

# Save the eigenfunctions and eigenvalues
A1 = np.abs(eigenfunctions).T  # Transpose to get 5 columns (one for each eigenfunction)
A2 = eigenvalues

# Display or save results
print("Eigenvalues (εn):", A2)
for i in range(n_eigenfunctions):
    plt.plot(xspan, A1[:, i], label=f'ϕ{i+1}(x)')

plt.title("First 5 Eigenfunctions of Quantum Harmonic Oscillator")
plt.xlabel("x")
plt.ylabel("|ϕn(x)|")
plt.legend()
plt.show()
