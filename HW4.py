import numpy as np
from scipy.sparse import spdiags

dx = 20/8
dy = 20/8

m = 8 # N value in x and y directions
n = m * m # total size of matrix
e0 = np.zeros((n, 1)) # vector of zeros
e1 = np.ones((n, 1)) # vector of ones
e2 = np.copy(e1) # copy the one vector
e4 = np.copy(e0) # copy the zero vector
for j in range(1, m+1):
    e2[m*j-1] = 0 # overwrite every m^th value with zero
    e4[m*j-1] = 1 # overwirte every m^th value with one
# Shift to correct positions
e3 = np.zeros_like(e2)
e3[1:n] = e2[0:n-1]
e3[0] = e2[n-1]
e5 = np.zeros_like(e4)
e5[1:n] = e4[0:n-1]
e5[0] = e4[n-1]
# Place diagonal elements
diagonals = [e1.flatten(), e1.flatten(), e5.flatten(),
e2.flatten(), -4 * e1.flatten(), e3.flatten(),
e4.flatten(), e1.flatten(), e1.flatten()]
offsets = [-(n-m), -m, -m+1, -1, 0, 1, m-1, m, (n-m)]
A1 = (1/(dx**2))*spdiags(diagonals, offsets, n, n).toarray()

# MATRIX B
e1 = np.ones((n, 1))
e2 = np.ones((n, 1))
e3 = np.ones((n,1))
e4 = np.ones((n,1))

diagonals = [e1.flatten(), -e2.flatten(), e3.flatten(), -e4.flatten()]
offsets = [(-n+m), -m, m, n-m]
B = spdiags(diagonals, offsets, n, n,).toarray()
A2 = B/(2*dx)

# MATRIX C
e1 = np.zeros((n, 1))
e2 = np.ones((n, 1))
e3 = np.ones((n,1))
e4 = np.zeros((n,1))
for index in range(8):
    e1[8*index] = 1
    e2[8*index + 7] = 0
    e3[8*index] = 0
    e4[8*index+7] = 1

diagonals = [e1.flatten(), -e2.flatten(), e3.flatten(), -e4.flatten()]
offsets = [(-m+1), -1, 1, m-1]
C = spdiags(diagonals, offsets, n, n,).toarray()
A3 = C/(2*dy)

# Output the matrices
print("Matrix A1 (Second Derivative):")
print(A1)
print("\nMatrix A2 (First Derivative d/dx):")
print(A2)
print("\nMatrix A3 (First Derivative d/dy):")
print(A3)
