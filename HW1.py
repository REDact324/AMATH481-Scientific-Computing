import numpy as np

######### I ##########

def function(x):
  result = x * np.sin(3 * x)
  result = result - np.exp(x)

  return result

def diff_function(x):
  result = np.sin(3 * x) + 3 * x * np.cos(3 * x)
  result = result - np.exp(x)

  return result

# x values in Newton method
A1 = [-1.6]
# Mid-point value in bisection method
A2 = []
# Number of iterations of Newton and bisection methods
A3 = [0, 0]

# Newton-Raphson method
count = 0

while count < 1000:
  x_ini = A1[-1]
  x_new = x_ini - (function(x_ini) / diff_function(x_ini))
  A1.append(x_new)
  count += 1
  if np.abs(x_new - x_ini) <= 1e-6:
    break


A3[0] = count

# Bisection method
count = 1
x_l = -0.7
x_r = -0.4
x_c = (x_l + x_r) / 2
A2.append(x_c)
error = np.abs(function(x_c))

while error > 1e-6:
  if function(x_c) > 0:
    x_l = x_c
  else:
    x_r = x_c

  x_c = (x_l + x_r) / 2
  A2.append(x_c)
  error = np.abs(function(x_c))
  count += 1

A3[1] = count


######### II ##########


A = np.array([[1, 2],[-1, 1]])
B = np.array([[2, 0], [0, 2]])
C = np.array([[2, 0, -3], [0, 0, -1]])
D = np.array([[1, 2], [2, 3], [-1, 0]])
x = np.array([[1], [0]])
y = np.array([[0], [1]])
z = np.array([[1], [2], [-1]])

A4 = np.ndarray.tolist(A + B)
A5 = np.ndarray.tolist((3 * x - 4 * y).reshape(-1))
A6 = np.ndarray.tolist((np.matmul(A, x)).reshape(-1))
A7 = np.ndarray.tolist((np.matmul(B, (x - y))).reshape(-1))
A8 = np.ndarray.tolist((np.matmul(D, x)).reshape(-1))
A9 = np.ndarray.tolist((np.matmul(D, y) + z).reshape(-1))
A10 = np.ndarray.tolist(np.matmul(A, B))
A11 = np.ndarray.tolist(np.matmul(B, C))
A12 = np.ndarray.tolist(np.matmul(C, D))
