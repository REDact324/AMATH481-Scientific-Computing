import numpy as np

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
A2 = 0
# Number of iterations of Newton and bisection methods
A3 = [0, 0]

# Newton-Raphson method
count = 0
error = np.abs(0 - function(A1[-1]))

while error > 1e-6:
  x_ini = A1[-1]
  x_new = x_ini - (function(x_ini) / diff_function(x_ini))
  A1.append(x_new)
  error = np.abs(0 - function(x_new))
  count += 1

A3[0] = count

# Bisection method
count = 1
x_l = -0.7
x_r = -0.4
x_c = (x_l + x_r) / 2
error = np.abs(function(x_c))

while error > 1e-6:
  if function(x_c) > 0:
    x_l = x_c
  else:
    x_r = x_c

  x_c = (x_l + x_r) / 2
  error = np.abs(function(x_c))
  count += 1

A2 = x_c
A3[1] = count

print(A1, A2, A3)
