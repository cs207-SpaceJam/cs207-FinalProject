import spacejam as sj
import pytest
import numpy as np

# relationship beteen time and step (N*h = T)
# h: timestep
# N: total number of steps
# T: total time

# Initial conditions
h = 0.01
N = 500
T = h*N
X = np.zeros(N)
X[0] = 1
t = np.arange(0, T, h)

# toy function
def f(x):
    f1 = -x
    return np.array([f1])

# analytic solution
def analytic(t):
    x = np.exp(-t)
    return x
X_test = analytic(t)

# run simulation
for n in range(N-1):
    X[n+1] = sj.integrators.amso(f, np.array([X[n]]), h=h, X_tol=1E-4)

tol = 0.0032
diff = np.abs(X[-1] - X_test[-1])
assert diff < tol
