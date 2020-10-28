import numpy as np
import matplotlib.pyplot as plt
import time as time
import scipy.linalg
import scipy.optimize

# Problem 1
N = 2000;
offDiag = np.ones(N-1)
A = -1*np.diag(np.ones(N)) + 4*np.diag(offDiag, 1) + 4*np.diag(offDiag, -1)

# Problem 1 - b
residuals = 0
start = time.time()
for k in range(100):
    b = np.random.rand(N, 1)
    x = np.linalg.solve(A, b)
    residuals = residuals + np.linalg.norm(A @ x - b)
stop = time.time() - start
print(stop)
print(residuals)

# Problem 1 - c
P, L, U = scipy.linalg.lu(A)
residuals = 0
start = time.time()
for k in range(100):
    b = np.random.rand(N, 1)
    y = scipy.linalg.solve(L, P @ b)
    x = scipy.linalg.solve(U, y)
    residuals = residuals + np.linalg.norm(A @ x - b)
stop = time.time() - start
print(stop)
print(residuals)

# Problem 1 - d
start = time.time()
residuals = 0
A1 = np.linalg.inv(A)
for k in range(100):
    b = np.random.rand(N, 1)
    x = A1 @ b
    residuals = residuals + np.linalg.norm(A @ x - b)
stop = time.time() - start
print(stop)
print(residuals)
