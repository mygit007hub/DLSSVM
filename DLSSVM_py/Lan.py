# Lagrange Multipler

import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import fsolve

def lagrangian_func(X):
    x = X[0]
    y = X[1]
    L = X[2] # this is the multiplier. lambda is a reserved keyword in python
    return x + y + L * (x**2 + y**2 - 1)

def derivative_func(X):
    dLambda = np.zeros(len(X))
    h = 1e-3 # this is the step size used in the finite difference.
    for i in range(len(X)):
        dX = np.zeros(len(X))
        dX[i] = h
        dLambda[i] = (lagrangian_func(X+dX)-lagrangian_func(X-dX))/(2*h)
    return dLambda

# this is the max
X1 = fsolve(derivative_func, [1, 1, 0])
print X1, lagrangian_func(X1)
plt.plot(X1,[1,1,0])
plt.show()
# this is the min
X2 = fsolve(derivative_func, [-1, -2, 0])
print X2, lagrangian_func(X2)