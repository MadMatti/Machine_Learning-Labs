from pdb import line_prefix
import numpy as np
from scipy.optimize import minimize
from matplotlib import pyplot as plt
import random
import math
import matplotlib.patches as mpatches


# Generate test data

'''Generate two data class with 2D Gaussian distribution'''
#np.random.seed(100)    # to get the same random numbers

classA = np.concatenate(
    (np.random.randn(10, 2) * 0.5 + [1.5, 0.5],
    np.random.randn(10, 2) * 0.5 + [-1.5, 0.5]))
classB = np.random.randn(20, 2) * 0.5 + [0.0, -0.5]

inputs = np.concatenate((classA, classB))
targets = np.concatenate((np.ones(classA.shape[0]), -np.ones(classB.shape[0])))

N = inputs.shape[0]   # Number of rows (samples)

permute = list(range(N))
random.shuffle(permute)
inputs = inputs[permute, :]
targets = targets[permute]


# Define kernel functions

def LinearKernel(x, y):
    return np.dot(x, y)

def PolynomialKernel(x, y, p=2):    # max value for p = 10
    '''Optional parameter p controls the degree of the polynomial'''
    return (1 + np.dot(x, y)) ** p

def RBFKernel(x, y, sigma=1.0):     # max value for sigma = 10
    '''Optional parameter sigma controls the width of the Gaussian'''
    return math.exp(-np.linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

Kernel = RBFKernel


# Generate P matrix

Pmatrix = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        Pmatrix[i, j] = targets[i] * targets[j] * Kernel(inputs[i], inputs[j])


# Define objective function

def objective(alpha):
    '''Implement expression (4), i.e., minimize function'''
    return 0.5 * np.dot(alpha, np.dot(alpha, Pmatrix)) - np.sum(alpha)


# Define zerfun function

def zerfun(alpha):
    '''Implement expression (10), i.e., the equality constraint'''
    return np.dot(alpha, targets)


# Call to minimize function

'''Initial guess of alpha'''
start = np.zeros(N)

'''Set B, i.e., the bounds for alpha vector'''

#B = [(0, None) for b in range(N)]  for having only lower bound

C = 10
B = [(0, C) for b in range(N)]  # for having both lower and upper bounds

'''Set the constraint, in this case the zerofun function. XC is given as a dictionary'''
XC = {'type':'eq', 'fun':zerfun}

'''Call to minimize function'''
ret = minimize(objective, start, bounds=B, constraints=XC)

if (not ret['success']):    # success is a Boolean flag indicating if the optimizer exited successfully
    raise ValueError('Cannot find optimal solution')

alpha = ret['x']    # x is the solution array


# extract non-zero alphas

nonzero = [(alpha[i], inputs[i], targets[i]) for i in range(N) if abs(alpha[i]) > 1e-5]

def bvalue():
    '''Implement expression (7)'''
    sum = 0
    for value in nonzero:
        sum += value[0] * value[2] * Kernel(value[1], nonzero[0][1])
    return sum - nonzero[0][2]


# Implement indicator function

def indicator(x, y):
    '''Implement expression (6)'''
    sum = 0
    for value in nonzero:
        sum += value[0] * value[2] * Kernel(value[1], [x, y])
    return sum - bvalue()


# Plot data points

plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')
plt.plot([p[1][0] for p in nonzero], [p[1][1] for p in nonzero], 'g+')
plt.axis('equal')   # set the axes to the same scale

# Plot the decision boundary

xgrid = np.linspace(-5, 5)
ygrid = np.linspace(-4, 4)
grid = np.array([[indicator(x, y) for x in xgrid] for y in ygrid])
plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))

blue_patch = mpatches.Patch(color='blue', label='ClassA')
red_patch = mpatches.Patch(color='red', label='ClassB')
black_patch = mpatches.Patch(color='black', label='Decision Boundary')
plt.legend(handles=[blue_patch, red_patch, black_patch])

plt.xlabel('X')
plt.ylabel('Y')

plt.savefig('SVM/resources/svmplot.png')  # save a copy in a file
plt.show()