#feedforward neural network

import numpy as np
x = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6], [0.6, 0.7, 0.8]])

print("|||||||||||||||||||||||||||||||||||||||||||||||||")
print()

def rlu(a):
    return np.maximum(0,a)

np.random.seed(0)

w = np.random.randn(3,4)*0.1
b = np.full(4, 0)

w1 = np.random.randn(4,2)*0.1
b1 = np.full(2,0)

z = x@w + b
p = rlu(z)

z1 = p@w1 + b1
q = z1

print("Input values of x: ")
print(x)
print("Hidden Layer output after rectifying linear unit: ")
print(p)
print("Final output of the network: ")
print(q)