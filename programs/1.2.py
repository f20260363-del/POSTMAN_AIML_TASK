#implementation of forward pass using linear layers and activation function
import numpy as np

#linear layering, done in the previous item as well, but compressed into a function due to repetitive use
def linlay(x, w, b):
    return x @ w + b

#Rectified linear unit function, same as that from the previous unit
def rlu(a):
    return np.maximum(0,a)

#input array
x = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6]])

#to keep the random values same in all runs of the program
np.random.seed(0)

#layer 1 weight and bias
w = np.random.randn(3, 4)*0.1
b = np.full(4, 0)

#layer 2 weight and bias
w1 = np.random.randn(4, 3)*0.1
b1 = np.full(3, 0)

#linear layering of first layer
z = linlay(x, w, b)
p = rlu(z)

#linear layering of second layer
z1 = linlay(p, w1, b1)
q = z1

#output
print("Input x: ")
print(x)
print("Hidden layer output after rectifying linear unit: ")
print(p)
print("Final output of the network: ")
print(q)
