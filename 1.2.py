import numpy as np

def linlay(x, w, b):
    return x @ w + b

def rlu(a):
    return np.maximum(0,a)

x = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6]])

np.random.seed(0)

w = np.random.randn(3, 4)*0.1
b = np.full(4, 0)

w1 = np.random.randn(4, 3)*0.1
b1 = np.full(3, 0)

z = linlay(x, w, b)
p = rlu(z)

z1 = linlay(p, w1, b1)
q = z1

print("Input x: ")
print(x)
print("Hidden layer output after rectifying linear unit: ")
print(p)
print("Final output of the network: ")
print(q)