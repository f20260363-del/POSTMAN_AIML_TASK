import numpy as np

def linlay(x, w, b):
    return x @ w + b

def rlu(a):
    return np.maximum(0,a)

def rlugrad(a):
    return (a>0)#.astype(float)

x = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]])

y = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])

np.random.seed(0)
w = np.random.randn(3, 5)*0.1
b = np.full(5, 0)

w1 = np.random.randn(5, 2)*0.1
b1 = np.full(2, 0)

z = linlay(x, w, b)
p = rlu(z)

z1 = linlay(p, w1, b1)
q = z1

n  = q.size
l = np.mean((q-y)**2)

dq = (2/n)*(q-y)
dz1 = dq

dw1 = p.T @ dz1
db1 = dz1.sum(axis = 0)
dp = dz1 @ w1.T

dz = dp *rlugrad(z)

dw = x.T @ dz
db = dz.sum(axis = 0)

print("Forward pass output: ")
print(q)
print("Mean Squared Error Loss: ", l)

print("dL/dw  (shape", dw.shape, "):\n", dw)
print("dL/db  (shape", db.shape, "):\n", db)
print("dL/dw1 (shape", dw1.shape, "):\n", dw1)
print("dL/db1 (shape", db1.shape, "):\n", db1)