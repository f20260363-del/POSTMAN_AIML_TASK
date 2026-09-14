import numpy as np

np.random.seed(0)

x = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]])
y = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])

w = np.random.randn(3, 2)*0.1
b = np.full(2, 0)

w1 = np.random.randn(2, 3)*0.1
b1 = np.full(3, 0)

def linlay(x, w, b):
    return x @ w + b


def rlu(a):
    return np.maximum(0,a)

def fw(x, w, b, w1, b1):
    z = linlay(x, w, b)
    p = rlu(z)
    q = linlay(p, w1, b1)
    return q, z, p

def l(x, y, w, b, w1, b1):
    q, z, p = fw(x, w, b, w1, b1)
    return np.mean((q-y)**2)

q, z, p = fw(x, w, b, w1, b1)
n = q.size

dq = (2.0/n)*(q-y)
dw1 = p.T @ dq
dp = dq @ w1.T
dz = dp * (z>0)
dw_anal = x.T @ dz

h = 1e-5

w[0][0] += h
lp = l(x, y, w, b, w1, b1)

w[0][0] -= 2*h

lm = l(x, y, w, b, w1, b1)

w[0][0]+=h

dw_num = (lp - lm)/(2*h)
x = dw_anal[0][0]
y = dw_num

print("Analytic gradient: ", x)
print("Numerical gradient: ", y)
print("Difference: ", abs(x-y))
