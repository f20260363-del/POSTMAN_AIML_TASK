#checkign gradients against numerical gradient checking
import numpy as np

#to keep random values same in every run
np.random.seed(0)

#input array and truth target (actual values needed to be predicted by network)
x = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]])
y = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])

#layer 1 weight and bias
w = np.random.randn(3, 2)*0.1
b = np.full(2, 0)

#layer 2 weight and bias
w1 = np.random.randn(2, 3)*0.1
b1 = np.full(3, 0)

#linear layering functoin
def linlay(x, w, b):
    return x @ w + b

#rectified linear unit function
def rlu(a):
    return np.maximum(0,a)

#forward passing of neural network (done in previous items as well, compressed into a function due to repetitive use)
def fw(x, w, b, w1, b1):
    z = linlay(x, w, b)
    p = rlu(z)
    q = linlay(p, w1, b1)
    return q, z, p

#loss computing function (done in item 1.3, used as a function over here)
def l(x, y, w, b, w1, b1):
    q, z, p = fw(x, w, b, w1, b1)
    return np.mean((q-y)**2)

q, z, p = fw(x, w, b, w1, b1)
n = q.size #no of scalar values in output matrix

#1. Derivative of MSE loss wrt q (same as in item 1.3)
dq = (2.0/n)*(q-y)

#2. Layer 2 weiht gradient
dw1 = p.T @ dq

#3. Gradient backpropagation
dp = dq @ w1.T

#4. Backpropagation with RLU
dz = dp * (z>0)

#Analytic gradient of layer 1 weight gradient
dw_anal = x.T @ dz

#Step size (10^-5, small value chosen)
h = 1e-5

#Double Checking Model (Numerical Gradient Calculation)
#w[0][0] slightly pushed up, then loss score measured
w[0][0] += h
lp = l(x, y, w, b, w1, b1)

#pushed down by the same value of h, then loss core is measured again
w[0][0] -= 2*h
lm = l(x, y, w, b, w1, b1)

#retained to original value
w[0][0]+=h

#numerical gradient calculation
dw_num = (lp - lm)/(2*h)
x = dw_anal[0][0]
y = dw_num

#output
print("Analytic gradient: ", x)
print("Numerical gradient: ", y)
print("Difference: ", abs(x-y))
