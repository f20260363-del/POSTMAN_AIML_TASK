#implementation of manual backward pass using chain rule and gradient derivation for every layer
import numpy as np

#linear layering function
def linlay(x, w, b):
    return x @ w + b

#rectifying linear unit function
def rlu(a):
    return np.maximum(0,a)

#gradient of rlu function, w.r.t. to its input
#it returns a boolean array, acting as a mask, for checking if values are greater than 0
def rlugrad(a):
    return (a>0)

#input array
x = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]])

#truth target: actual correct answers, which should be predicted by neural network
y = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])

#to keep random values same in every run
np.random.seed(0)

#layer 1 weight and bias
w = np.random.randn(3, 5)*0.1
b = np.full(5, 0)

#layer 2 weight and bias
w1 = np.random.randn(5, 2)*0.1
b1 = np.full(2, 0)

#linear layering of first layer
z = linlay(x, w, b)
p = rlu(z)

#linear layering of second layer
z1 = linlay(p, w1, b1)
q = z1

n  = q.size #to obtain the total scalar values in output matrix
#calculation of mean squared error
l = np.mean((q-y)**2)

#1. Derivation of mse loss w.r.t. output q
dq = (2/n)*(q-y)
dz1 = dq

#2. Application of chain rule and matrix calculus
#p is the hidden layer, and dw1 is the output layer with guessed values
dw1 = p.T @ dz1
db1 = dz1.sum(axis = 0)
dp = dz1 @ w1.T

#3. Backpropagating through layer 1, using RLU gradient
dz = dp *rlugrad(z)

#4. Computing the weight and bias gradients of first layer
dw = x.T @ dz
db = dz.sum(axis = 0)

#output of forward pass and mse loss
print("Forward pass output: ")
print(q)
print("Mean Squared Error Loss: ", l)

#gradient of loss w.r.t weights and biases of the two layers
print("dL/dw  (shape", dw.shape, "):")
print(dw)      
print("dL/db  (shape", db.shape, "):")
print(db)
print("dL/dw1 (shape", dw1.shape, "):")
print(dw1)
print("dL/db1 (shape", db1.shape, "):")
print(db1)
