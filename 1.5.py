import numpy as np
import time
import matplotlib

from sklearn.datasets import load_digits

np.random.seed(0) #keeps the random numbers the same

d = load_digits()
x = d.data/16.0
l = d.target

y = np.full((len(l), 10), 0)
y[np.arange(len(l)), l] = 1

w = np.random.randn(64, 20)*0.1
b = np.full(20, 0.0) #arrays need to be in form of decimal, in order to perform operations later on without any error
w1 = np.random.randn(20, 10)*0.1
b1 = np.full(10, 0.0)

rate = 0.5

def linlay(x, w, b):
    return x@w + b

def rlu(a):
    return np.maximum(0,a)

for i in range(500):

    #guessing the forward pass
    z = linlay(x, w, b)
    p = rlu(z)
    a1 = np.exp(linlay(p, w1, b1))
    a2 = a1/a1.sum(axis = 1, keepdims= True)

    #checking for loss
    if (i+1) %10 == 0:
        cp = np.sum(y*a2, axis = 1)
        l = -np.mean(np.log(cp))
        print("round: ", (i+1))
        print("loss: ", l)
        time.sleep(0.25)

    #backwards working

    dz1 = (a2-y)/len(x)
    dw1 = p.T @ dz1
    db1 = dz1.sum(axis = 0)

    masker = z>0
    dz = (dz1@w1.T)*(z>0)

    dw = x.T@dz
    db = dz.sum(axis = 0)

    #updation
    w-=(rate*dw)
    b-=(rate*db)
    w1-=(rate*dw1)
    b1-=(rate*db1)