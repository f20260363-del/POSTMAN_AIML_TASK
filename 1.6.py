import numpy as np
from sklearn.datasets import load_digits
import time

np.random.seed(0)

d = load_digits()
x = d.data/16
l_ = int(input("Enter the digit to be identified using neural networks: "))
l = (d.target == l_).astype(float).reshape(-1, 1)
lab = d.target

y = np.full((len(lab), 10), 0)
y[np.arange(len(lab)), lab] = 1


w = np.random.randn(64, 20)*0.1
b = np.full(20, 0.0) #arrays need to be in form of decimal, in order to perform operations later on without any error
w1 = np.random.randn(20, 1)*0.1
b1 = np.full(1, 0.0) #1 column due to single sigmoid output
pr = {"w": w, "b": b, "w1": w1, "b1": b1}

rate = 0.01
l1 = {"w": 0, "b": 0, "w1": 0, "b1": 0}
l2 = {"w": 0, "b": 0, "w1": 0, "b1": 0}

def linlay(x, w, b):
    return x@w + b

def rlu(a):
    return np.maximum(0,a)

#training loop
for i in range(500):
    z = linlay(x, w, b)
    p = rlu(z)
    z1 = linlay(p, w1, b1)
    a2 = 1/(1+np.exp(-z1))

    dz1 = (a2-l)/len(x)
    dz = (dz1 @ w1.T)
    dz = dz*(z>0)

    wb = {"w": x.T @ dz, "b": dz.sum(axis = 0), "w1": p.T @dz1, "b1": dz1.sum(axis = 0)}

    for j in pr:
        l1[j] = 0.9* l1[j] + 0.1*wb[j]
        l2[j] = 0.999 * l2[j] + 0.001*wb[j]**2
        l1_ = l1[j]/(1 - 0.9**(i+1))
        l2_ = l2[j]/(1 - 0.999 **(i+1))
        pr[j] -= rate * l1_ / (np.sqrt(l2_)+ 10**(-8))

    if (i+1)%20 ==0:
        # how confident was the network in the RIGHT answer, for each picture?
        cp = np.where(l == 1, a2, 1 - a2)
        #   if the real answer was "yes" (l==1)  -> use a2       (confidence it's yes)
        #   if the real answer was "no"  (l==0)  -> use 1 - a2   (confidence it's no)

        ll = -np.mean(np.log(cp))
        print("round: ", (i+1))
        print("loss: ", ll)
        time.sleep(0.25)
