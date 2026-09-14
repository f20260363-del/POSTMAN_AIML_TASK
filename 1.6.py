#stretch goal
#activation loss combination and derviation of gradients, through the implementation of adam
import numpy as np
from sklearn.datasets import load_digits
import time

np.random.seed(0)

#loading dataset from scikit-learn
d = load_digits()
x = d.data/16
l_ = int(input("Enter the digit to be identified using neural networks: ")) #inputting of digit directly from user that needs to be identified
l = (d.target == l_).astype(float).reshape(-1, 1)

#layer 1
w = np.random.randn(64, 20)*0.1
b = np.full(20, 0.0) #arrays need to be in form of decimal, in order to perform operations later on without any error

#layer 2
w1 = np.random.randn(20, 1)*0.1
b1 = np.full(1, 0.0) #1 column due to single sigmoid output

#parameters dictionary (weights and biases of both layers)
pr = {"w": w, "b": b, "w1": w1, "b1": b1}

#learning rate
rate = 0.01
l1 = {"w": 0, "b": 0, "w1": 0, "b1": 0} #for gradients
l2 = {"w": 0, "b": 0, "w1": 0, "b1": 0} #for squared gradients

#linear layering
def linlay(x, w, b):
    return x@w + b

#rectified linear unit
def rlu(a):
    return np.maximum(0,a)

#training loop
for i in range(500):
    #forward passing
    z = linlay(x, w, b)
    p = rlu(z)
    z1 = linlay(p, w1, b1)
    a2 = 1/(1+np.exp(-z1))

    #backward pass (gradient derivation)
    dz1 = (a2-l)/len(x)
    dz = (dz1 @ w1.T)
    dz = dz*(z>0)

    #mapping parameters to analytical gradients
    wb = {"w": x.T @ dz, "b": dz.sum(axis = 0), "w1": p.T @dz1, "b1": dz1.sum(axis = 0)}

    for j in pr:
        #bias updation: this is done using the mathematical conept of exponentially weighted averages
        #1st moment bias update
        #dividing by (1-x^(i+1)) scales the process of moving the predictions away from the intial value of zero
        l1[j] = 0.9* l1[j] + 0.1*wb[j]
        #2nd moment bias update
        l2[j] = 0.999 * l2[j] + 0.001*wb[j]**2
        #bias corrected 1st moment
        l1_ = l1[j]/(1 - 0.9**(i+1))
        #bias correct 2nd moment
        l2_ = l2[j]/(1 - 0.999 **(i+1))
        pr[j] -= rate * l1_ / (np.sqrt(l2_)+ 10**(-8))

    if (i+1)%20 ==0:
        #probability extraction
        cp = np.where(l == 1, a2, 1 - a2)
        #if truth target l == 1, then cp = a2, else it is 1-a2 (converse probability)
        #utilisation of ternary operator, if-else takes up quite a lot of lines

        #binary cross entropy (BCE)
        #harsh grading system type model - prediciton of answer + sureness of it beign correct, both account in this model
        ll = -np.mean(np.log(cp))
        print("round: ", (i+1))
        print("loss: ", ll)
        time.sleep(0.25)
