#feedforward neural network

import numpy as np

#creation of array with 5 samples (rows) and 3 features (the specific data in each row)
x = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6], [0.6, 0.7, 0.8]])

#rectified linear unit: creates a separate layer with all negative values changed to 0
#prevents the collapsing of two matrix multiplications into a single linear transformation
#i did not completely comprehend how it collapses, but i do understand why the negative values need to be changed to zero
#if the value is negative, it means the neuron found no evidence for the pattern
def rlu(a):
    return np.maximum(0,a)

#same random numbers every time you run the program
np.random.seed(0)

#Layer 1
#w = weight matrix connects 3 features to 5 hidden units
#0.1 is multiplied to keep the input units small, too big causes numbers to explode through each layer
#too small results in excessive shrinking through the network
w = np.random.randn(3,5)*0.1
b = np.full(5, 0) #bias, a (5,0) zero matrix for the 5 hidden units, lets neurons shift output independent of input

#Layer 2
w1 = np.random.randn(5,2)*0.1 #weights with the 5 hidden neurons from previous layer
b1 = np.full(2,0) #bias for the second time
#though it is zero right now, it is the fine tuning knob for the network, which is important for training later on

#First Linar Layer: Matrix Multiplication with weight matrix, then add bias
z = x@w + b 
p = rlu(z) #rlu application for the purpose mentioned above

#Second Layer: First matrix multiplied with second weigh matrix, then add bias
z1 = p@w1 + b1
q = z1

#output
print("Input values of x: ")
print(x)
print("Hidden Layer output after rectifying linear unit: ")
print(p)
print("Final output of the network: ")
print(q)
