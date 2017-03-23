import numpy as np
import pandas as pd
def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1/(1+np.exp(-x))
    
def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

learnrate = 0.5

df = pd.read_csv('binary.csv')

x = np.array([1, 2])
y = np.array(0.5)

# Initial weights
w = np.array([0.5, -0.5])

# Calculate one gradient descent step for each weight
# TODO: Calculate output of neural network
# remember that the activation function is f(x)= 1 / ( 1 + e(−z))
# and z is sum of the product of the weight and inputs
nn_output = sigmoid(np.dot(x,w))

# TODO: Calculate error of neural network
# here y is the true value and nn_output is the predicted value
error = y - nn_output

# TODO: Calculate change in weights

# where​​ δj = (y - y_pred(j)) * f'(hj)​ = err_gradient
# f'(hj) is the gradient of the activation function 
# the derivative of the activation function is f′​(h) = f(h) * (1 − f(h))
# When we use the sigmoid_prime
# So err_gradient = error * f'(hj) the derivative of the activation function
err_gradient = error * sigmoid_prime(np.dot(x,w))

# The formula to calculate the new weight => Δw​ij​​ =η * δj * ​​x​i
del_w = learnrate * err_gradient * x

print('Neural Network output:')
print(nn_output)
print('Amount of Error:')
print(error)
print('Change in Weights:')
print(del_w)