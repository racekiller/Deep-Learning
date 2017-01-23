
# coding: utf-8

# In[6]:

from numpy import *


# In[7]:

# y = mx * b
# m is the slope and b is y-intercept


# In[44]:

def compute_error_for_line_given_points(b,m,points):
    for i in range(0, len(points)):
        x = points[i,0]
        y = points[i,1]
        totalError += (y - (m * x + b) ** 2)
    return totalError / float(len(points))


# In[18]:

import matplotlib.pyplot as plt


## In[19]:
#
#get_ipython().magic('matplotlib inline')
#
#
## In[34]:
#
#x = points[:,0]
#y = points[:,1]
#
#
## In[36]:
#
#plt.plot(m*x+b,y)
#plt.scatter(x,y)


# In[69]:

def step_gradient(b_current, m_current, points, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0,len(points)):
        x = points[i, 0]
        y = points[i, 1]
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]


# In[63]:

def gradient_descent_runner(points, starting_b, starting_m, learningRate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learningRate)
    return [b, m]

# In[67]:

def run():
    points = genfromtxt("data.csv", delimiter=",")
    learningRate = 0.0001
    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    num_iterations = 1000
    print ("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
    print ("Running...")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learningRate, num_iterations)
    print ("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))


# In[70]:

if __name__ == '__main__':
    run()

