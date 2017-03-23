
# coding: utf-8

# In[3]:

import numpy as np


# In[4]:

def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))


# In[5]:

x = np.array([0.5, 0.1, -0.2])
target = 0.6
learnrate = 0.5


# In[42]:

x


# In[41]:

x[:,None]


# In[6]:

weights_input_hidden = np.array([[0.5, -0.6],
                                 [0.1, -0.2],
                                 [0.1, 0.7]])


# In[36]:

weights_hidden_output = np.array([0.1, -0.3])
weights_hidden_output


# In[15]:

## Forward pass
hidden_layer_input = np.dot(x, weights_input_hidden)
hidden_layer_input


# In[34]:

# This is the f(h)
hidden_layer_output = sigmoid(hidden_layer_input)
hidden_layer_output


# In[17]:

output_layer_in = np.dot(hidden_layer_output, weights_hidden_output)
output_layer_in


# In[19]:

output = sigmoid(output_layer_in)
output


# In[26]:

## Backwards pass
## TODO: Calculate error
error = None
error = target - output
error


# In[25]:

# TODO: Calculate error gradient for output layer
del_err_output = None
del_err_output = error * (output * (1 - output))
del_err_output


# In[35]:

# TODO: Calculate error gradient for hidden layer
del_err_hidden = None
del_err_hidden = weights_hidden_output * del_err_output * hidden_layer_output * (1 - hidden_layer_output)
del_err_hidden


# In[37]:

# TODO: Calculate change in weights for hidden layer to output layer
delta_w_h_o = None
delta_w_h_o = learnrate * del_err_output * hidden_layer_output
delta_w_h_o


# In[51]:

# TODO: Calculate change in weights for input layer to hidden layer
delta_w_i_h = None
delta_w_i_h = learnrate * del_err_hidden * x[:,None]
delta_w_i_h


# In[52]:

print('Change in weights for hidden layer to output layer:')
print(delta_w_h_o)
print('Change in weights for input layer to hidden layer:')
print(delta_w_i_h)


# In[ ]:

