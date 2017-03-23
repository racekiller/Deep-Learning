
# coding: utf-8

# In[1]:

from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

# Load the data from the the boston house-prices dataset 
boston_data = load_boston()
x = boston_data['data']
y = boston_data['target']

# Make and fit the linear regression model
# TODO: Fit the model and Assign it to the model variable
model = None

# Make a prediction using the model
sample_house = [[2.29690000e-01, 0.00000000e+00, 1.05900000e+01, 0.00000000e+00, 4.89000000e-01,
                6.32600000e+00, 5.25000000e+01, 4.35490000e+00, 4.00000000e+00, 2.77000000e+02,
                1.86000000e+01, 3.94870000e+02, 1.09700000e+01]]
# TODO: Predict housing price for the sample_house
prediction = None


# In[2]:

import pandas as pd
import numpy as np


# In[3]:

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[7]:

x


# In[4]:

model = LinearRegression()


# In[8]:

model.fit(x, y)


# In[17]:

sample_house


# In[10]:

laos_life_exp = model.predict(sample_house)


# In[11]:

laos_life_exp


# In[59]:

# The coefficients
print('Coefficients: \n', bmi_life_model.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((bmi_life_model.predict(bmi_life_data[['BMI']]) - bmi_life_data[['Life expectancy']]) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % bmi_life_model.score(bmi_life_data[['BMI']], bmi_life_data[['Life expectancy']]))


# In[ ]:

plt.scatter(x,y)
plt.plot(bmi_life_data['BMI'], bmi_life_model.predict(bmi_life_data[['BMI']]), color = 'red')
plt.xlabel = 'BMI'
plt.ylabel = 'Life Expectancy'
plt.title('BMI vs Live Expectancy')
plt.show()

