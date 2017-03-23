
# coding: utf-8

# In[11]:

# TODO: Add import statements

# Assign the dataframe to this variable.
# TODO: Load the data
bmi_life_data = None 

# Make and fit the linear regression model
#TODO: Fit the model and Assign it to bmi_life_model
bmi_life_model = None

# Mak a prediction using the model
# TODO: Predict life expectancy for a BMI value of 21.07931
laos_life_exp = None

# >>> from sklearn.linear_model import LinearRegression
# >>> model = LinearRegression()
# >>> model.fit(x_values, y_values)


# In[2]:

import pandas as pd
import numpy as np


# In[42]:

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[12]:

bmi_life_data = pd.read_csv('bmi_and_life_expectancy.csv')


# In[10]:

from sklearn.linear_model import LinearRegression


# In[53]:

plt.scatter(bmi_life_data['BMI'],bmi_life_data['Life expectancy'])
plt.plot(bmi_life_data['BMI'], bmi_life_model.predict(bmi_life_data[['BMI']]), color = 'red')
plt.xlabel = 'BMI'
plt.ylabel = 'Life Expectancy'
plt.title('BMI vs Live Expectancy')
plt.show()


# In[31]:

bmi_life_model = LinearRegression()


# In[32]:

bmi_life_model.fit(bmi_life_data[['BMI']], bmi_life_data[['Life expectancy']])


# In[55]:

laos_life_exp = bmi_life_model.predict(21.07931)


# In[56]:

laos_life_exp


# In[59]:

# The coefficients
print('Coefficients: \n', bmi_life_model.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((bmi_life_model.predict(bmi_life_data[['BMI']]) - bmi_life_data[['Life expectancy']]) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % bmi_life_model.score(bmi_life_data[['BMI']], bmi_life_data[['Life expectancy']]))

