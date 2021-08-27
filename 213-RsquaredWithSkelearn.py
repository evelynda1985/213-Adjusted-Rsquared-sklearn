#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbs
sbs.set()

from sklearn.linear_model import LinearRegression


# In[9]:


data = pd.read_csv('1.02. Multiple linear regression.csv')
data.head()


# In[10]:


data.describe()


# In[11]:


x = data[['SAT', 'Rand 1,2,3']]
y = data['GPA']


# In[12]:


reg = LinearRegression()


# In[13]:


reg.fit(x,y)


# In[14]:


reg.coef_


# In[15]:


reg.intercept_


# In[16]:


# R-squared
reg.score(x,y)


# ### Formula for Adjusted R-squared
# $R^2_{adj.} = 1 - (1-R^2)*\frac{n-1}{n-p-1}$

# In[18]:


x.shape


# In[19]:


r2 = reg.score(x,y)
n = x.shape[0]
p = x.shape[1]
adjusted_r2 = 1 - (1-r2) * (n-1)/(n-p-1)
adjusted_r2


# In[ ]:




