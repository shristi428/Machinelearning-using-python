#!/usr/bin/env python
# coding: utf-8

# In[4]:


import matplotlib.pyplot  as plt
import pandas as pd
import numpy as np
import pylab as pl
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


df=pd.read_csv("FuelConsumptionML01.csv")
df.head()


# In[6]:


#summerize the data
df.describe()


# In[9]:


cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)


# In[11]:


viz = cdf[['ENGINESIZE' , 'CYLINDERS' , 'FUELCONSUMPTION_COMB','CO2EMISSIONS']]
viz.hist()
plt.show()


# In[14]:


plt.scatter(cdf.FUELCONSUMPTION_COMB,cdf.CO2EMISSIONS,color='red')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("CO2EMISSIONS")
plt.show()


# In[18]:


plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,color='purple')
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2EMISSIONS")
plt.show()


# In[19]:


plt.scatter(cdf.CYLINDERS,cdf.CO2EMISSIONS,color='yellow')
plt.xlabel("CYLINDERS")
plt.ylabel("CO2EMISSION")
plt.show()


# In[26]:


from sklearn import linear_model
msk=np.random.rand(len(df))<0.8
train=cdf[msk]
test=cdf[~msk]
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (train_x, train_y)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)


# In[31]:


plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS,color='blue')
plt.plot(train_x ,regr.coef_[0][0]*train_x + regr.intercept_[0],'-r')
plt.xlabel("ENGINESIZE")
plt.ylabel("EMISSIONS")


# In[34]:


from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_hat = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )


# In[ ]:




