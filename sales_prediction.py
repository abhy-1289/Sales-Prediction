#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv('Downloads/perrin-freres-monthly-champagne-.csv')


# In[3]:


df.head()


# In[4]:


df.columns=['Month','Sales']
df.head()


# In[5]:


df.isnull().sum()


# In[6]:


df.tail()


# In[7]:


df.drop(106,axis=0,inplace=True)


# In[8]:


df.drop(105,axis=0,inplace=True)


# In[9]:


df.isnull().sum()


# In[10]:


df.dtypes


# In[11]:


df['Month']=pd.to_datetime(df['Month'])


# In[12]:


df.dtypes


# In[13]:


df.set_index('Month',inplace=True)


# In[14]:


df.head()


# In[15]:


df.describe()


# In[16]:


df.plot()


# In[17]:


import statsmodels
from statsmodels.tsa.stattools import adfuller


# In[18]:


def adfuller_test(sales):
    result=adfuller(sales)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")


# In[19]:


adfuller_test(df['Sales'])


# In[20]:


df['sales_first_diff']=df['Sales']-df['Sales'].shift(1)


# In[21]:


df.head()


# In[22]:


df['sesonal_first_diff']=df['Sales']-df['Sales'].shift(12)


# In[23]:


df.head()


# In[24]:


adfuller_test(df['sesonal_first_diff'].dropna())


# In[25]:


df['sesonal_first_diff'].plot()


# In[26]:


from statsmodels.graphics.tsaplots import plot_acf,plot_pacf


# In[27]:


df.head(15)


# In[28]:


import matplotlib.pyplot as plt


# In[31]:


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = plot_acf(df['sesonal_first_diff'].iloc[13:],lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = plot_pacf(df['sesonal_first_diff'].iloc[13:],lags=40,ax=ax2)


# In[32]:


from statsmodels.tsa.arima_model import ARIMA


# In[33]:


model=ARIMA(df['Sales'],order=(1,1,1))


# In[34]:


model=model.fit()


# In[35]:


model.summary()


# In[36]:


df['forecast']=model.predict(start=90,end=103,dynamic=True)


# In[37]:


df.tail(20)


# In[38]:


df[['Sales','forecast']].plot()


# In[39]:


import statsmodels.api as sm


# In[40]:


model1=sm.tsa.statespace.SARIMAX(df['Sales'],order=(1,1,1),seasonal_order=(1,1,1,12))


# In[41]:


model1=model1.fit()


# In[42]:


df['forecast_sarimax']=model1.predict(start=90,end=103,dynamic=True)


# In[43]:


df.tail(20)


# In[45]:


df[['Sales','forecast_sarimax']].plot(figsize=(12,10))


# In[46]:


from pandas.tseries.offsets import DateOffset


# In[48]:


future_dates=[df.index[-1] + DateOffset(months=x)for x in range(0,24)]
future_dates


# In[49]:


future_dataset=pd.DataFrame(index=future_dates[1:],columns=df.columns)


# In[50]:


future_dataset.tail()


# In[51]:


future_dataset=pd.concat([df,future_dataset])


# In[52]:


future_dataset


# In[53]:


future_dataset['forecast_sarimax2']=model1.predict(start=104,end=124,dynamic=True)


# In[54]:


future_dataset[['Sales','forecast_sarimax2']].plot(figsize=(12,10))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




