#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas_datareader as pdr


# In[2]:


key = "fc7ac86a44d2346062fddd2f060ca6110b3cc3e1"


# In[3]:


df = pdr.get_data_tiingo('MSFT',api_key=key)


# In[4]:


df.to_csv('MSFT.csv')


# In[5]:


df.head()


# In[6]:


df.tail()


# In[7]:


df1=df.reset_index()['close']
df1


# In[11]:


import matplotlib.pyplot as plt
plt.plot(df1)


# In[25]:


import numpy as np
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))


# In[26]:


df1


# In[27]:


training_size=int(len(df1)*0.70)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]
training_size,test_size


# In[28]:


import numpy as np
def create_dataset(dataset,time_step=1):
    datax,datay=[], []
    for i in range(len(dataset)-time_step-1):
        a=dataset[i:(i+time_step),0]
        datax.append(a)
        datay.append(dataset[i+time_step,0])
    return np.array(datax), np.array(datay)


# In[29]:


time_step=100
x_train,y_train = create_dataset(train_data, time_step)
x_test,y_test = create_dataset(test_data, time_step)


# In[30]:


print(x_train.shape),print(y_train.shape)


# In[31]:


print(x_test.shape),print(y_test.shape)


# In[32]:


#creating stacked lstm model
#lstm requires data in 3 dimensions so
x_train=x_train.reshape(x_train.shape[0],x_train.shape[1], 1)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1], 1)


# In[33]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[34]:


model= Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


# In[35]:


model.fit(x_train,y_train,validation_data=(x_test,y_test), epochs=100,batch_size=64, verbose=1)


# In[36]:


train_predict=model.predict(x_train)
test_predict=model.predict(x_test)


# In[37]:


train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# In[38]:


import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


# In[39]:


math.sqrt(mean_squared_error(y_test,test_predict))


# In[50]:


import numpy as np
look_back = 100
trainpredictplot = np.empty_like(df1)
trainpredictplot[:,:] = np.nan
trainpredictplot[look_back:len(train_predict)+look_back, :] = train_predict

# test prediction for plotting
testpredictplot = np.empty_like(df1)
testpredictplot[:,:] = np.nan
testpredictplot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict

#ploting prediction
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainpredictplot)
plt.plot(testpredictplot)
plt.show()

# orange ir train data
#green is test data




# In[ ]:




