#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


data=pd.read_csv('Desktop/brca.csv')


# In[3]:


data.head()


# In[4]:


data.isnull().values.any()


# In[5]:


data.drop(["Unnamed: 0"],axis=1,inplace=True)


# In[6]:


data.shape


# In[7]:


data.info()


# In[8]:


data.describe()


# In[9]:


data.y = [1 if each == "M" else 0 for each in data.y]
#benign-0
#malignant-1


# In[10]:


data['y'].value_counts()


# In[11]:


data.tail()
#converted the y values to either 0 or 1


# In[12]:


data.groupby('y').mean()


# In[13]:


#seperating feature and target data
X=data.drop(columns='y',axis=1)
y=data['y']


# In[14]:


X


# In[15]:


y


# In[16]:


#splitting the data to training and testing
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)


# In[17]:


print(X.shape,X_train.shape,X_test.shape)


# In[18]:


#model training-Logistic regression
model=LogisticRegression()


# In[19]:


model.fit(X_train,y_train)


# In[20]:


X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,y_train)
print('Accuracy on training data:{}'.format(training_data_accuracy))


# In[21]:


X_test_prediction=model.predict(X_test)
testing_data_accuracy=accuracy_score(X_test_prediction,y_test)
print('Accuracy on testing data:{}'.format(testing_data_accuracy))


# In[24]:


#building the predictive system
input_data=(11.08,18.83,73.3,361.6,0.1216,0.2154,0.1689,0.06367,0.2196,0.0795,0.2114,1.027,1.719,13.99,0.007405,0.04549,0.04588,0.01339,0.01738,0.004435,13.24,32.82,91.76,508.1,0.2184,0.9379,0.8402,0.2524,0.4154,0.1403)

#result should be 1 as the data is for the malignant tumor
input_data_to_numpy_array=np.asarray(input_data)

#reshaping the numpy array 
input_data_reshape=input_data_to_numpy_array.reshape(1,-1)
prediction=model.predict(input_data_reshape)
print(prediction)

if prediction[0]==1:
    print("The breast cancer is Malignant")
else:
    print("The breast cancer is Benign")
    


# In[ ]:




