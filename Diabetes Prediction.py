#!/usr/bin/env python
# coding: utf-8

# # Importing the dependencies

# In[54]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# # Data Collection and Analysis

# In[55]:


#loading the diabetes to a panda DataFarame
diabetes_dataset=pd.read_csv('diabetes.csv')


# In[56]:


#printing the first 5 rows of the dataset
diabetes_dataset.head()


# In[57]:


# Number od rows and columns in this dataset
diabetes_dataset.shape


# In[58]:


#getting the statical measure of the data
diabetes_dataset.describe()


# In[59]:


diabetes_dataset['Outcome'].value_counts()


# ## 0--> Non-Diabetic
# ## 1--> Diabetic
# 

# In[60]:


diabetes_dataset.groupby('Outcome').mean()


# In[61]:


#seprating the data and labels
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']


# In[62]:


print(X)


# In[63]:


print(Y)


# # Data Standardization

# In[64]:


scaler=StandardScaler()


# In[65]:


scaler.fit(X)


# In[66]:


Standardized_data=scaler.transform(X)
print(Standardized_data)


# In[67]:


X= Standardized_data
Y=diabetes_dataset['Outcome']


# In[68]:


print(X)
print(Y)


# # Train and Split 

# In[69]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2, stratify=Y,random_state=2)


# In[70]:


print(X.shape, X_train.shape,X_test.shape)


# # Training the Model

# In[71]:


classifer=svm.SVC(kernel='linear')


# In[72]:


#training the support vector Machine Classifier
classifer.fit(X_train,Y_train)


# 
# # Model Evaluation

# # Accuracy Score

# In[73]:


#accuracy score on the trainig data
X_train_prediction = classifer.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction, Y_train)


# In[74]:


print('Accuracy score of the trainig data : ', training_data_accuracy)


# In[75]:


#accuracy score on the test data
X_test_prediction = classifer.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction, Y_test)


# In[76]:


print('Accuracy score of the test data : ', test_data_accuracy)


# # Making a Predictive System

# In[82]:


input_data=(4,110,92,0,0,37.6,0.191,30)

#changing the input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

#standarized the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifer.predict(std_data)
print(prediction)

if (prediction[0]==0):
 print('The Person is not Diabetic')
else: 
    print('The Person Is Diabetic')


# In[ ]:




