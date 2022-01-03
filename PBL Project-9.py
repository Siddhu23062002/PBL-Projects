#!/usr/bin/env python
# coding: utf-8

# # Summarizing Dataset

# In[6]:


import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


# In[7]:


l1=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny','Rainy','Sunny','Overcast','Overcast','Rainy']
l2=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']
l3 =['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']
df = pd.DataFrame(list(zip(l1,l2,l3)), columns =['Weather','Temp','Play']) 
print(df) 


# In[8]:


df.head()


# In[9]:


df.info()


# In[10]:


df.describe()


# # Using LabelEncoder

# In[11]:


label_encoder = LabelEncoder()
for col in df.columns:
    df[col] = label_encoder.fit_transform(df[col])
df


# In[12]:


l2=label_encoder.fit_transform(l2)
print("Temperature:",l2)


# In[13]:


l1=label_encoder.fit_transform(l1)
print("Weather:",l1)


# # Using zip function

# In[14]:


features=list(zip(l1,l2))
print("Combined weather and temp:",features)


# In[15]:


X = df.drop(columns=["Play"],axis=1)
Y = df["Play"]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2,random_state = 42)


# # Creating & fitting Naive Bayes Classifier on dataset

# In[16]:


model = GaussianNB()
model.fit(x_train, y_train)
train_score = model.score(x_train, y_train)
test_score = model.score(x_test, y_test)
print("Training score is:",train_score)
print("Testing score is:",test_score)


# # Predicting Value

# In[18]:


model.fit(features,l2)
predicted= model.predict([[0,2]])
print("Predicted Value is:",predicted)


# In[ ]:




