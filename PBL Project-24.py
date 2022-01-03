#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries and Packages

# In[172]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# # Importing Dataset

# In[173]:


df = pd.read_csv('winequalityN.csv')
df


# # Preprocessing 

# In[174]:


df.head()


# In[175]:


df.info()


# In[176]:


df.describe()


# In[177]:


df.isnull().sum()


# # Data Visualization

# In[178]:


df.hist(bins=20,figsize=(10,10))
plt.show()


# In[179]:


df.boxplot()


# In[180]:


label_encoder = LabelEncoder()
for col in df.columns:
    df[col] = label_encoder.fit_transform(df[col])
df


# # Correlation 

# In[181]:


df.corr()


# In[182]:


plt.figure(figsize=[10,6],facecolor='white')
sns.heatmap(df.corr(),annot=True)


# In[183]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB


# In[184]:


X = df.drop(columns=["quality"],axis=1)
y = df["quality"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state = 42)


# In[185]:


models = [
    "Linear SVM",
    "Random Forest",
    "AdaBoost",
    "Naive Bayes",
]


# In[186]:


classifiers = [
    SVC(kernel="linear", C=0.025),
    RandomForestClassifier(n_estimators=200),
    AdaBoostClassifier(),
    GaussianNB()
]


# In[187]:


print(classification_report(y_test, y_pred))


# In[188]:


train_scores = []
test_scores = []
for i in classifiers:
    model = model.fit(X_train, y_train)
    train_scores.append(model.score(X_train, y_train))
    test_scores.append(model.score(X_test, y_test))


# In[189]:


pd.DataFrame({'classifiers':names, 'train scores':train_scores, 'test scores':test_scores})


# In[190]:


forest = RandomForestClassifier()
forest.fit(X_train, y_train)
train_score = forest.score(X_train, y_train)
test_score = forest.score(X_test, y_test)
print("Training score is:",train_score)
print("Testing score is:",test_score)


# In[191]:


y_pred_test = forest.predict(X_test)


# In[192]:


accuracy_score(y_test,y_pred_test)


# In[ ]:





# In[ ]:




