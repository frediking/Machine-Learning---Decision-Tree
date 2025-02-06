#!/usr/bin/env python
# coding: utf-8

# In[51]:


#LOAD PACKAGES and DATASET
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
music_data = pd.read_csv("music.csv")
music_data


# In[53]:


# DEFINING FEATURES AND TARGET VARIABLES
X = music_data.drop(columns = ['genre'])
Y = music_data['genre']
print(X)


# In[55]:


print(Y)


# In[57]:


#MODEL PREDICTION BEFORE TRAINING AND TESTING
#model = DecisionTreeClassifier()
#model.fit(X, Y)
#predictions = model.predict([[21, 1], [22, 0]])
#predictions -  'Hiphop', 'Dance'


# In[59]:


#TRAINING AND TESTING
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)


# In[125]:


#PREDICTIONS AFTER TRAINING AND TESTING
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
predictions = model.predict(X_test)
predictions


# In[129]:


#CALCULATING ACCURACY FOR MODEL
score = accuracy_score(Y_test, predictions)
score
print('Model accuracy is :', score, 'which means 100 %')


# In[ ]:




