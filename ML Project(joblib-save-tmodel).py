#!/usr/bin/env python
# coding: utf-8

# In[4]:


#LOAD PACKAGES and DATASET
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib as jb
from sklearn import tree
music_data = pd.read_csv("music.csv")
music_data


# In[6]:


# DEFINING FEATURES AND TARGET VARIABLES
X = music_data.drop(columns = ['genre'])
Y = music_data['genre']
print(X)


# In[8]:


print(Y)


# In[12]:


#MODEL PREDICTION BEFORE TRAINING AND TESTING
model = DecisionTreeClassifier()
model.fit(X, Y)

#store trained model in a file
jb.dump(model, 'music-recommender.joblib' )


# In[14]:


#using saved trained model file in predictions
model = jb.load('music-recommender.joblib')


# In[18]:


tree.export_graphviz(model, out_file = 'music-recommender.dot', 
                     feature_names = ['age', 'gender'], 
                     class_names = sorted(Y.unique()), 
                     label = 'all',
                     rounded=True,
                     filled=True)


# In[158]:


# PREDICTIONS
predictions = model.predict([[21, 1], [22, 0]])
predictions


# In[ ]:




