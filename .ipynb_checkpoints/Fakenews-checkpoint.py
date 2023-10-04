#!/usr/bin/env python
# coding: utf-8

# ###Importing necessary Libraries

# In[1]:


import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ###Reading CSV

# In[ ]:


data=pd.read_csv("news.csv")


# ###Displaying Top 5 Rows

# In[3]:


data.head()


# ###Displaying count of Rows And Columns

# In[4]:


data.shape


# In[5]:


data.isna().sum()


# In[6]:


data=data.fillna(' ')


# In[7]:


data.isna().sum()


# In[8]:


data['title'][0]


# ###Stemming

# In[ ]:


ps=PorterStemmer()
def stemming(title):
    stemmed_title=re.sub('[^a-zA-Z]'," ",title)
    stemmed_title=stemmed_title.lower()
    stemmed_title=stemmed_title.split()
    stemmed_title=[ps.stem(word) for word in stemmed_title if not word in stopwords.words('english')]
    stemmed_title=" ".join(stemmed_title)
    return stemmed_title


# In[10]:


data['title']=data['title'].apply(stemming)


# In[11]:


data['title'][0]


# ###Seperating data and label

# In[12]:


X=data['title'].values
Y=data['label'].values


# In[13]:


print(X)
print(Y)


# ###Coverting text data to numerical data

# In[14]:


vector=TfidfVectorizer()
vector.fit(X)
X=vector.transform(X)


# In[15]:


print(X)


# ###Train test Split

# In[16]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=42)


# In[17]:


model=LogisticRegression()
model.fit(X_train,Y_train)


# In[18]:


train_y_pred=model.predict(X_train)
print("Train Accuracy=",accuracy_score(train_y_pred,Y_train))


# In[19]:


test_y_pred=model.predict(X_test)
print("Train Accuracy=",accuracy_score(test_y_pred,Y_test))


# ###Prediction System

# In[20]:


input=X_test[3]
prediction=model.predict(input)
if prediction[0]==1:
    print("Fake News")
else:
    print("Real News")


# In[ ]:




