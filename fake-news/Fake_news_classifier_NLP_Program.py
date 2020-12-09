#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 


# In[2]:


trainData=pd.read_csv('fake-news/train.csv')
data=trainData
testData=pd.read_csv('fake-news/test.csv')
data.head()


# In[3]:


data.shape


# In[4]:


testData.head()


# In[5]:


data.isnull().sum()


# In[6]:


testData.isnull().sum()


# In[38]:


testData.fillna(0)


# In[7]:


data=data.dropna()
data.shape


# In[8]:


X_train=data.drop('label',axis=1)
X_train.head()


# In[15]:


y=data['label']


# In[16]:


message=X_train.copy()
message.reset_index(inplace=True)


# In[17]:


import nltk
import re
from nltk.corpus import stopwords


# In[18]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
#Creating a List
corpus=[]

for i in range(0,len(message)):
    review=re.sub('[^A-Za-z]',' ',message['title'][i])
    review=review.lower()
    review=review.split()
    
    review=[ps.stem(word) for word in review if not word in stopwords.words('english')]
    review=' '.join(review)
    corpus.append(review)


# In[19]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_v=TfidfVectorizer(max_features=5000,ngram_range=(1,3))
X=tfidf_v.fit_transform(corpus).toarray()


# In[20]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)


# In[21]:


from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()


# In[24]:


from sklearn.metrics import confusion_matrix 
from sklearn import metrics


# In[23]:


classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
score


# In[25]:


cm = confusion_matrix(y_test,pred)  
cm


# In[27]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# In[28]:


rf = RandomForestClassifier()
param = {'n_estimators': [10, 150, 300],
        'max_depth': [30, 60, 90, None]}

gs = GridSearchCV(rf, param, cv=5, n_jobs=-1)# n_jobs=-1 for parallelizing search
gs_fit = gs.fit(X_train, y_train)
pd.DataFrame(gs_fit.cv_results_).sort_values('mean_test_score', ascending=False).head()


# In[29]:


gs_fit.cv_results_


# In[31]:


clf = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=0)
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)
metrics.accuracy_score(list(y_test), predicted)


# In[35]:


cm = confusion_matrix(y_test,predicted)  
cm


# In[36]:


import matplotlib.pyplot as plt
cm = confusion_matrix(list(y_test), predicted)
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
labels=list(np.unique(list(y_test)))
plt.title('Confusion matrix of the classifier', y=-0.1)
fig.colorbar(cax)
ax.set_xticklabels([''] + labels, rotation=40)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[39]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
#Creating a List
corpus=[]

for i in range(0,len(testData)):
    review=re.sub('[^A-Za-z]',' ',message['title'][i])
    review=review.lower()
    review=review.split()
    
    review=[ps.stem(word) for word in review if not word in stopwords.words('english')]
    review=' '.join(review)
    corpus.append(review)


# In[40]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_v=TfidfVectorizer(max_features=5000,ngram_range=(1,3))
x_test=tfidf_v.fit_transform(corpus).toarray()


# In[49]:


y_pred_test = clf.predict(x_test)


# In[51]:


df_sub = pd.DataFrame()
df_sub['id'] =testData['id']
df_sub['label'] = y_pred_test


# In[52]:


submission_data = pd.read_csv('fake-news/submit.csv')


# In[53]:


len(testData['id']), len(y_pred_test)


# In[54]:


df_sub['label'] = df_sub['label'].apply(lambda x:0 if x<=0.5 else 1)


# In[55]:


df_sub.to_csv('sample_submission_random_forest_classifier.csv', index=False)


# In[56]:


df_sub.head()


# In[ ]:




