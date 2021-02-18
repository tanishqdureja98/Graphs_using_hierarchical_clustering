#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv(r'C:\Users\hp\Downloads\Project_2.csv')
df


# # KMEANS

# In[3]:


from sklearn.cluster import KMeans


# In[4]:


X=pd.DataFrame(df.iloc[:,[5,6,7]])
X


# In[5]:


X.columns=['math_score','reading_score','writing_score']
X


# In[6]:


df.isnull().sum()


# In[7]:


sse = []
k_rnge = range(1,10)
for k in k_rnge:
    km = KMeans(n_clusters=k)
    km.fit(X)
    sse.append(km.inertia_)
sse


# In[8]:


plt.xlabel('KMean')
plt.plot(k_rnge,sse)


# In[9]:


y_pred = KMeans(n_clusters=3, random_state=1).fit_predict(X)
y_pred


# In[10]:


km=KMeans(n_clusters=3)
km.fit(X)
km.labels_


# In[11]:


colormap=np.array(['red','blue','green'])
plt.figure(figsize=(20,8))
plt.subplot(1,3,1)
plt.scatter(X.math_score,X.reading_score,c=colormap[y_pred],s=30)
plt.title('KMEANS1')
plt.subplot(1,3,2)
plt.scatter(X.reading_score,X.writing_score, c=colormap[y_pred],s=30)
plt.title('KMEANS2')
plt.subplot(1,3,3)
plt.scatter(X.writing_score,X.math_score,c=colormap[y_pred],s=30)
plt.title('KMEANS3')


# # hierarchical clustering

# In[12]:


X=df.iloc[:,[5,6,7]].values
X


# In[13]:


import scipy.cluster.hierarchy as sch


# In[14]:


plt.figure(figsize=(70,70))
dendrogram=sch.dendrogram(sch.linkage(X,method='complete'))
plt.title('Dendrogram')
plt.show()


# In[15]:


from sklearn.cluster import AgglomerativeClustering
ab=AgglomerativeClustering(n_clusters=4,affinity='euclidean',linkage='average')


# In[16]:


y_ab=ab.fit_predict(X)
y_ab


# In[17]:


len(X)


# In[18]:


len(y_ab)


# In[19]:


plt.figure(figsize=(12,8))
plt.scatter(X[:, 0], X[:, 1],c=y_ab, cmap='plasma',s=80)


# In[20]:


plt.figure(figsize=(12,8))
plt.scatter(X[:, 1], X[:, 2],c=y_ab, cmap='cividis',s=80)


# In[21]:


plt.figure(figsize=(12,8))
plt.scatter(X[:, 0], X[:, 2],c=y_ab, cmap='Dark2',s=80)


# # DBSCAN

# In[22]:


from sklearn.cluster import DBSCAN


# In[23]:


X=df.iloc[:,[5,6,7]].values
X


# In[24]:


dbscan=DBSCAN(eps = 4, min_samples = 8)
clusters=dbscan.fit_predict(X)


# In[25]:


clusters


# In[26]:


plt.figure(figsize=(12,8))
plt.scatter(X[:, 0], X[:, 1],c=clusters, cmap='plasma',s=80)
plt.title('DBSCAN')


# In[27]:


plt.figure(figsize=(12,8))
plt.scatter(X[:, 1], X[:, 2],c=clusters, cmap='plasma',s=80)
plt.title('DBSCAN')


# In[28]:


plt.figure(figsize=(12,8))
plt.scatter(X[:, 0], X[:, 2],c=clusters, cmap='plasma',s=80)
plt.title('DBSCAN')

