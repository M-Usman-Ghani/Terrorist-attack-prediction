#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[2]:


terrorism_dataset = pd.read_csv("globalterrorismdb_0718dist.csv",encoding = "ISO-8859-1")


# In[4]:


terrorism_dataset


# In[5]:


td2=terrorism_dataset
for i in terrorism_dataset.columns:
    x =terrorism_dataset[f"{i}"].isna().sum()
    if(x>50000):
        td2 = td2.drop(f"{i}",axis=1)
td2   


# In[4]:


pd.set_option('display.max_columns', None)


# In[6]:


td2.dropna(subset=["provstate"],axis=0 , inplace=True)
td2.dropna(subset=["city"],axis=0 , inplace=True)

td2.drop("attacktype1_txt",axis=1,inplace=True)
td2.drop("targtype1_txt",axis=1,inplace=True)
td2.drop("targsubtype1_txt",axis=1,inplace=True)
td2.drop("natlty1_txt",axis=1,inplace=True)
td2.drop("region_txt",axis=1,inplace=True)
td2.drop("country_txt",axis=1,inplace=True)
td2.drop("weaptype1_txt",axis=1,inplace=True)
td2.drop("weapsubtype1_txt",axis=1,inplace=True)

td2.drop("INT_LOG",axis=1,inplace=True)
td2.drop("INT_IDEO",axis=1,inplace=True)
td2.drop("INT_MISC",axis=1,inplace=True)
td2.drop("INT_ANY",axis=1,inplace=True)
td2.drop("dbsource",axis=1,inplace=True)
td2.drop("corp1",axis=1,inplace=True)
td2.drop("ishostkid",axis=1,inplace=True)
td2.dropna(subset=["doubtterr"],axis=0 , inplace=True)
td2.dropna(subset=["multiple"],axis=0 , inplace=True)
td2.dropna(subset=["specificity"],axis=0 , inplace=True)

td2.drop("guncertain1",axis=1,inplace=True)
td2.drop("longitude",axis=1,inplace=True)
td2.drop("latitude",axis=1,inplace=True)
td2.drop("targsubtype1",axis=1,inplace=True)
td2.drop("weapsubtype1",axis=1,inplace=True)


# In[7]:


td2["nwound"].fillna(3,inplace=True)
td2["nkill"].fillna(2,inplace=True)
td2.natlty1.fillna(0,inplace=True)
td2.target1.fillna("Unknown",inplace=True)


# In[8]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
encode_result = td2.apply(label_encoder.fit_transform)


# In[9]:


encode_result


# In[12]:


import matplotlib.pyplot as plt  
corrmat = encode_result.corr()
plt.figure(figsize=(28,28))
#plot heat map
g=sns.heatmap(corrmat,annot=True,cmap="RdYlGn")


# In[10]:


label_encoder.classes_


# In[19]:


x=encode_result[["iyear","country","region","city","provstate","gname","target1","weaptype1"]]
y=encode_result["attacktype1"]


# In[20]:


from sklearn.model_selection import train_test_split
np.random.seed(42)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


# In[21]:


# Setup model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()


model.fit(x_train, y_train)
model.score(x_test, y_test)


# In[23]:


from sklearn.metrics import confusion_matrix
rclassifier_pred = model.predict(x_test)
a=confusion_matrix(y_test,rclassifier_pred)
a1=a.flatten()
x=a1[0:4]
print(x)


# In[25]:


labels = 'TruePositive', 'FalseNegative', 'FalsePositive', 'TrueNegative'
colors = ['gold', 'blue', 'lightcoral', 'lightskyblue']

# Plot
plt.pie(x, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')
plt.show()


# In[26]:


from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler()

scale = scalar.fit_transform(td2[['iyear','country']])
df_scale = pd.DataFrame(scale, columns = ['iyear','country']);
df_scale.head(5)


# In[43]:


from sklearn.cluster import KMeans
import sklearn.cluster as cluster
km=KMeans(n_clusters=2)
y_predicted = km.fit_predict(td2[['iyear','region','attacktype1','country','imonth','iday','nkill','nwound']])
td2['Clusters'] = km.labels_
sns.scatterplot(x="iyear", y="country",hue = 'Clusters', 
data=td2,palette='Set2')


# In[44]:


#Finding optimum value of K
K=range(2,12)
wss = []
for k in K:
    kmeans=cluster.KMeans(n_clusters=k)
    kmeans=kmeans.fit(df_scale)
    wss_iter = kmeans.inertia_
    wss.append(wss_iter)
    
import matplotlib.pyplot as plt    
#Plotting the graph
plt.xlabel('K')
plt.ylabel('Within-Cluster-Sum of Squared Errors (WSS)')
plt.plot(K,wss)    


# 
# ![image.png](attachment:image.png)
# ![image-2.png](attachment:image-2.png)
# ![image-3.png](attachment:image-3.png)
# ![image-4.png](attachment:image-4.png)
# 

# In[46]:


km=KMeans(n_clusters=4)
y_predicted = km.fit_predict(td2[['iyear','region','attacktype1','country','imonth','iday','nkill','nwound']])
y_predicted
td2['Clusters'] = km.labels_
sns.scatterplot(x="iyear", y="country",hue = 'Clusters', 
data=td2,palette='Set2')


# In[54]:


td2


# In[55]:


scale = scalar.fit_transform(td2[['iyear','region']])
df_scale = pd.DataFrame(scale, columns = ['iyear','region']);


# In[56]:


K=range(2,12)
wss = []
for k in K:
    kmeans=cluster.KMeans(n_clusters=k)
    kmeans=kmeans.fit(df_scale)
    wss_iter = kmeans.inertia_
    wss.append(wss_iter)
    
plt.xlabel('K')
plt.ylabel('Within-Cluster-Sum of Squared Errors (WSS)')
plt.plot(K,wss)


# In[42]:


km=KMeans(n_clusters=4)
y_predicted = km.fit_predict(td2[['iyear','region','attacktype1','country','imonth','iday']])
td2['Clusters2'] = km.labels_
sns.scatterplot(x="iyear", y="region",hue = 'Clusters2', 
data=td2,palette='Set2')


# In[47]:


km=KMeans(n_clusters=4)
y_predicted = km.fit_predict(td2[['iyear','region','attacktype1','country','imonth','iday','nkill','nwound']])
td2['Clusters'] = km.labels_
sns.scatterplot(x="country", y="attacktype1",hue = 'Clusters', 
data=td2,palette='Set2')


# In[41]:


K=range(2,12)
wss = []
for k in K:
    kmeans=cluster.KMeans(n_clusters=k)
    kmeans=kmeans.fit(df_scale)
    wss_iter = kmeans.inertia_
    wss.append(wss_iter)
    
plt.xlabel('K')
plt.ylabel('Within-Cluster-Sum of Squared Errors (WSS)')
plt.plot(K,wss)


# In[50]:


km=KMeans(n_clusters=4)
y_predicted = km.fit_predict(td2[['iyear','region','attacktype1','country','imonth','iday','nkill','nwound']])
td2['Clusters'] = km.labels_
sns.scatterplot(x="iyear", y="nkill",hue = 'Clusters', 
data=td2,palette='Set2')


# In[51]:


K=range(2,12)
wss = []
for k in K:
    kmeans=cluster.KMeans(n_clusters=k)
    kmeans=kmeans.fit(df_scale)
    wss_iter = kmeans.inertia_
    wss.append(wss_iter)
    
plt.xlabel('K')
plt.ylabel('Within-Cluster-Sum of Squared Errors (WSS)')
plt.plot(K,wss)


# In[52]:


km=KMeans(n_clusters=4)
y_predicted = km.fit_predict(td2[['iyear','region','attacktype1','country','imonth','iday','nkill','nwound']])
td2['Clusters'] = km.labels_
sns.scatterplot(x="region", y="nkill",hue = 'Clusters', 
data=td2,palette='Set2')


# In[53]:


K=range(2,12)
wss = []
for k in K:
    kmeans=cluster.KMeans(n_clusters=k)
    kmeans=kmeans.fit(df_scale)
    wss_iter = kmeans.inertia_
    wss.append(wss_iter)
    
plt.xlabel('K')
plt.ylabel('Within-Cluster-Sum of Squared Errors (WSS)')
plt.plot(K,wss)


# In[ ]:




