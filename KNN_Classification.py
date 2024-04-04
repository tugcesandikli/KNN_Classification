#!/usr/bin/env python
# coding: utf-8

# In[136]:


import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[137]:


data = pd.read_csv("/Users/tugcesandikli/Downloads/data.csv")


# In[138]:


data.head()


# In[139]:


data.info()


# In[140]:


data.drop(["Unnamed: 32","id"],axis=1,inplace=True)


# In[141]:


M =data[data.diagnosis=="M"]
B =data[data.diagnosis=="B"]


# In[142]:


M.info()


# In[143]:


B.info()


# In[144]:


plt.scatter(M.radius_mean,M.area_mean,color="red",label="malignant")
plt.scatter(B.radius_mean,B.area_mean,color="green",label="benign")
plt.legend()
plt.xlabel("radius_mean")
plt.ylabel("area_mean")
plt.show()


# In[145]:


plt.scatter(M.radius_mean,M.texture_mean,color="red",label="malignant")
plt.scatter(B.radius_mean,B.texture_mean,color="green",label="benign")
plt.legend()
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.show()


# In[146]:


data.diagnosis=[1 if each=="M" else 0 for each in data.diagnosis]


# In[147]:


data.diagnosis


# In[148]:


y = data.diagnosis.values


# In[149]:


x_data= data.iloc[:,1:3].values


# In[150]:


x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data))


# In[151]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=1)


# In[152]:


# knn model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train,y_train)


# In[153]:


y_head = knn.predict(x_test)


# In[154]:


y_head


# In[155]:


print("when k is {}, accuracy of KNN classification {} " .format(3,knn.score(x_test,y_test)))


# In[156]:


test_accuracy=[]
for each in range(1,15):
    knn2=KNeighborsClassifier(n_neighbors=each)
    knn2.fit(x_train,y_train)
    test_accuracy.append(knn2.score(x_test,y_test))
    
plt.figure(figsize=(5,5))
plt.plot(range(1, 15),test_accuracy)
plt.title("k values vs accuracy")
plt.xlabel("k labels")
plt.ylabel("accuracy")
plt.grid()
plt.show()
print("best accuracy is {} with K ={}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))

