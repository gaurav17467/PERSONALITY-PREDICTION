#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


#reading data from csv file
df=pd.read_csv('./pseudo_facebook.csv')


# In[4]:


df


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


#dropping useless columns
df.drop(['userid','age','dob_day','dob_year','dob_month','mobile_likes','mobile_likes_received','www_likes','www_likes_received','gender'],axis=1,inplace=True)


# In[8]:


df


# In[9]:


#deleting rows with unavailable values
df.dropna(subset = ['tenure'], inplace=True)


# In[10]:


df.describe()


# In[11]:


df


# In[12]:


#converting tenure value from float to integer
df['tenure'] = df['tenure'].apply(lambda x: int(x))


# In[13]:


df


# # EXTROVERSION

# In[14]:


#since extroverted person has more friend count
#So,we have assumed that a extroverted person will have friend_count 
#greater than the mean of all friend_count value


# In[15]:


#creating numpy array
w=np.array(df)


# In[16]:


w


# In[17]:


w.shape


# In[18]:


#removing friend count column
X=np.delete(w,1,1)


# In[19]:


X


# In[20]:


Y=np.delete(w,[0,2,3,4],1)


# In[21]:


Y


# In[22]:


X.shape


# In[23]:


Y.shape


# In[24]:


#dividing the data in training and testing data


# In[25]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)


# In[26]:


X_train.shape


# In[27]:


X_train


# In[28]:


X_test.shape


# In[29]:


X_test


# In[30]:


Y_train.shape


# In[31]:


Y_train


# In[32]:


Y_test.shape


# In[33]:


Y_test


# In[34]:


#calculating mean of y_train


# In[35]:


mn=np.mean(Y_train)


# In[36]:


mn


# In[37]:


#since y should have binary value 
#so changing y_train and y_test


# In[38]:


Y_train[Y_train<mn]=0


# In[39]:


Y_train[Y_train>mn]=1


# In[40]:


Y_test[Y_test<mn]=0


# In[41]:


Y_test[Y_test>mn]=1


# In[42]:


Y_test


# In[43]:


#NAIVE BAYES


# In[48]:


from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB() 
gnb.fit(X_train,Y_train.ravel())
Y_pred = gnb.predict(X_test)
from sklearn import metrics
from sklearn.metrics import f1_score
naive_acc=metrics.accuracy_score(Y_test, Y_pred)*100
naive_f1 = f1_score(Y_test, Y_pred, average='binary')*100
print('F-Measure: ',naive_f1)
print("Gaussian Naive Bayes model accuracy(in %):",naive_acc)


# In[49]:


#SUPPORT VECTOR MACHINE


# In[50]:


from sklearn import svm
from sklearn import metrics
from sklearn.metrics import f1_score


# In[51]:


cls=svm.SVC(kernel="rbf",gamma="auto")


# In[52]:


cls.fit(X_train,Y_train.ravel())


# In[53]:


pred=cls.predict(X_test)


# In[72]:


svm_acc=metrics.accuracy_score(Y_test, pred)*100
svm_f1=f1_score(Y_test, pred, average='binary')*100
print('F-Measure: ',svm_f1)
print("SVM model accuracy is: ",svm_acc)


# In[55]:


#DECISION TREE


# In[56]:


from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import f1_score
dt = DecisionTreeClassifier(criterion = 'entropy', random_state = 42)
dt.fit(X_train, Y_train.ravel())
dt_pred = dt.predict(X_test)
dt_acc=metrics.accuracy_score(Y_test, dt_pred)*100
dt_f1=f1_score(Y_test,dt_pred, average='binary')*100
print('F-Measure: ',dt_f1)
print("Decision Tree model accuracy is: ",dt_acc)


# In[57]:


#RANDOM FOREST


# In[58]:


from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import f1_score
rfc = RandomForestClassifier(criterion = 'entropy', random_state = 42)
rfc.fit(X_train, Y_train.ravel())
rfc_pred = rfc.predict(X_test)
rfc_acc=metrics.accuracy_score(Y_test, rfc_pred)*100
rfc_f1=f1_score(Y_test,rfc_pred, average='binary')*100
print('F-Measure: ',rfc_f1)
print("Random Forest model accuracy is: ",rfc_acc)


# In[69]:


import matplotlib.pyplot as plt


# In[79]:


plt.plot(naive_acc,naive_f1,'^')
plt.plot(svm_acc,svm_f1,'+')
plt.plot(dt_acc,dt_f1,'o')
plt.plot(rfc_acc,rfc_f1,'*')


# # CONSIOUSNESS

# In[81]:


#since consious person uses facebook less
#we assumed that a person who uses less than mean of 'tenure' column as consious 


# In[82]:


df


# In[83]:


#creating numpy array


# In[84]:


u=np.array(df)


# In[85]:


u


# In[86]:


#deleting tenure column


# In[87]:


x=np.delete(u,0,1)


# In[88]:


x


# In[89]:


y=np.delete(u,[1,2,3,4],1)


# In[90]:


y


# In[91]:


x.shape


# In[92]:


y.shape


# In[93]:


#diving the data in training and testing data


# In[94]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[95]:


x_train


# In[96]:


x_train.shape


# In[97]:


x_test.shape


# In[98]:


y_train.shape


# In[99]:


y_test.shape


# In[100]:


y_train


# In[101]:


#calculating mean of y_train


# In[102]:


tmean=np.mean(y_train)


# In[103]:


tmean


# In[104]:


#converting value of y as binary


# In[105]:


y_train[y_train<tmean]=1


# In[106]:


y_train[y_train>tmean]=0


# In[107]:


y_test[y_test<tmean]=1


# In[108]:


y_test[y_test>tmean]=0


# In[109]:


y_train


# In[110]:


y_test


# In[111]:


#NAIVE BAYES


# In[112]:


from sklearn.naive_bayes import GaussianNB 
gnb1 = GaussianNB() 
gnb1.fit(x_train,y_train.ravel())
y_pred = gnb1.predict(x_test)
from sklearn import metrics 
from sklearn.metrics import f1_score
naive1_acc=metrics.accuracy_score(y_test, y_pred)*100
naive1_f1 = f1_score(y_test, y_pred, average='binary')*100
print('F-Measure: ',naive1_f1)
print("Gaussian Naive Bayes model accuracy(in %):",naive1_acc)


# In[ ]:


# DECISION TREE


# In[113]:


from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
dt1 = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dt1.fit(x_train, y_train.ravel())
dt1_pred = dt1.predict(x_test)
from sklearn.metrics import f1_score
dt1_acc=metrics.accuracy_score(y_test, dt1_pred)*100
dt1_f1 = f1_score(y_test, dt1_pred, average='binary')*100
print('F-Measure: ',dt1_f1)
print("Decision Tree model accuracy is: ",dt1_acc)


# In[ ]:


# RANDOM FOREST


# In[114]:


from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import f1_score
rfc1 = RandomForestClassifier(criterion = 'entropy', random_state = 42)
rfc1.fit(x_train, y_train.ravel())
rfc1_pred = rfc1.predict(x_test)
rfc1_acc=metrics.accuracy_score(y_test, rfc1_pred)*100
rfc1_f1=f1_score(y_test,rfc_pred, average='binary')*100
print('F-Measure: ',rfc1_f1)
print("Random Forest model accuracy is: ",rfc1_acc)


# In[115]:


#SUPPORT VECTOR MACHINE


# In[121]:


from sklearn import svm
from sklearn import metrics
from sklearn.metrics import f1_score


# In[122]:


cls1=svm.SVC(kernel="rbf",gamma="auto")


# In[123]:


cls1.fit(x_train,y_train.ravel())


# In[124]:


svm_pred=cls1.predict(x_test)


# In[125]:


svm1_acc=metrics.accuracy_score(y_test, svm_pred)*100
svm1_f1=f1_score(y_test, svm_pred, average='binary')*100
print('F-Measure: ',svm1_f1)
print("SVM model accuracy is: ",svm1_acc)


# In[126]:


import matplotlib.pyplot as plt


# In[127]:


plt.plot(naive1_acc,naive1_f1,'^')
plt.plot(svm1_acc,svm1_f1,'+')
plt.plot(dt1_acc,dt1_f1,'o')
plt.plot(rfc1_acc,rfc1_f1,'*')


# # NEUROTISM

# In[128]:


#since a neurotic person likes more posts
#so, a person is assumed neurotic if the person has likes more than average


# In[129]:


#creating numpy array


# In[130]:


v=np.array(df)


# In[131]:


v


# In[132]:


#removing likes column


# In[133]:


x1=np.delete(v,3,1)


# In[134]:


x1


# In[135]:


y1=np.delete(v,[0,1,2,4],1)


# In[136]:


y1


# In[137]:


#dividing data in testing and training data


# In[138]:


from sklearn.model_selection import train_test_split
x1_train,x1_test,y1_train,y1_test=train_test_split(x1,y1,test_size=0.3,random_state=0)


# In[139]:


x1_train.shape


# In[140]:


x1_test.shape


# In[141]:


y1_train.shape


# In[142]:


y1_test.shape


# In[143]:


#calculating mean of y1_train


# In[144]:


lmean=np.mean(y1_train)


# In[145]:


lmean


# In[146]:


#converting y1_train and y1_test as binary values


# In[147]:


y1_train[y1_train<lmean]=0


# In[148]:


y1_train[y1_train>lmean]=1


# In[149]:


y1_train


# In[150]:


y1_test[y1_test<lmean]=0


# In[151]:


y1_test[y1_test>lmean]=1


# In[152]:


y1_test


# In[153]:


#NAIVE BAYES


# In[155]:


from sklearn.naive_bayes import GaussianNB 
gnb2 = GaussianNB() 
gnb2.fit(x1_train,y1_train.ravel())
y1_pred = gnb2.predict(x1_test)
from sklearn import metrics 
from sklearn.metrics import f1_score
naive2_acc=metrics.accuracy_score(y1_test, y1_pred)*100
naive2_f1 = f1_score(y1_test, y1_pred, average='binary')*100
print('F-Measure: ',naive2_f1)
print("Gaussian Naive Bayes model accuracy(in %):",naive2_acc)


# In[ ]:


#DECISION TREE


# In[156]:


from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import f1_score
dt2 = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dt2.fit(x1_train, y1_train.ravel())
dt2_pred = dt2.predict(x1_test)
dt2_acc=metrics.accuracy_score(y1_test, dt2_pred)*100
dt2_f1 = f1_score(y1_test, dt2_pred, average='binary')*100
print('F-Measure: ',dt2_f1)
print("Gaussian Naive Bayes model accuracy(in %):",dt2_acc)


# In[ ]:


#RANDOM FOREST


# In[157]:


from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import f1_score
rfc2 = RandomForestClassifier(criterion = 'entropy', random_state = 42)
rfc2.fit(x1_train, y1_train.ravel())
rfc2_pred = rfc2.predict(x1_test)
rfc2_acc=metrics.accuracy_score(y1_test, rfc2_pred)*100
rfc2_f1 = f1_score(y1_test, rfc2_pred, average='binary')*100
print('F-Measure: ',rfc2_f1)
print("Gaussian Naive Bayes model accuracy(in %):",rfc2_acc)


# In[158]:


#SUPPORT VECTOR MACHINE


# In[166]:


from sklearn import svm
from sklearn import metrics
from sklearn.metrics import f1_score


# In[167]:


cls2=svm.SVC(kernel="rbf",gamma="auto")


# In[168]:


cls2.fit(x1_train,y1_train.ravel())


# In[169]:


svm_pred1=cls2.predict(x1_test)


# In[170]:


svm2_acc=metrics.accuracy_score(y1_test, svm_pred1)*100
svm2_f1 = f1_score(y1_test, svm_pred1, average='binary')*100
print('F-Measure: ',svm2_f1)
print("Gaussian Naive Bayes model accuracy(in %):",svm2_acc)


# In[171]:


import matplotlib.pyplot as plt


# In[172]:


plt.plot(naive2_acc,naive2_f1,'^')
plt.plot(svm2_acc,svm2_f1,'+')
plt.plot(dt2_acc,dt2_f1,'o')
plt.plot(rfc2_acc,rfc2_f1,'*')


# In[ ]:




