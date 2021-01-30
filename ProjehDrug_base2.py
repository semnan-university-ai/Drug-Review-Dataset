#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from  sklearn  import tree
from sklearn import preprocessing
from sklearn import tree


# In[2]:


df = pd.read_csv('...........\\drugsCom_raw\\Process.csv')


# In[3]:


df.head()


# In[4]:


data=df.iloc[:,:-1]
data


# In[5]:


data_df=data.drop(['Unnamed: 0'],axis=1)


# In[6]:


data_df.head()


# In[7]:


x0 = data_df.iloc[ : , 0 :-1]
x0


# In[8]:


y0 = df['vaderTarget']
y0


# In[9]:


l1 = preprocessing.LabelEncoder()
l1.fit(['neutral', 'positive','negative'])


# In[10]:


x = x0
x.iloc[:,3] = l1.transform(x0.iloc[:,3])
x0


# In[11]:


l0 = preprocessing.LabelEncoder()
l0.fit(['neutral', 'positive','negative'])
y= l0.transform(y0)


# In[12]:


y


# In[13]:


x0


# In[14]:


clf = tree.DecisionTreeClassifier()


# In[15]:


clf_fit =clf.fit(x,y)
tree.plot_tree(clf_fit)


# In[16]:


clf = tree.DecisionTreeClassifier(criterion='entropy')
clf_fit =clf.fit(x,y)
tree.plot_tree(clf_fit)


# In[ ]:





# In[17]:


data_bayes=x0


# # ================================Bayse

# In[18]:


import nltk
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
import numpy
data_bayes = numpy.random.rand(100, 5)
numpy.random.shuffle(data_bayes)
training, test = data_bayes[:80,:], data_bayes[80:,:]
print(training,test)

classifier = nltk.NaiveBayesClassifier.train(training)

print("Naive Bayes Algo Accuracy:", (nltk.classify.accuracy(classifier,test))*100)


# In[ ]:


#============================


# # ================================KNN

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


# In[ ]:


dataset=x0
print(len(dataset))
dataset.head(22)


# In[ ]:


zero_not_accepted =['Id','rating','ratingSentiment','ratingSentimentLabel','vaderReviewScore']


# In[ ]:


for column in zero_not_accepted:
    dataset[column]=dataset[column].replace(0,np.NaN)
    mean = int(dataset[column].mean(skipna=True))
    dataset[column] = dataset[column].replace(np.NAN,mean)


# In[ ]:


print(dataset['Id'])


# In[ ]:


x=dataset.iloc[:,0:5]
y=dataset.iloc[:, 4]
x_train , x_test , y_train , y_test =train_test_split(x,y,random_state=0 ,test_size=0.2)


# In[ ]:


sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)


# In[ ]:


x_train


# In[ ]:


x_test


# In[ ]:


y_train


# In[ ]:


y_test


# In[ ]:


import math


# In[ ]:


math.sqrt(len(y_train))


# In[ ]:


math.sqrt(len(y_test))


# In[ ]:


#classifier = KNeighborsClassifier(n_neighbors=13,p=2,metric='euclidean')
#classifier.fit(x_train,y_train)
#cm = confusion_matrix(y_test , y_pred)
#print(cm)


# # ================================MLP

# In[ ]:


from sklearn.neural_network import MLPClassifier

from sklearn.datasets import make_classification

from sklearn.model_selection import train_test_split


X, y = make_classification(n_samples=100, random_state=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,random_state=1)


clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)

clf.predict_proba(X_test[:1])

clf.predict(X_test[:5, :])


clf.score(X_test, y_test)


# # ========================================================

# In[ ]:


# Import required libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score


# In[ ]:


print(x0.shape)
x0.describe().transpose()


# In[ ]:


x0


# In[ ]:


target_column = ['vaderReviewScore'] 
predictors = list(set(list(x0.columns))-set(target_column))
x0[predictors] = x0[predictors]/x0[predictors].max()
x0.describe().transpose()


# In[ ]:


X = x0[predictors].values
y = x0[target_column].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)
print(X_train.shape); print(X_test.shape)


# In[ ]:


from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)
mlp.fit(X_train,y_train)

predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_train,predict_train))
print(classification_report(y_train,predict_train))


# # =======================Logistic Regression

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_csv('...............\\Proj_paython_drug\\processed.csv.gz')

train.head()


# In[ ]:


train.isnull()


# In[5]:


sns.heatmap(train.isnull())


# In[6]:


sns.countplot(x='rating',data=train)


# In[ ]:


sns.countplot(x='review',hue='rating',data=train)


# In[ ]:


plt.figure(figsize=(12, 7))
sns.boxplot(x='rating',y='ratingSentimentLabel',data=train,palette='winter')


# In[ ]:


def impute_age(cols):
    Id = cols[0]
    rating = cols[1]
    if pd.isnull(Id):
        
        if rating == 1:
            return 37
        elif rating == 2:
            return 29
        else:
            return 24
    else:
        return Id


# In[ ]:


train['Id'] = train[['Id','rating']].apply(impute_age,axis=1)


# In[ ]:


#sns.heatmap(train.isnull(),yticklabels=False,cbar=False)


# In[ ]:


train.drop('ratingSentiment',axis=1,inplace=True)


# In[ ]:


train.info()


# In[ ]:


train


# In[ ]:


#review = pd.get_dummies(train['review'],drop_first=True)
#cleanReview = pd.get_dummies(train['cleanReview'],drop_first=True)


# In[ ]:


train.drop(['Id','review','cleanReview'],axis=1,inplace=True)


# In[ ]:


#train = pd.concat([train,sex,embark],axis=1)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.drop('vaderSentiment',axis=1), train['vaderSentiment'], test_size=0.30, random_state=101)


# In[ ]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))

