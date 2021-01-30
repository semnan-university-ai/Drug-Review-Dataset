#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('processed.csv.gz')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df = df.drop(columns=df.columns[0])


# In[6]:


df.head()


# In[7]:


df.groupby('vaderSentimentLabel').size()


# In[8]:


import matplotlib.pyplot as plt


# In[9]:



df.groupby('vaderSentimentLabel').count().plot.bar()
plt.show()


# In[10]:


df.groupby('ratingSentimentLabel').size()


# In[11]:


df.groupby('ratingSentimentLabel').count().plot.bar()
plt.show()


# In[12]:


df.groupby('ratingSentiment').size()


# In[13]:


positive_vader_sentiments = df[df.ratingSentiment == 2]
positive_string = []
for s in positive_vader_sentiments.cleanReview:
  positive_string.append(s)
positive_string = pd.Series(positive_string).str.cat(sep=' ')


# In[14]:


from wordcloud import WordCloud
wordcloud = WordCloud(width=2000,height=1000,max_font_size=200).generate(positive_string)
plt.imshow(wordcloud,interpolation='bilinear')
plt.show()


# In[15]:


for s in positive_vader_sentiments.cleanReview[:20]:
  if 'side effect' in s:
    print(s)


# In[16]:


negative_vader_sentiments = df[df.ratingSentiment == 1]
negative_string = []
for s in negative_vader_sentiments.cleanReview:
  negative_string.append(s)
negative_string = pd.Series(negative_string).str.cat(sep=' ')


# In[17]:


from wordcloud import WordCloud
wordcloud = WordCloud(width=2000,height=1000,max_font_size=200).generate(negative_string)
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.show()


# In[18]:


neutral_vader_sentiments = df[df.ratingSentiment == 0]
neutral_string = []
for s in neutral_vader_sentiments.cleanReview:
  neutral_string.append(s)
neutral_string = pd.Series(neutral_string).str.cat(sep=' ')


# In[19]:


from wordcloud import WordCloud
wordcloud = WordCloud(width=2000,height=1000,max_font_size=200).generate(neutral_string)
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.show()


# In[20]:


for s in neutral_vader_sentiments.cleanReview[:20]:
  if 'side effect' in s:
    print(s)


# In[21]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[22]:


tfidf = TfidfVectorizer(stop_words='english',ngram_range=(1,2))
features = tfidf.fit_transform(df.cleanReview)
labels   = df.vaderSentiment


# In[23]:


features.shape


# In[24]:



from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


# In[25]:


x_train,x_test,y_train,y_test = train_test_split(df['cleanReview'],df['ratingSentimentLabel'],random_state=0)


# In[26]:



from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


# In[27]:


models = [RandomForestClassifier(n_estimators=200,max_depth=3,random_state=0),LinearSVC(),MultinomialNB(),LogisticRegression(random_state=0,solver='lbfgs',max_iter=2000,multi_class='auto')]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model,features,labels,scoring='accuracy',cv=CV)
  for fold_idx,accuracy in enumerate(accuracies):
    entries.append((model_name,fold_idx,accuracy))
cv_df = pd.DataFrame(entries,columns=['model_name','fold_idx','accuracy'])


# In[28]:


cv_df


# In[29]:


cv_df.groupby('model_name').accuracy.mean()


# In[30]:


from sklearn.preprocessing import Normalizer


# In[31]:


model = LinearSVC('l2')
x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size=0.25,random_state=0)
normalize = Normalizer()
x_train = normalize.fit_transform(x_train)
x_test = normalize.transform(x_test)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)


# In[32]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


# In[33]:


from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test,y_pred)
conf_mat


# In[34]:


from mlxtend.plotting import plot_confusion_matrix


# In[35]:


fig,ax = plot_confusion_matrix(conf_mat=conf_mat,colorbar=True,show_absolute=True,cmap='viridis')


# In[36]:


from  sklearn.metrics import classification_report
print(classification_report(y_test,y_pred,target_names= df['ratingSentimentLabel'].unique()))


# In[37]:


df.head()


# In[38]:


df.info()


# In[39]:


y0 = df['vaderSentimentLabel']
y0

