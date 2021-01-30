#!/usr/bin/env python
# coding: utf-8

# # UCI

# # Drug Review Dataset 

# In[ ]:


import pandas as pd


# In[2]:


data_train = pd.read_csv('.....\\drugsCom_raw\\drugsComTrain_raw.tsv',delimiter='\t')
data_test = pd.read_csv('......\\drugsCom_raw\\drugsComTest_raw.tsv' ,delimiter='\t')


# In[ ]:





# In[3]:


df = pd.concat([data_train,data_test])  # combine the two dataFrames into one for a bigger data size and ease of preprocessing


# In[4]:


data_train.shape


# In[5]:


data_test.shape


# In[6]:


df.head()


# In[7]:


df.columns = ['Id','drugName','condition','review','rating','date','usefulCount']    #rename columns


# In[8]:


df.head()


# In[9]:


df['date'] = pd.to_datetime(df['date'])    #convert date to datetime eventhough we are not using date in this


# In[10]:


df['date'].head()             #confirm conversion


# In[11]:


df2 = df[['Id','review','rating']].copy()    # create a new dataframe with just review and rating for sentiment analysis


# In[12]:


df.head()             #confirm conversion


# In[13]:


df2.head()


# In[14]:


df2.isnull().any().any()    # check for null


# In[15]:


df2.info(null_counts=True)         #another way to check for null


# In[16]:


df2.info()       #check for datatype, also shows null


# In[17]:


df2['Id'].unique()       # shows unique Id as array


# In[18]:


df2['Id'].count()      #count total number of items in the Id column


# In[19]:


df2['Id'].nunique()     #shows unique Id values


# In[20]:


df['review'][1]         # access indivdual value


# In[21]:


df.review[1]            # another method to assess individual value in a Series


# In[22]:


import nltk
nltk.download(['punkt','stopwords'])


# In[23]:


from nltk.corpus import stopwords
stopwords = stopwords.words('english')


# In[24]:


df2['cleanReview'] = df2['review'].apply(lambda x: ' '.join([item for item in x.split() if item not in stopwords]))     # remove stopwords from review


# In[26]:


df2['cleanReview'] = df2['review'].apply(lambda x: ' '.join([item for item in x.split() if item not in stopwords]))     # remove stopwords from review


# In[56]:


import vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()


# In[57]:


df2['vaderReviewScore'] = df2['cleanReview'].apply(lambda x: analyzer.polarity_scores(x)['compound'])


# In[59]:


positive_num = len(df2[df2['vaderReviewScore'] >=0.05])
neutral_num = len(df2[(df2['vaderReviewScore'] >-0.05) & (df2['vaderReviewScore']<0.05)])
negative_num = len(df2[df2['vaderReviewScore']<=-0.05])


# In[60]:


positive_num,neutral_num, negative_num


# In[61]:


df2['vaderSentiment']= df2['vaderReviewScore'].map(lambda x:int(2) if x>=0.05 else int(1) if x<=-0.05 else int(0) )


# In[62]:


df2['vaderSentiment'].value_counts()


# In[63]:


Total_vaderSentiment = positive_num + neutral_num + negative_num
Total_vaderSentiment


# In[64]:


df2.loc[df2['vaderReviewScore'] >=0.05,"vaderSentimentLabel"] ="positive"
df2.loc[(df2['vaderReviewScore'] >-0.05) & (df2['vaderReviewScore']<0.05),"vaderSentimentLabel"]= "neutral"
df2.loc[df2['vaderReviewScore']<=-0.05,"vaderSentimentLabel"] = "negative"


# In[65]:


df2.shape


# In[66]:


positive_rating = len(df2[df2['rating'] >=7.0])
neutral_rating = len(df2[(df2['rating'] >=4) & (df2['rating']<7)])
negative_rating = len(df2[df2['rating']<=3])


# In[67]:


positive_rating,neutral_rating,negative_rating


# In[68]:


Total_rating = positive_rating+neutral_rating+negative_rating
Total_rating


# In[69]:


df2['ratingSentiment']= df2['rating'].map(lambda x:int(2) if x>=7 else int(1) if x<=3 else int(0) )


# In[70]:


df2['ratingSentiment'].value_counts()


# In[72]:


df2.loc[df2['rating'] >=7.0,"ratingSentimentLabel"] ="positive"
df2.loc[(df2['rating'] >=4.0) & (df2['rating']<7.0),"ratingSentimentLabel"]= "neutral"
df2.loc[df2['rating']<=3.0,"ratingSentimentLabel"] = "negative"


# In[98]:


df2 = df2[['Id','review','cleanReview','rating','ratingSentiment','ratingSentimentLabel','vaderReviewScore','vaderSentimentLabel','vaderSentiment']]


# # =============================

# In[104]:


data_df=df2.drop(['review','cleanReview'],axis=1)


# In[149]:


data_df.head()


# In[150]:


data_df.info()


# In[145]:


#data_df=df2.drop(['ratingSentimentLabel'],axis=1)


# In[169]:


from sklearn.preprocessing import LabelEncoder


# In[188]:


encoder = LabelEncoder()
data_cat = data_df["review"]
data_cat_encod = encoder.fit_transform(data_cat)
data_cat_encod = pd.DataFrame(data_cat_encod,columns=["review"])
data_cat_encod.head()


# In[152]:


encoder = LabelEncoder()
data_cat = data_df["cleanReview"]
data_cat_encod = encoder.fit_transform(data_cat)
data_cat_encod = pd.DataFrame(data_cat_encod,columns=["cleanReview"])
data_cat_encod.head()


# In[153]:


encoder = LabelEncoder()
data_cat = data_df["vaderSentimentLabel"]
data_cat_encod = encoder.fit_transform(data_cat)
data_cat_encod = pd.DataFrame(data_cat_encod,columns=["vaderSentimentLabel"])
data_cat_encod.head()


# In[189]:


encoder


# In[148]:


#df2.to_csv('processed.csv')    # To save preprocessed dataset to csv


# In[ ]:





# In[ ]:





# In[78]:


import os
#os.stat('processed.csv').st_size         # Check size of csv file About 181MB


# In[ ]:





# In[79]:


df2.info()


# In[80]:


#df2.to_csv('processed.csv.gz',compression='gzip')


# In[53]:


#os.stat('processed.csv.gz').st_size    #compressed to about 54MB


# In[54]:


df2.head()


# In[ ]:


dfcopy = df2.copy()

