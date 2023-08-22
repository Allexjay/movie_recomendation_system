#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[24]:


data=pd.read_csv('movies.csv')
data.head()


# In[25]:


data.columns


# In[26]:


data['title'].head()


# In[27]:


selected_column=['genres','keywords','original_title','tagline','cast','director']


# In[28]:


for feature in selected_column:
    data[feature]=data[feature].fillna(' ')


# In[29]:


combined_features= data['genres']+data['keywords']+data['original_title']+data['tagline']+data['cast']+data['director']
combined_features


# In[30]:


vectorizer= TfidfVectorizer()


# In[31]:


feature_vectorizer=vectorizer.fit_transform(combined_features)
print(feature_vectorizer)


# In[32]:


similarity=cosine_similarity(feature_vectorizer)
print(similarity)


# In[33]:


similarity.shape


# In[34]:


def movie_name():
    name=input('Enter the name you want to search:')
    all_movie_name=data['title'].tolist()
    closest_match=difflib.get_close_matches(name,all_movie_name)
    closest_match=closest_match[0]
    index=data[data['title']==closest_match]['index'].values[0]
    similarity_score=list(enumerate(similarity[index]))
    similarity_score=sorted(similarity_score,key=lambda x:x[1],reverse=True)
    j=1
    for i in similarity_score[:20]:
        print(j,data[data['index']==i[0]]['title'].values[0])
        j+=1


# In[40]:


choice=input("Enter your choice:\n\nSearching by\t\tType\nmovie name\t------\t1\ndirector name\t------\t2\ncast name\t------\t3\nmovie rating\t------\t4\nmovie genre\t------\t5\n\n")
if(choice=='1'):
    movie_name()
elif(choice=='2'):
    director_name()
elif(choice=='3'):
    cast_name()
elif(choice=='4'):
    movie_rating()
elif(choice=='5'):
    movie_genre()

