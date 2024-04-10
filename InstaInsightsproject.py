#!/usr/bin/env python
# coding: utf-8

# In[6]:


# Import necessary libraries
#Nisha tyagi
get_ipython().system('pip install wordcloud Wordcloud')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



# In[7]:


# Load the data

data = pd.read_csv('instagram_data.csv', encoding='ISO-8859-1')
df = pd.DataFrame(data)
df


# In[8]:


# Check for missing values
data.isnull().sum()




# In[9]:


data.info()


# In[10]:


# Plot distribution of Impressions from different sources
plt.figure(figsize=(10, 8))
plt.style.use('fivethirtyeight')
plt.title("Distribution of Impressions From Home")
sns.distplot(data['From Home'])
plt.show()



# In[11]:


plt.figure(figsize=(10, 8))
plt.title("Distribution of Impressions From Hashtags")
sns.distplot(data['From Hashtags'])
plt.show()



# In[12]:


plt.figure(figsize=(10, 8))
plt.title("Distribution of Impressions From Explore")
sns.distplot(data['From Explore'])
plt.show()



# In[13]:


# Plot pie chart for Impressions from different sources using Plotly
home = data["From Home"].sum()
hashtags = data["From Hashtags"].sum()
explore = data["From Explore"].sum()
other = data["From Other"].sum()

labels = ['From Home', 'From Hashtags', 'From Explore', 'Other']
values = [home, hashtags, explore, other]

fig = px.pie(data, values=values, names=labels, title="Impressions on Instagram Posts From Various Sources",hole=0.5)
fig.show()



# In[14]:


# Generate and plot word cloud for captions and hashtags
text = " ".join(i for i in data.Caption)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()



# In[15]:


text_hashtags = " ".join(i for i in data.Hashtags)
stopwords_hashtags = set(STOPWORDS)
wordcloud_hashtags = WordCloud(stopwords=stopwords_hashtags, background_color="white").generate(text_hashtags)
plt.figure(figsize=(12, 10))
plt.imshow(wordcloud_hashtags, interpolation="bilinear")
plt.axis("off")
plt.show()



# In[16]:


# Scatter plots for relationships between variables using Plotly Express
fig = px.scatter(data_frame=data, x="Impressions", y="Likes", size="Likes", trendline="ols", title="Relationship Between Likes and Total Impressions")
fig.show()




# In[17]:


fig = px.scatter(data_frame=data, x="Impressions", y="Comments", size="Comments", trendline="ols", title="Relationship Between Comments and Total Impressions")
fig.show()



# In[18]:


fig = px.scatter(data_frame=data, x="Impressions", y="Shares", size="Shares", trendline="ols", title="Relationship Between Shares and Total Impressions")
fig.show()



# In[19]:


fig = px.scatter(data_frame=data, x="Impressions", y="Saves", size="Saves", trendline="ols", title="Relationship Between Post Saves and Total Impressions")
fig.show()



# In[20]:


# Calculate and print correlation values for different variables
correlation = data.corr()
print(correlation["Impressions"].sort_values(ascending=False))



# In[21]:


# Calculate and print conversion rate
conversion_rate = (data["Follows"].sum() / data["Profile Visits"].sum()) * 100
print(conversion_rate)



# In[22]:


# Scatter plot for relationship between Profile Visits and Followers Gained
fig = px.scatter(data_frame=data, x="Profile Visits", y="Follows", size="Follows", trendline="ols", title="Relationship Between Profile Visits and Followers Gained")
fig.show()



# In[23]:


# Train a PassiveAggressiveRegressor model
x = np.array(data[['Likes', 'Saves', 'Comments', 'Shares', 'Profile Visits', 'Follows']])
y = np.array(data["Impressions"])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

model = PassiveAggressiveRegressor()
model.fit(xtrain, ytrain)
score = model.score(xtest, ytest)
print(score)


# In[24]:


# Predict with the trained model
features = np.array([[282.0, 233.0, 4.0, 9.0, 165.0, 54.0]])
prediction = model.predict(features)
print(prediction)


# In[ ]:





# In[ ]:




